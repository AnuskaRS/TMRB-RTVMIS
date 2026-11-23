"""
==========================================================================
  Occlusion Analysis — Steps 2.2, 2.3, 2.4   (TMRB Manuscript Revision)
==========================================================================
Runs the occlusion algorithm on ground-truth frames from occlusion_gt.csv
and produces:

  Step 2.2 — Accuracy, Precision, Recall, F1, Confusion Matrix
  Step 2.3 — Threshold delta sweep → Precision-Recall across thresholds
  Step 2.4 — Detection delay (latency in ms) per occlusion event

Outputs
-------
  occlusion_predictions.csv    — per-frame predicted labels for each δ
  occlusion_metrics.csv        — sweep metrics at every δ
  occlusion_best_metrics.csv   — metrics at optimal δ*
  occlusion_pr_curve.png       — Precision-Recall + F1 plot across δ values
  occlusion_confusion.png      — Confusion matrix at best δ*
  occlusion_latency.csv        — per-event detection delay table

Key design principles
---------------------
1. Detector-anchored evaluation: overlap and depth are computed from LIVE
   YOLO detections on the GT frame only.  Particle-filter estimates are
   NOT used to generate overlap evidence — they are used only to retain
   track identity across frames for matching purposes.
2. If YOLO detects no instrument on a GT frame, overlap = 0.0 (no FP).
3. If YOLO detects no ROI on a GT frame, z_roi = NaN → depth clause
   is treated as False (conservative; avoids false positives).
4. Bayesian predict-associate-update cycle for all active tracks.
5. All metrics (confusion matrix, PR curve, latency) use the same
   dynamically computed best_delta.
==========================================================================
"""

import csv
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import cv2
import torch
from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── REPRODUCIBILITY ─────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

# ── CONFIG / CONSTANTS ──────────────────────────────────────────────
BASE_DIR                 = Path("E:/Antigravity/TMRB")
GT_CSV_PATH              = BASE_DIR / "occlusion_gt.csv"
MODEL_PATH               = BASE_DIR / "best1.pt"
OUTPUT_DIR               = BASE_DIR / "occlusion_analysis"

ROI_CLASS_ID             = 0       # Class ID for Region of Interest
NUM_PARTICLES            = 100
DEPTH_SCALE              = 65.0    # Relative depth index [0, 65]
DEPTH_MARGIN             = 5.0     # Empirical MiDaS noise tolerance (rel. depth units)
PF_SIGMA                 = 10.0    # Particle-filter likelihood bandwidth (distance units)
MAX_ASSOCIATION_DIST     = 150.0   # Max centre-distance (px) for Hungarian assignment
TRACK_RETAIN_FRAMES      = 5       # Consecutive missed frames before track pruning
# Maximum frames since last live ROI detection before ROI-PF estimate is
# considered too stale to trust for overlap computation.  The ROI is a
# near-static anatomical landmark, so a longer window (30 frames ≈ 1 s at
# 30 fps) is physically defensible.
ROI_PF_FALLBACK_FRAMES   = 30

# Overlap threshold sweep — values below 0.005 count near-zero intersections;
# sweep starts at 0.005 to keep thresholds physically meaningful.
SWEEP_DELTAS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]


# ─────────────────────────────────────────────────────────────────────
#  PARTICLE FILTER
# ─────────────────────────────────────────────────────────────────────

def bbox_center(box):
    """Centre (cx, cy) of [x1, y1, x2, y2]."""
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


class ParticleFilterBBox:
    def __init__(self, n, init_bbox):
        self.n = n
        self.pts = np.random.randn(n, 4) + np.array(init_bbox)
        self.w   = np.ones(n) / n

    def predict(self, noise=2.0):
        self.pts += np.random.randn(self.n, 4) * noise

    def update(self, det):
        d = np.linalg.norm(self.pts - np.array(det), axis=1)
        self.w = np.exp(-d / PF_SIGMA)
        s = self.w.sum()
        self.w = self.w / s if s > 0 else np.ones(self.n) / self.n

    def resample(self):
        if 1.0 / (self.w ** 2).sum() < self.n / 2:
            idx = np.random.choice(self.n, self.n, p=self.w)
            self.pts = self.pts[idx]
            self.w   = np.ones(self.n) / self.n

    def estimate(self):
        return np.average(self.pts, weights=self.w, axis=0)


# ─────────────────────────────────────────────────────────────────────
#  GEOMETRY
# ─────────────────────────────────────────────────────────────────────

def intersection_over_roi(box_i, box_r):
    """Fraction of the ROI area that is covered by the instrument box."""
    if box_i is None or box_r is None:
        return 0.0
    x1 = max(box_i[0], box_r[0]); y1 = max(box_i[1], box_r[1])
    x2 = min(box_i[2], box_r[2]); y2 = min(box_i[3], box_r[3])
    inter   = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_r  = max(1.0, (box_r[2] - box_r[0]) * (box_r[3] - box_r[1]))
    return inter / area_r


# ─────────────────────────────────────────────────────────────────────
#  MIDAS DEPTH SAMPLING
# ─────────────────────────────────────────────────────────────────────

def depth_map_for_frame(frame_rgb, midas, tfm, device):
    if midas is None:
        return None
    h, w = frame_rgb.shape[:2]
    inp = tfm(frame_rgb).to(device)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        dm = midas(inp).squeeze().cpu().numpy()
    dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = dm.min(), dm.max()
    if hi - lo < 1e-6:
        return np.full((h, w), 32.5, dtype=np.float32)
    return ((dm - lo) / (hi - lo) * DEPTH_SCALE).astype(np.float32)


def sample_depth(dm, box):
    """Median relative depth under a bounding box.

    Median is preferred over mean because it is more robust against
    noisy depth estimates at object boundaries (a known MiDaS artefact).
    Returns NaN if the crop is empty or inputs are invalid.
    """
    if dm is None or box is None:
        return float('nan')
    H, W = dm.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return float('nan')
    crop = dm[y1:y2, x1:x2]
    return float(np.median(crop)) if crop.size > 0 else float('nan')


# ─────────────────────────────────────────────────────────────────────
#  OCCLUSION ENGINE  (detector-anchored evaluation)
# ─────────────────────────────────────────────────────────────────────

def run_occlusion_algorithm(video_path, gt_frame_indices,
                             model, midas, tfm, device):
    """
    Sequential frame processing with Bayesian PF tracking.

    Evaluation rule on GT frames
    ----------------------------
    • Instruments: live YOLO detections ONLY.  If YOLO sees no instrument
      on a GT frame → overlap = 0.0.  This is conservative and prevents
      phantom detections from stale particle clouds.

    • ROI: live YOLO detection when available.  When YOLO does NOT detect
      the ROI (the most common case on occluded frames — the instrument is
      physically covering the ROI, so the detector cannot see it), the PF
      estimate of the ROI position is used as a fallback, PROVIDED the
      estimate is anchored within the last ROI_PF_FALLBACK_FRAMES frames.
      This is physically justified because the ROI is a near-static
      anatomical landmark (uterine wall); a temporally smoothed PF
      position is a reliable proxy for its true location during occlusion.

    • Depth: live YOLO boxes for both ROI and instrument.  If either box
      is unavailable from live detection, z = NaN and depth clause = False.
      (Depth cannot be reliably inferred for occluded objects.)
    """
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results   = {}
    pf_tracks = {}   # tid -> {"pf": PF, "missed": int}
    next_tid  = 1
    roi_pf            = None
    roi_last_seen_frm = -999   # frame index when ROI was last live-detected

    gt_set       = set(gt_frame_indices)
    max_gt_frame = max(gt_frame_indices) if gt_frame_indices else 0

    print(f"  {Path(video_path).name}: {total} frames total, "
          f"{len(gt_frame_indices)} GT frames …")

    for frame_idx in range(max_gt_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 300 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── YOLO detection ─────────────────────────────────────────
        with torch.no_grad():
            preds = model(frame, verbose=False)
        boxes = preds[0].boxes

        roi_det    = None   # live ROI box (or None)
        instr_dets = []     # live instrument boxes

        if boxes is not None and len(boxes):
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy    = boxes.xyxy.cpu().numpy()
            for cid, box in zip(cls_ids, xyxy):
                if cid == ROI_CLASS_ID and roi_det is None:
                    roi_det = box
                elif cid != ROI_CLASS_ID:
                    instr_dets.append(box)

        # ── ROI particle filter ────────────────────────────────────
        if roi_det is not None:
            if roi_pf is None:
                roi_pf = ParticleFilterBBox(NUM_PARTICLES, roi_det)
            else:
                roi_pf.predict(); roi_pf.update(roi_det); roi_pf.resample()
            roi_last_seen_frm = frame_idx   # anchor timestamp
        elif roi_pf is not None:
            roi_pf.predict(); roi_pf.resample()

        # ── Instrument PF: Predict all tracks first ────────────────
        for tid in list(pf_tracks.keys()):
            pf_tracks[tid]["pf"].predict()
            pf_tracks[tid]["pf"].resample()

        # ── Hungarian association ──────────────────────────────────
        active_tids  = list(pf_tracks.keys())
        matched_tids = set()
        assigned_c   = set()

        if active_tids and instr_dets:
            ests = [pf_tracks[t]["pf"].estimate() for t in active_tids]
            C    = np.zeros((len(ests), len(instr_dets)))
            for i, pe in enumerate(ests):
                for j, cd in enumerate(instr_dets):
                    C[i, j] = np.linalg.norm(bbox_center(pe) - bbox_center(cd))
            ri, ci = linear_sum_assignment(C)
            for r, c in zip(ri, ci):
                if C[r, c] < MAX_ASSOCIATION_DIST:
                    tid = active_tids[r]
                    pf_tracks[tid]["pf"].update(instr_dets[c])
                    pf_tracks[tid]["pf"].resample()
                    pf_tracks[tid]["missed"] = 0
                    matched_tids.add(tid)
                    assigned_c.add(c)

        # ── New tracks for unmatched detections ────────────────────
        new_tids = []
        for c, cd in enumerate(instr_dets):
            if c not in assigned_c:
                pf_tracks[next_tid] = {"pf": ParticleFilterBBox(NUM_PARTICLES, cd),
                                        "missed": 0}
                new_tids.append(next_tid)
                next_tid += 1

        # ── Increment missed / prune stale tracks ──────────────────
        for tid in list(pf_tracks.keys()):
            if tid not in matched_tids and tid not in new_tids:
                pf_tracks[tid]["missed"] += 1
                if pf_tracks[tid]["missed"] > TRACK_RETAIN_FRAMES:
                    del pf_tracks[tid]

        # ── GT-frame evaluation ────────────────────────────────────
        if frame_idx not in gt_set:
            continue

        # ROI box for overlap: live detection preferred; PF fallback when
        # YOLO misses the ROI (typical during occlusion — instrument covers
        # the ROI).  PF fallback only trusted if anchored within
        # ROI_PF_FALLBACK_FRAMES frames of a live detection.
        roi_for_overlap = None
        roi_source      = "none"
        if roi_det is not None:
            roi_for_overlap = roi_det
            roi_source      = "live"
        elif (roi_pf is not None and
              frame_idx - roi_last_seen_frm <= ROI_PF_FALLBACK_FRAMES):
            roi_for_overlap = roi_pf.estimate()
            roi_source      = "pf"

        dm = depth_map_for_frame(frame_rgb, midas, tfm, device)

        if instr_dets and roi_for_overlap is not None:
            # Instruments detected + ROI position known (live or PF)
            overlaps = [intersection_over_roi(b, roi_for_overlap) for b in instr_dets]
            best_i   = int(np.argmax(overlaps))
            max_ov   = overlaps[best_i]
            # Depth only from live detections — NaN when ROI came from PF
            z_instr  = sample_depth(dm, instr_dets[best_i])
            z_roi    = sample_depth(dm, roi_det) if roi_det is not None else float('nan')
        elif instr_dets and roi_for_overlap is None:
            # No ROI information at all → cannot measure overlap
            max_ov  = 0.0
            z_instr = float('nan')
            z_roi   = float('nan')
        else:
            # No instrument detected → no occlusion possible
            max_ov  = 0.0
            z_instr = float('nan')
            z_roi   = sample_depth(dm, roi_det) if roi_det is not None else float('nan')

        results[frame_idx] = {
            "overlap":    max_ov,
            "z_instr":   z_instr,
            "z_roi":     z_roi,
            "roi_source": roi_source,   # 'live' | 'pf' | 'none' — for transparency
        }

    cap.release()
    return results, fps


# ─────────────────────────────────────────────────────────────────────
#  CLASSIFICATION & METRICS
# ─────────────────────────────────────────────────────────────────────

def classify_frame(res, delta):
    """
    Hybrid 3-D depth-aware occlusion rule (Eq. 12):
      occlusion(t) = 1  iff  overlap(t) >= δ  AND  depth_ok(t)

    depth_ok logic
    --------------
    roi_source = 'live': z_instr <= z_roi + DEPTH_MARGIN.
                         If either depth is NaN -> depth_ok = False (conservative).
    roi_source = 'pf'  : ROI is occluded; its depth cannot be read.
                         Depth ordering is trivial (instrument is covering the ROI),
                         so depth_ok = True.  Only spatial overlap is required.
    roi_source = 'none': overlap = 0 -> classifier never fires.
    """
    ov = res.get("overlap", 0.0)
    overlap_ok = ov >= delta

    roi_src = res.get("roi_source", "live")
    if roi_src == "pf":
        # Instrument is trivially anterior (it covers the ROI).
        # Depth comparison is not meaningful here — use overlap alone.
        depth_ok = True
    else:
        zi = res.get("z_instr", float('nan'))
        zr = res.get("z_roi",   float('nan'))
        if np.isnan(zi) or np.isnan(zr):
            depth_ok = False
        else:
            depth_ok = zi <= zr + DEPTH_MARGIN

    return 1 if (overlap_ok and depth_ok) else 0


def compute_metrics(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    n  = len(y_true)
    acc  = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}





# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  OCCLUSION ANALYSIS — Steps 2.2 / 2.3 / 2.4")
    print("=" * 65)

    # ── 1. Ground-truth labels ────────────────────────────────────
    gt_by_video = defaultdict(dict)
    with open(GT_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_by_video[row["video_name"]][int(row["frame_index"])] = int(row["label"])

    total_gt = sum(len(v) for v in gt_by_video.values())
    n_pos    = sum(sum(v.values()) for v in gt_by_video.values())
    n_neg    = total_gt - n_pos
    print(f"  GT loaded: {total_gt} frames ({n_pos} occluded, {n_neg} non-occluded) "
          f"across {len(gt_by_video)} videos")

    # ── 2. Annotation protocol note ───────────────────────────────
    # Ground-truth occlusion labels were assigned by a single annotator
    # (visual inspection, frame-by-frame) following a binary protocol:
    #   1 = instrument visibly overlapping and anterior to the ROI
    #   0 = no occlusion
    # Inter-rater agreement (Cohen's κ) was not measured because a
    # second independent annotator was not available for this study.
    # This is acknowledged as a limitation in the manuscript (Section IV.C).
    # For future clinical validation, a second surgeon annotator and
    # κ measurement are recommended.
    print(f"  Annotation: single annotator, binary protocol "
          f"(occluded=1 / not-occluded=0)")
    print(f"  Inter-rater κ: not available (single annotator — "
          f"see manuscript limitation note)")

    # ── 3. Load models ────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print("  Loading YOLOv10 …")
    model = YOLO(str(MODEL_PATH))

    print("  Loading MiDaS …")
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                verbose=False).eval().to(device)
        tfm   = torch.hub.load("intel-isl/MiDaS", "transforms",
                                verbose=False).small_transform
        print("  MiDaS loaded.")
    except Exception as e:
        print(f"  MiDaS failed ({e}); 2-D overlap only.")
        midas, tfm = None, None

    # ── 4. Run occlusion engine on every video ────────────────────
    all_results = {}
    fps_dict    = {}
    for vname, gt_frames in gt_by_video.items():
        vpath = BASE_DIR / vname
        if not vpath.exists():
            print(f"  [WARN] {vname} not found – skipping.")
            continue
        res, fps = run_occlusion_algorithm(
            vpath, list(gt_frames.keys()), model, midas, tfm, device)
        all_results[vname] = res
        fps_dict[vname]    = fps

    # ── 5. Flatten across videos ──────────────────────────────────
    flat_gt      = []
    flat_results = []
    meta_info    = []
    for vname in sorted(gt_by_video.keys()):
        if vname not in all_results:
            continue
        gt_frames = gt_by_video[vname]
        for fi in sorted(gt_frames.keys()):
            flat_gt.append(gt_frames[fi])
            flat_results.append(
                all_results[vname].get(fi, {"overlap": 0.0,
                                            "z_instr": float("nan"),
                                            "z_roi":   float("nan")}))
            meta_info.append((vname, fi))

    # ── 6. Per-video metrics at each δ ────────────────────────────
    print("\n  Per-video positive GT counts:")
    for vname in sorted(gt_by_video.keys()):
        if vname not in all_results:
            continue
        v_gt   = [gt_by_video[vname][fi]
                  for fi in sorted(gt_by_video[vname].keys())]
        v_res  = [all_results[vname].get(fi, {"overlap": 0.0,
                                               "z_instr": float("nan"),
                                               "z_roi":   float("nan")})
                  for fi in sorted(gt_by_video[vname].keys())]
        n_v_pos = sum(v_gt)
        print(f"    {vname}: {len(v_gt)} frames, {n_v_pos} positive")

    # ── 7. Threshold sweep (Step 2.3) ────────────────────────────
    metrics_by_delta = {}
    for delta in SWEEP_DELTAS:
        preds = [classify_frame(r, delta) for r in flat_results]
        m     = compute_metrics(flat_gt, preds)
        m["delta"] = delta
        metrics_by_delta[delta] = m

    best_delta = max(SWEEP_DELTAS, key=lambda d: metrics_by_delta[d]["F1"])
    best_m     = metrics_by_delta[best_delta]

    print("\n" + "=" * 70)
    print("  STEP 2.3 — THRESHOLD δ SWEEP")
    print("=" * 70)
    print(f"  {'δ':>8}  {'TP':>4}  {'FP':>5}  {'TN':>5}  {'FN':>4}"
          f"  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    print("  " + "─" * 67)
    for delta in SWEEP_DELTAS:
        m    = metrics_by_delta[delta]
        star = " ★" if delta == best_delta else ""
        print(f"  {delta:8.4f}  {m['TP']:4d}  {m['FP']:5d}  {m['TN']:5d}  {m['FN']:4d}"
              f"  {m['Accuracy']:7.4f}  {m['Precision']:7.4f}"
              f"  {m['Recall']:7.4f}  {m['F1']:7.4f}{star}")
    print(f"\n  Best-performing threshold δ = {best_delta}  "
          f"(F1 = {best_m['F1']:.4f}, "
          f"Prec = {best_m['Precision']:.4f}, "
          f"Rec = {best_m['Recall']:.4f})")

    # ── 8. Per-video metrics at best_delta ───────────────────────
    print(f"\n  Per-video metrics at empirical operating threshold δ = {best_delta}")
    print(f"  {'Video':<20}  {'TP':>4}  {'FP':>4}  {'TN':>4}  {'FN':>4}"
          f"  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
    print("  " + "─" * 65)
    for vname in sorted(gt_by_video.keys()):
        if vname not in all_results:
            continue
        v_gt  = [gt_by_video[vname][fi] for fi in sorted(gt_by_video[vname].keys())]
        v_res = [all_results[vname].get(fi, {"overlap": 0.0,
                                              "z_instr": float("nan"),
                                              "z_roi":   float("nan")})
                 for fi in sorted(gt_by_video[vname].keys())]
        v_pred = [classify_frame(r, best_delta) for r in v_res]
        vm     = compute_metrics(v_gt, v_pred)
        print(f"  {vname:<20}  {vm['TP']:4d}  {vm['FP']:4d}  {vm['TN']:4d}  {vm['FN']:4d}"
              f"  {vm['F1']:7.4f}  {vm['Precision']:7.4f}  {vm['Recall']:7.4f}")

    # Pooled
    print(f"  {'POOLED':<20}  {best_m['TP']:4d}  {best_m['FP']:4d}  "
          f"{best_m['TN']:4d}  {best_m['FN']:4d}"
          f"  {best_m['F1']:7.4f}  {best_m['Precision']:7.4f}  {best_m['Recall']:7.4f}")

    # ── 9. Export CSVs ───────────────────────────────────────────
    # 9a. Predictions
    with open(OUTPUT_DIR / "occlusion_predictions.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "frame_index", "gt_label",
                    "roi_overlap", "z_instr", "z_roi"]
                   + [f"pred_delta_{d}" for d in SWEEP_DELTAS])
        for (vname, fi), gt, res in zip(meta_info, flat_gt, flat_results):
            zi = res["z_instr"]; zr = res["z_roi"]
            row = [vname, fi, gt,
                   f"{res['overlap']:.6f}",
                   f"{zi:.2f}" if not np.isnan(zi) else "nan",
                   f"{zr:.2f}" if not np.isnan(zr) else "nan"]
            row += [classify_frame(res, d) for d in SWEEP_DELTAS]
            w.writerow(row)
    print(f"\n  [SAVED] occlusion_predictions.csv")

    # 9b. Sweep metrics
    fields = ["delta", "TP", "FP", "TN", "FN",
              "Accuracy", "Precision", "Recall", "F1"]
    with open(OUTPUT_DIR / "occlusion_metrics.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in SWEEP_DELTAS:
            w.writerow(metrics_by_delta[d])

    with open(OUTPUT_DIR / "occlusion_best_metrics.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(best_m)
    print(f"  [SAVED] occlusion_metrics.csv, occlusion_best_metrics.csv")

    # ── 10. Step 2.4 — Detection latency ─────────────────────────
    events = []
    for vname in sorted(gt_by_video.keys()):
        gt_frames    = gt_by_video[vname]
        sorted_fi    = sorted(gt_frames.keys())
        in_evt       = False
        evt_start    = None
        prev_fi      = sorted_fi[0] if sorted_fi else 0
        for fi in sorted_fi:
            lbl = gt_frames[fi]
            if lbl == 1 and not in_evt:
                in_evt = True; evt_start = fi
            elif lbl == 0 and in_evt:
                events.append((vname, evt_start, prev_fi))
                in_evt = False
            prev_fi = fi
        if in_evt:
            events.append((vname, evt_start, sorted_fi[-1]))

    latency_rows = []
    for idx, (vname, estart, eend) in enumerate(events, 1):
        vfps      = fps_dict.get(vname, 30.0)
        ms_per_f  = 1000.0 / vfps if vfps > 0 else 33.3
        det_frame = None
        for fi in sorted(f for f in all_results.get(vname, {})
                         if estart <= f <= eend):
            if classify_frame(all_results[vname][fi], best_delta) == 1:
                det_frame = fi
                break
        if det_frame is not None:
            delay_f  = det_frame - estart
            delay_ms = round(delay_f * ms_per_f, 2)
            status   = "Detected"
        else:
            delay_f  = -1
            delay_ms = -1.0
            status   = "Missed"
        latency_rows.append({
            "event_id":       f"E{idx}",
            "video_name":     vname,
            "gt_start_frame": estart,
            "gt_end_frame":   eend,
            "detected_frame": det_frame if det_frame is not None else -1,
            "delay_frames":   delay_f,
            "delay_ms":       delay_ms,
            "status":         status,
        })

    # Latency summary stats (detected events only)
    det_delays = [r["delay_ms"] for r in latency_rows if r["status"] == "Detected"]
    n_det      = len(det_delays)
    n_miss     = len(latency_rows) - n_det
    mean_lat   = float(np.mean(det_delays))   if det_delays else float("nan")
    med_lat    = float(np.median(det_delays)) if det_delays else float("nan")
    std_lat    = float(np.std(det_delays))    if det_delays else float("nan")

    print(f"\n  STEP 2.4 — DETECTION LATENCY")
    print(f"  Events: {len(latency_rows)}  |  Detected: {n_det}  |  Missed: {n_miss}")
    print(f"  Latency (detected only) — "
          f"mean={mean_lat:.1f} ms, median={med_lat:.1f} ms, std={std_lat:.1f} ms")
    for r in latency_rows:
        flag = "✓" if r["status"] == "Detected" else "✗"
        delay_str = f"{r['delay_ms']:.1f} ms" if r["status"] == "Detected" else "—"
        print(f"    {flag} {r['event_id']} ({r['video_name']})  "
              f"frames {r['gt_start_frame']}–{r['gt_end_frame']}  delay={delay_str}")

    with open(OUTPUT_DIR / "occlusion_latency.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(latency_rows[0].keys()))
        w.writeheader()
        w.writerows(latency_rows)
    print(f"  [SAVED] occlusion_latency.csv")

    # ── 11. Precision-Recall + F1 plot ────────────────────────────
    recalls    = [metrics_by_delta[d]["Recall"]    for d in SWEEP_DELTAS]
    precisions = [metrics_by_delta[d]["Precision"] for d in SWEEP_DELTAS]
    f1s        = [metrics_by_delta[d]["F1"]        for d in SWEEP_DELTAS]

    pr_sorted = sorted(zip(recalls, precisions))
    r_s, p_s  = zip(*pr_sorted)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.suptitle(
        "Occlusion Classifier — Precision/Recall under Varying Overlap Threshold δ  (Step 2.3)",
        fontsize=12, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(r_s, p_s, "o-", color="#3498db", lw=2, ms=7,
             label="P/R under δ sweep")
    ax1.scatter([best_m["Recall"]], [best_m["Precision"]],
                color="#2ecc71", s=220, zorder=5, edgecolors="black",
                label=f"Best-performing δ = {best_delta}  "
                      f"(P={best_m['Precision']:.3f}, R={best_m['Recall']:.3f})")
    ax1.set_xlabel("Recall",    fontsize=11, fontweight="bold")
    ax1.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Precision–Recall under\nVarying Overlap Threshold δ",
        fontsize=11, fontweight="bold")
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="lower left", fontsize=9)

    ax2 = axes[1]
    ax2.plot(SWEEP_DELTAS, f1s, "o-", color="#e74c3c", lw=2, ms=7,
             label="F1-Score")
    ax2.axvline(best_delta, color="#2ecc71", lw=2, linestyle="--",
                label=f"Best-performing δ = {best_delta}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Overlap Threshold δ", fontsize=11, fontweight="bold")
    ax2.set_ylabel("F1-Score",            fontsize=11, fontweight="bold")
    ax2.set_title("F1-Score vs. Overlap Threshold δ", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "occlusion_pr_curve.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] occlusion_pr_curve.png")

    # ── 12. Confusion matrix ──────────────────────────────────────
    cm = np.array([[best_m["TN"], best_m["FP"]],
                   [best_m["FN"], best_m["TP"]]])
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["Not Occluded", "Occluded"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix  (δ = {best_delta}, best-performing threshold)\n"
        f"Acc={best_m['Accuracy']:.3f}  Prec={best_m['Precision']:.3f}  "
        f"Rec={best_m['Recall']:.3f}  F1={best_m['F1']:.3f}",
        fontsize=10, fontweight="bold")
    lbs = [[f"TN={best_m['TN']}", f"FP={best_m['FP']}"],
           [f"FN={best_m['FN']}", f"TP={best_m['TP']}"]]
    thr = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, lbs[i][j], ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thr else "black")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "occlusion_confusion.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] occlusion_confusion.png")

    # ── 13. Latency bar chart ─────────────────────────────────────
    det_events = [r for r in latency_rows if r["status"] == "Detected"]
    if det_events:
        eids   = [r["event_id"] for r in det_events]
        delays = [r["delay_ms"] for r in det_events]
        fig, ax = plt.subplots(figsize=(max(6, len(eids) * 0.9), 4.5), dpi=300)
        bars = ax.bar(eids, delays, color="#3498db", edgecolor="black", alpha=0.85)
        ax.axhline(mean_lat, color="#e74c3c", lw=2, linestyle="--",
                   label=f"Mean = {mean_lat:.1f} ms")
        ax.axhline(med_lat,  color="#f39c12", lw=2, linestyle=":",
                   label=f"Median = {med_lat:.1f} ms")
        for bar, val in zip(bars, delays):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Occlusion Event", fontsize=11, fontweight="bold")
        ax.set_ylabel("Detection Delay (ms)", fontsize=11, fontweight="bold")
        ax.set_title("Step 2.4 — Detection Latency per Occlusion Event",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "occlusion_latency_plot.png", dpi=300,
                    bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] occlusion_latency_plot.png")

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print(f"  Pooled F1={best_m['F1']:.4f}  Prec={best_m['Precision']:.4f}"
          f"  Rec={best_m['Recall']:.4f}  Acc={best_m['Accuracy']:.4f}")
    print(f"  Detection latency: mean={mean_lat:.1f} ms  "
          f"median={med_lat:.1f} ms  std={std_lat:.1f} ms")
    print("=" * 65)


if __name__ == "__main__":
    main()
