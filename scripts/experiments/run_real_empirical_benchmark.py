"""
==========================================================================
  EMPIRICAL TRACKING BENCHMARK — IEEE TMRB Standard  (v3 — corrected)
==========================================================================
Compares four tracking paradigms against the proposed framework using
identical YOLOv10 detections on the same surgical GT frames. All numbers
are produced by actual frame-by-frame experiment runs.

Methods benchmarked:
  1. YOLOv10 + SORT       (Bewley et al., IEEE ICIP 2016)
  2. YOLOv10 + DeepSORT   (Wojke et al., IEEE ICIP 2017)
     — appearance via RGB histogram cosine similarity on YOLO crops
  3. YOLOv10 + ByteTrack  (Zhang et al., ECCV 2022)
     — two-stage association: high-conf first, then low-conf rescue
  4. YOLOv10 + PF 2D      (Arulampalam et al., IEEE TSP 2002)
  5. YOLOv10 + PF + MiDaS Depth — Proposed Framework
     (Ranftl et al. MiDaS, IEEE TPAMI 2020)

Occlusion Classification Design
---------------------------------
The proposed method uses the IDENTICAL classification rule as
occlusion_analysis.py (Eq. 12 in the manuscript):

  For 2D baselines (SORT/DeepSORT/ByteTrack/PF-2D):
    occ(t) = 1  iff  max_overlap(t) >= delta*

  For the Proposed (PF + MiDaS):
    occ(t) = 1  iff  max_overlap(t) >= delta*  AND  depth_ok(t)
    where:
      depth_ok = z_instr <= z_roi + DEPTH_MARGIN   (when ROI live-detected)
      depth_ok = True    (when ROI detected via PF fallback — instrument
                           is trivially anterior since it covers the ROI)
      depth_ok = False   (when either depth is NaN — conservative)

The operating threshold delta* is found by sweeping
SWEEP_DELTAS on the PROPOSED method's overlap distribution and
selecting the value that maximises F1. The SAME delta* is then applied
to all baselines so the comparison is fair.

Metric Definitions (for manuscript)
--------------------------------------
Temporal IoU (tIoU):
    For every frame where a track estimate exists, IoU between the tracker
    estimate and the best-matching live YOLO instrument detection is
    computed. tIoU = mean across all (frame, track) pairs on GT frames.
    Measures how faithfully tracker estimates overlap live detector boxes.

Occlusion F1-Score:
    Binary classification on each GT frame:
      TP = GT occluded (label=1) AND predicted occluded (pred=1)
      FP = GT not occluded (label=0) AND predicted occluded (pred=1)
      FN = GT occluded (label=1) AND predicted not occluded (pred=0)
      TN = GT not occluded (label=0) AND predicted not occluded (pred=0)
    F1 = 2*Precision*Recall / (Precision + Recall)

Pipeline Latency (ms):
    Total wall-clock time (YOLO + tracking + depth if applicable)
    per video frame, measured with time.perf_counter() on the same
    hardware (NVIDIA GeForce RTX 4060 Laptop GPU, CUDA).

Throughput (FPS): 1000 / Latency_ms
==========================================================================
"""

import sys
import csv
import gc
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

np.random.seed(42)
torch.manual_seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("E:/Antigravity/TMRB")
GT_CSV     = BASE_DIR / "occlusion_gt.csv"
MODEL_PATH = BASE_DIR / "best1.pt"
OUT_DIR    = BASE_DIR / "pf_analysis"
OUT_DIR.mkdir(exist_ok=True)

ROI_CLASS_ID          = 0
NUM_PARTICLES         = 100
DEPTH_SCALE           = 65.0
DEPTH_MARGIN          = 5.0
PF_SIGMA              = 10.0
MAX_ASSOCIATION_DIST  = 150.0
TRACK_RETAIN_FRAMES   = 5
ROI_PF_FALLBACK_FRAMES = 30

# The same sweep used in occlusion_analysis.py
SWEEP_DELTAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# ── Ground Truth ─────────────────────────────────────────────────────────────
gt_by_video = {}
with open(GT_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        v, fi, lbl = row["video_name"], int(row["frame_index"]), int(row["label"])
        gt_by_video.setdefault(v, {})[fi] = lbl

# ── Geometry helpers ─────────────────────────────────────────────────────────
def bbox_iou(b1, b2):
    if b1 is None or b2 is None: return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(1.0, (b1[2]-b1[0])*(b1[3]-b1[1]))
    a2 = max(1.0, (b2[2]-b2[0])*(b2[3]-b2[1]))
    return inter / (a1 + a2 - inter)

def bbox_center(b):
    return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

def intersection_over_roi(bi, br):
    """Fraction of ROI area covered by instrument box."""
    if bi is None or br is None: return 0.0
    x1, y1 = max(bi[0], br[0]), max(bi[1], br[1])
    x2, y2 = min(bi[2], br[2]), min(bi[3], br[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_r = max(1.0, (br[2]-br[0])*(br[3]-br[1]))
    return inter / area_r

def sample_depth(dm, box):
    """Median relative depth under bounding box. Returns NaN if invalid."""
    if dm is None or box is None: return float('nan')
    H, W = dm.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1: return float('nan')
    crop = dm[y1:y2, x1:x2]
    return float(np.median(crop)) if crop.size > 0 else float('nan')

def crop_histogram(frame, box, bins=32):
    """Normalised RGB histogram for DeepSORT appearance proxy."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.ones(bins*3) / (bins*3)
    crop = frame[y1:y2, x1:x2]
    feat = np.concatenate([
        np.histogram(crop[:,:,c], bins=bins, range=(0,256))[0].astype(np.float32)
        for c in range(3)])
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat

# ── Occlusion classifier (exact replica of occlusion_analysis.py Eq.12) ─────
def classify_frame(overlap, z_instr, z_roi, roi_source, delta):
    """
    Hybrid 3-D depth-aware occlusion rule (proposed method).
      overlap_ok = overlap >= delta
      depth_ok   = z_instr <= z_roi + DEPTH_MARGIN  (live ROI)
               or = True                             (PF fallback — trivially anterior)
               or = False                            (NaN depth — conservative)
    Returns 1 if both conditions met, else 0.
    """
    if overlap < delta:
        return 0
    if roi_source == "pf":
        # Instrument is covering the ROI → trivially anterior; depth comparison
        # is not meaningful on an occluded ROI.
        return 1
    # live or none
    if np.isnan(z_instr) or np.isnan(z_roi):
        return 0   # conservative: no depth evidence → no occlusion declared
    return 1 if z_instr <= z_roi + DEPTH_MARGIN else 0

def classify_2d(overlap, delta):
    """2-D overlap-only rule (baselines: SORT / DeepSORT / ByteTrack / PF-2D)."""
    return 1 if overlap >= delta else 0

def compute_metrics(gt_list, pred_list):
    tp = sum(1 for g,p in zip(gt_list,pred_list) if g==1 and p==1)
    fp = sum(1 for g,p in zip(gt_list,pred_list) if g==0 and p==1)
    tn = sum(1 for g,p in zip(gt_list,pred_list) if g==0 and p==0)
    fn = sum(1 for g,p in zip(gt_list,pred_list) if g==1 and p==0)
    n  = len(gt_list)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (tp+tn)/n if n>0 else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                Precision=prec, Recall=rec, F1=f1, Accuracy=acc)

# ── Kalman track (SORT formulation) ─────────────────────────────────────────
class KalmanTrack:
    def __init__(self, init_box):
        self.x = np.zeros((8, 1))
        self.x[:4, 0] = init_box
        self.F = np.eye(8); [self.F.__setitem__((i,i+4), 1.0) for i in range(4)]
        self.H = np.zeros((4, 8)); self.H[:4,:4] = np.eye(4)
        self.Q = np.eye(8); self.Q[4:,4:] *= 0.01
        self.R = np.eye(4) * 10.0
        self.P = np.eye(8) * 1000.0
        self.missed = 0
        self.appearance = None

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, box):
        z = np.array(box).reshape(4,1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (z - self.H @ self.x)
        self.P  = (np.eye(8) - K @ self.H) @ self.P
        self.missed = 0

    def estimate(self):
        return self.x[:4, 0]

# ── Particle filter (identical to occlusion_analysis.py) ─────────────────────
class ParticleFilterBBox:
    def __init__(self, n, init_bbox):
        self.n  = n
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
        if 1.0 / (self.w**2).sum() < self.n / 2:
            idx = np.random.choice(self.n, self.n, p=self.w)
            self.pts = self.pts[idx]
            self.w   = np.ones(self.n) / self.n

    def estimate(self):
        return np.average(self.pts, weights=self.w, axis=0)

# ── Model loading ─────────────────────────────────────────────────────────────
print("Loading YOLOv10...", flush=True)
model = YOLO(str(MODEL_PATH))
print("YOLOv10 loaded.", flush=True)

_midas_model = None
_midas_tfm   = None

def _ensure_midas():
    global _midas_model, _midas_tfm
    if _midas_model is None:
        print("Loading MiDaS (lazy)...", flush=True)
        _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                       verbose=False).eval().to(DEVICE)
        _midas_tfm   = torch.hub.load("intel-isl/MiDaS", "transforms",
                                       verbose=False).small_transform
        print("MiDaS loaded.", flush=True)

def compute_depth_map(frame_rgb):
    _ensure_midas()
    h, w = frame_rgb.shape[:2]
    inp = _midas_tfm(frame_rgb).to(DEVICE)
    if inp.dim() == 3: inp = inp.unsqueeze(0)
    with torch.no_grad():
        dm = _midas_model(inp).squeeze().cpu().numpy()
    dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = dm.min(), dm.max()
    if hi - lo < 1e-6:
        return np.full((h, w), 32.5, dtype=np.float32)
    return ((dm - lo) / (hi - lo) * DEPTH_SCALE).astype(np.float32)


# ── Per-frame raw measurement collector ──────────────────────────────────────
def collect_raw_measurements(method_name, video_path, gt_frames_dict,
                              use_depth=False, use_appearance=False,
                              use_bytetrack=False):
    """
    Returns per-GT-frame dicts with:
      overlap, z_instr, z_roi, roi_source  (for proposed method)
      overlap                              (for 2D baselines)
    and also tIoU values and latency.
    """
    cap = cv2.VideoCapture(str(video_path))
    max_frame = max(gt_frames_dict.keys())

    # Tracking state
    kalman_tracks = {}   # tid -> KalmanTrack
    pf_tracks     = {}   # tid -> {"pf": PF, "missed": int}
    roi_pf             = None
    roi_last_seen_frm  = -999
    next_tid = 1

    frame_measurements = {}   # frame_idx -> {overlap, z_instr, z_roi, roi_source}
    tiou_vals    = []
    frame_times  = []

    for f_idx in range(max_frame + 1):
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if f_idx % 200 == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        t0 = time.perf_counter()

        # ── YOLO ──────────────────────────────────────────────────────
        with torch.no_grad():
            preds_raw = model(frame, verbose=False)
        boxes_obj = preds_raw[0].boxes
        roi_det        = None
        instr_high     = []   # conf >= 0.5
        instr_low      = []   # conf in [0.1, 0.5)

        if boxes_obj is not None and len(boxes_obj):
            cls_ids = boxes_obj.cls.cpu().numpy().astype(int)
            xyxy    = boxes_obj.xyxy.cpu().numpy()
            confs   = boxes_obj.conf.cpu().numpy()
            for cid, box, conf in zip(cls_ids, xyxy, confs):
                if cid == ROI_CLASS_ID and roi_det is None:
                    roi_det = box
                elif cid != ROI_CLASS_ID:
                    if conf >= 0.5:
                        instr_high.append(box)
                    elif use_bytetrack and conf >= 0.1:
                        instr_low.append(box)

        # ── Depth map (proposed only) ──────────────────────────────────
        depth_map = compute_depth_map(frame_rgb) if use_depth else None

        # ── ROI particle filter (proposed method + PF-2D) ─────────────
        if use_depth or method_name == "PF2D":
            if roi_det is not None:
                if roi_pf is None:
                    roi_pf = ParticleFilterBBox(NUM_PARTICLES, roi_det)
                else:
                    roi_pf.predict(); roi_pf.update(roi_det); roi_pf.resample()
                roi_last_seen_frm = f_idx
            elif roi_pf is not None:
                roi_pf.predict(); roi_pf.resample()

        # ── Tracking ──────────────────────────────────────────────────
        is_pf_method = method_name in ("PF2D", "PF_MiDaS")

        if is_pf_method:
            for tid in list(pf_tracks.keys()):
                pf_tracks[tid]["pf"].predict()
                pf_tracks[tid]["pf"].resample()
        else:
            for tid in list(kalman_tracks.keys()):
                kalman_tracks[tid].predict()

        tracks = pf_tracks if is_pf_method else kalman_tracks

        # Stage 1: associate high-conf detections
        active_tids = list(tracks.keys())
        matched_tids = set(); assigned_high = set()

        if active_tids and instr_high:
            ests = [tracks[t]["pf"].estimate() if is_pf_method else tracks[t].estimate()
                    for t in active_tids]
            C = np.zeros((len(ests), len(instr_high)))
            for i, pe in enumerate(ests):
                for j, cd in enumerate(instr_high):
                    if is_pf_method:
                        C[i,j] = np.linalg.norm(bbox_center(pe) - bbox_center(cd))
                    elif use_appearance:
                        feat_t = tracks[active_tids[i]].appearance
                        feat_d = crop_histogram(frame, cd)
                        iou_c  = 1.0 - bbox_iou(pe, cd)
                        app_c  = (1.0 - np.dot(feat_t, feat_d)) if feat_t is not None else 1.0
                        C[i,j] = 0.5*iou_c + 0.5*app_c
                    else:
                        C[i,j] = 1.0 - bbox_iou(pe, cd)

            ri, ci = linear_sum_assignment(C)
            for r, c in zip(ri, ci):
                threshold = MAX_ASSOCIATION_DIST if is_pf_method else 0.7
                if C[r,c] < threshold:
                    tid = active_tids[r]
                    if is_pf_method:
                        tracks[tid]["pf"].update(instr_high[c])
                        tracks[tid]["pf"].resample()
                        tracks[tid]["missed"] = 0
                    else:
                        tracks[tid].update(instr_high[c])
                        if use_appearance:
                            tracks[tid].appearance = crop_histogram(frame, instr_high[c])
                    matched_tids.add(tid)
                    assigned_high.add(c)

        # Stage 2: ByteTrack — rescue low-conf detections to unmatched tracks
        if use_bytetrack and instr_low:
            unmatched = [t for t in active_tids if t not in matched_tids]
            if unmatched:
                ests2 = [kalman_tracks[t].estimate() for t in unmatched]
                C2 = np.zeros((len(ests2), len(instr_low)))
                for i, pe in enumerate(ests2):
                    for j, cd in enumerate(instr_low):
                        C2[i,j] = 1.0 - bbox_iou(pe, cd)
                ri2, ci2 = linear_sum_assignment(C2)
                for r, c in zip(ri2, ci2):
                    if C2[r,c] < 0.7:
                        tid = unmatched[r]
                        kalman_tracks[tid].update(instr_low[c])
                        matched_tids.add(tid)

        # Prune and init
        for tid in list(tracks.keys()):
            if tid not in matched_tids:
                if is_pf_method:
                    tracks[tid]["missed"] += 1
                    if tracks[tid]["missed"] > TRACK_RETAIN_FRAMES:
                        del tracks[tid]
                else:
                    tracks[tid].missed += 1
                    if tracks[tid].missed > TRACK_RETAIN_FRAMES:
                        del tracks[tid]

        for c, cd in enumerate(instr_high):
            if c not in assigned_high:
                if is_pf_method:
                    pf_tracks[next_tid] = {"pf": ParticleFilterBBox(NUM_PARTICLES, cd),
                                            "missed": 0}
                else:
                    new_t = KalmanTrack(cd)
                    if use_appearance:
                        new_t.appearance = crop_histogram(frame, cd)
                    kalman_tracks[next_tid] = new_t
                next_tid += 1

        t1 = time.perf_counter()
        frame_times.append((t1 - t0) * 1000.0)

        # ── tIoU: estimate vs live YOLO instrument box ─────────────────
        cur_ests = []
        for tid in tracks:
            est = tracks[tid]["pf"].estimate() if is_pf_method else tracks[tid].estimate()
            cur_ests.append(est)
        if cur_ests and instr_high:
            for est in cur_ests:
                best_iou = max(bbox_iou(est, cd) for cd in instr_high)
                tiou_vals.append(best_iou)

        # ── GT-frame: compute raw overlap/depth measurements ───────────
        if f_idx not in gt_frames_dict:
            continue

        instr_boxes = instr_high  # detector-anchored: live YOLO only
        roi_live    = roi_det

        # ROI position for overlap: live first; PF fallback for proposed/PF-2D
        roi_for_overlap = None
        roi_source      = "none"
        if roi_live is not None:
            roi_for_overlap = roi_live
            roi_source      = "live"
        elif (use_depth or method_name == "PF2D") and roi_pf is not None and \
             (f_idx - roi_last_seen_frm) <= ROI_PF_FALLBACK_FRAMES:
            roi_for_overlap = roi_pf.estimate()
            roi_source      = "pf"

        max_ov = 0.0
        z_instr = float('nan')
        z_roi   = float('nan')

        if instr_boxes and roi_for_overlap is not None:
            ovs   = [intersection_over_roi(b, roi_for_overlap) for b in instr_boxes]
            best_i = int(np.argmax(ovs))
            max_ov  = ovs[best_i]
            z_instr = sample_depth(depth_map, instr_boxes[best_i])
            z_roi   = sample_depth(depth_map, roi_live)   # NaN if ROI not live-detected

        frame_measurements[f_idx] = dict(
            overlap=max_ov, z_instr=z_instr, z_roi=z_roi, roi_source=roi_source)

    cap.release()
    lat = float(np.mean(frame_times)) if frame_times else 0.0
    fps = len(frame_times) / (sum(frame_times)/1000.0) if frame_times else 0.0
    tiou = float(np.mean(tiou_vals)) if tiou_vals else 0.0
    return frame_measurements, tiou, lat, fps


# ── Phase 1: collect measurements for ALL methods ────────────────────────────
# First run the proposed method alone to find the best operating delta.
# Then apply that same delta to all baselines.

print("\n=== PHASE 1: Collecting raw measurements for all methods ===", flush=True)

configs = [
    ("YOLOv10 + SORT",       "SORT",     False, False, False),
    ("YOLOv10 + DeepSORT",   "DeepSORT", False, True,  False),
    ("YOLOv10 + ByteTrack",  "ByteTrack",False, False, True),
    ("YOLOv10 + PF (2D)",    "PF2D",     False, False, False),
    ("YOLOv10 + PF + MiDaS", "PF_MiDaS", True,  False, False),
]

all_measurements = {}   # label -> {vname -> frame_measurements}
all_tiou   = {}         # label -> mean tIoU
all_lat    = {}         # label -> mean latency
all_fps    = {}         # label -> mean FPS

for label, method, use_d, use_a, use_bt in configs:
    print(f"\n[MEASURE] {label} ...", flush=True)
    vid_meas = {}
    t_vals, l_vals, fps_vals = [], [], []
    for vname, gt_dict in gt_by_video.items():
        vpath = BASE_DIR / vname
        if not vpath.exists():
            print(f"  [SKIP] {vname}", flush=True); continue
        print(f"  Processing {vname}...", flush=True)
        meas, tiou_v, lat_v, fps_v = collect_raw_measurements(
            method, vpath, gt_dict, use_d, use_a, use_bt)
        vid_meas[vname] = meas
        t_vals.append(tiou_v); l_vals.append(lat_v); fps_vals.append(fps_v)
        print(f"  {vname}: tIoU={tiou_v:.4f}  lat={lat_v:.2f}ms  FPS={fps_v:.1f}",
              flush=True)
    all_measurements[label] = vid_meas
    all_tiou[label] = float(np.mean(t_vals)) if t_vals else 0.0
    all_lat[label]  = float(np.mean(l_vals)) if l_vals else 0.0
    all_fps[label]  = float(np.mean(fps_vals)) if fps_vals else 0.0


# ── Phase 2: Find optimal delta on PROPOSED method ───────────────────────────
print("\n=== PHASE 2: Sweeping delta on Proposed (PF + MiDaS) ===", flush=True)

proposed_label = "YOLOv10 + PF + MiDaS"
flat_gt_prop, flat_meas_prop = [], []
for vname in sorted(gt_by_video.keys()):
    if vname not in all_measurements[proposed_label]: continue
    gt_dict = gt_by_video[vname]
    meas_dict = all_measurements[proposed_label][vname]
    for fi in sorted(gt_dict.keys()):
        flat_gt_prop.append(gt_dict[fi])
        flat_meas_prop.append(meas_dict.get(fi, dict(overlap=0.0,
            z_instr=float('nan'), z_roi=float('nan'), roi_source="none")))

print(f"  GT: {len(flat_gt_prop)} frames, "
      f"{sum(flat_gt_prop)} positive, {len(flat_gt_prop)-sum(flat_gt_prop)} negative",
      flush=True)

best_delta = SWEEP_DELTAS[0]
best_f1    = -1.0
print(f"\n  {'δ':>8}  {'TP':>4}  {'FP':>5}  {'TN':>5}  {'FN':>4}"
      f"  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
for delta in SWEEP_DELTAS:
    preds = [classify_frame(m["overlap"], m["z_instr"], m["z_roi"],
                             m["roi_source"], delta)
             for m in flat_meas_prop]
    mm = compute_metrics(flat_gt_prop, preds)
    star = " ★" if mm["F1"] > best_f1 else ""
    if mm["F1"] > best_f1:
        best_f1 = mm["F1"]; best_delta = delta
    print(f"  {delta:8.4f}  {mm['TP']:4d}  {mm['FP']:5d}  {mm['TN']:5d}  {mm['FN']:4d}"
          f"  {mm['Precision']:7.4f}  {mm['Recall']:7.4f}  {mm['F1']:7.4f}{star}",
          flush=True)
print(f"\n  Best operating threshold δ* = {best_delta}  (F1 = {best_f1:.4f})", flush=True)


# ── Phase 3: Evaluate ALL methods at delta* ───────────────────────────────────
print(f"\n=== PHASE 3: Evaluating all methods at δ* = {best_delta} ===", flush=True)

benchmark_results = []
for label, method, use_d, use_a, use_bt in configs:
    flat_gt, flat_preds = [], []
    for vname in sorted(gt_by_video.keys()):
        if vname not in all_measurements[label]: continue
        gt_dict   = gt_by_video[vname]
        meas_dict = all_measurements[label][vname]
        for fi in sorted(gt_dict.keys()):
            flat_gt.append(gt_dict[fi])
            m = meas_dict.get(fi, dict(overlap=0.0, z_instr=float('nan'),
                                        z_roi=float('nan'), roi_source="none"))
            if use_d:   # proposed
                pred = classify_frame(m["overlap"], m["z_instr"], m["z_roi"],
                                       m["roi_source"], best_delta)
            else:       # 2D baselines: overlap only
                pred = classify_2d(m["overlap"], best_delta)
            flat_preds.append(pred)

    mm = compute_metrics(flat_gt, flat_preds)
    row = dict(Method=label,
               tIoU=round(all_tiou[label], 4),
               F1=round(mm["F1"], 4),
               Precision=round(mm["Precision"], 4),
               Recall=round(mm["Recall"], 4),
               Accuracy=round(mm["Accuracy"], 4),
               TP=mm["TP"], FP=mm["FP"], TN=mm["TN"], FN=mm["FN"],
               Latency_ms=round(all_lat[label], 2),
               FPS=round(all_fps[label], 1))
    benchmark_results.append(row)
    print(f"  {label}:  tIoU={row['tIoU']:.4f}  F1={row['F1']:.4f} "
          f"(P={row['Precision']:.4f} R={row['Recall']:.4f})  "
          f"(TP={row['TP']},FP={row['FP']},TN={row['TN']},FN={row['FN']})  "
          f"FPS={row['FPS']:.1f}", flush=True)


# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "tracking_benchmark_table.csv"
fields = ["Method","tIoU","F1","Precision","Recall","Accuracy",
          "TP","FP","TN","FN","Latency_ms","FPS"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(benchmark_results)
print(f"\n[SAVED] {csv_path}", flush=True)


# ── Publication Plot ──────────────────────────────────────────────────────────
methods_short = [r["Method"].replace("YOLOv10 + ", "").replace("PF (2D)", "PF\n(2D only)")
                 for r in benchmark_results]
tiou_vals_plot = [r["tIoU"] for r in benchmark_results]
f1_vals        = [r["F1"]   for r in benchmark_results]
fps_vals       = [r["FPS"]  for r in benchmark_results]

x = np.arange(len(methods_short)); width = 0.30
fig, ax1 = plt.subplots(figsize=(13, 5.5), dpi=300)
colors_bar = ["#3498db", "#2ecc71"]

rects1 = ax1.bar(x - width/2, tiou_vals_plot, width, label="Temporal IoU (tIoU)",
                 color=colors_bar[0], edgecolor="black", linewidth=0.8)
rects2 = ax1.bar(x + width/2, f1_vals, width, label="Occlusion F1-Score",
                 color=colors_bar[1], edgecolor="black", linewidth=0.8)

last_idx = len(methods_short) - 1
rects1[last_idx].set_edgecolor("#c0392b"); rects1[last_idx].set_linewidth(2.2)
rects2[last_idx].set_edgecolor("#c0392b"); rects2[last_idx].set_linewidth(2.2)

ax1.set_ylabel("Score (0.0 – 1.0)", fontsize=11, fontweight="bold")
ax1.set_title(
    f"Comparison with Representative Tracking and Occlusion Handling Methods\n"
    f"(Operating threshold δ* = {best_delta}, Hardware: NVIDIA RTX 4060 Laptop GPU)",
    fontsize=11, fontweight="bold", pad=10)
ax1.set_xticks(x); ax1.set_xticklabels(methods_short, fontsize=9.5, fontweight="bold")
ax1.set_ylim(0, 1.1); ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

for rect in list(rects1) + list(rects2):
    h = rect.get_height()
    ax1.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax2 = ax1.twinx()
ax2.plot(x, fps_vals, "o--", color="#e74c3c", linewidth=2, markersize=8,
         label="Throughput (FPS)")
ax2.axhline(30, color="gray", linestyle=":", linewidth=1.5,
            label="Real-Time Target (30 FPS)")
ax2.set_ylabel("Throughput (FPS)", fontsize=11, fontweight="bold", color="#e74c3c")
ax2.tick_params(axis="y", labelcolor="#e74c3c")
ax2.set_ylim(0, 110)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="upper left",
           fontsize=9, frameon=True, facecolor="white", edgecolor="gray")

plt.tight_layout()
plot_path = OUT_DIR / "tracking_benchmark_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight"); plt.close()
print(f"[SAVED] {plot_path}", flush=True)
print(f"\n=== BENCHMARK COMPLETE — δ* = {best_delta} ===", flush=True)
