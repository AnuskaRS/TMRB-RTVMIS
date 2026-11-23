"""
Journal-Ready Update Script for E:/Antigravity/TMRB/occlusion_analysis.py
Implements:
1. True Instance Track ID Assignment (track_id) via Hungarian spatial matching for Particle Filters.
2. Explicit DEPTH_MARGIN = 5.0 for relative depth ordering noise tolerance.
3. Cleaned unused imports (os, sys, time, warnings, PIL, torchvision).
4. Efficient per-video latency lookup with video-specific FPS (fps_dict[vname]).
5. Renamed plot title to "Precision-Recall Across Overlap Thresholds (θ)".
"""
from pathlib import Path

CODE = '''"""
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
  occlusion_predictions.csv   — per-frame predicted labels for each δ
  occlusion_metrics.csv       — summary metrics at best δ
  occlusion_pr_curve.png      — Precision-Recall plot across δ values
  occlusion_confusion.png     — Confusion matrix at best δ
  occlusion_latency.csv       — per-event detection delay table
  occlusion_latency_plot.png  — bar chart of detection delays

==========================================================================
"""

import cv2
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import torch
from ultralytics import YOLO

# Matplotlib headless mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG / CONSTANTS ──
BASE_DIR       = Path("E:/Antigravity/TMRB")
GT_CSV_PATH    = BASE_DIR / "occlusion_gt.csv"
MODEL_PATH     = BASE_DIR / "best1.pt"
OUTPUT_DIR     = BASE_DIR / "occlusion_analysis"

ROI_CLASS_ID   = 0     # Class ID for Region of Interest
NUM_PARTICLES  = 100
DEPTH_SCALE    = 65.0  # Relative depth index scaling [0, 65]
DEPTH_MARGIN   = 5.0   # Relative depth noise tolerance margin

SWEEP_DELTAS   = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]

# ─────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS & PARTICLE FILTER
# ─────────────────────────────────────────────────────────────────────

class ParticleFilterBBox:
    def __init__(self, num_particles, init_bbox):
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, 4) + np.array(init_bbox)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, noise=2.0):
        self.particles += np.random.randn(self.num_particles, 4) * noise

    def update(self, detected_bbox):
        distances = np.linalg.norm(self.particles - np.array(detected_bbox), axis=1)
        self.weights = np.exp(-distances / 10.0)
        total = np.sum(self.weights)
        if total == 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= total

    def resample(self):
        N_eff = 1.0 / np.sum(self.weights ** 2)
        if N_eff < self.num_particles / 2:
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

def intersection_over_roi(box_instr, box_roi):
    if box_instr is None or box_roi is None: return 0.0
    x1 = max(box_instr[0], box_roi[0]); y1 = max(box_instr[1], box_roi[1])
    x2 = min(box_instr[2], box_roi[2]); y2 = min(box_instr[3], box_roi[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_roi = max(1.0, (box_roi[2] - box_roi[0]) * (box_roi[3] - box_roi[1]))
    return inter / area_roi


# ─────────────────────────────────────────────────────────────────────
#  MIDAS RELATIVE DEPTH SAMPLING
# ─────────────────────────────────────────────────────────────────────

def compute_frame_depth_map(frame_rgb, midas_model, midas_transforms, device):
    if midas_model is None: return None
    h, w = frame_rgb.shape[:2]
    inp = midas_transforms(frame_rgb).to(device)
    if inp.dim() == 3: inp = inp.unsqueeze(0)
    with torch.no_grad():
        dmap = midas_model(inp).squeeze().cpu().numpy()
    dmap = cv2.resize(dmap, (w, h), interpolation=cv2.INTER_LINEAR)
    dmin, dmax = dmap.min(), dmap.max()
    if dmax - dmin < 1e-6:
        return np.full((h, w), 32.5, dtype=np.float32)
    norm_depth = (dmap - dmin) / (dmax - dmin)
    return (norm_depth * DEPTH_SCALE).astype(np.float32)

def sample_bbox_depth(depth_map, box):
    if box is None or depth_map is None: return 32.5
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: return 32.5
    crop = depth_map[y1:y2, x1:x2]
    return float(crop.mean()) if crop.size > 0 else 32.5


# ─────────────────────────────────────────────────────────────────────
#  OCCLUSION ALGORITHM EXECUTION ENGINE (INSTANCES TRACKED WITH TRACK_ID)
# ─────────────────────────────────────────────────────────────────────

def run_occlusion_algorithm(video_path, gt_frame_indices, model, depth_model,
                              transform, device):
    """
    Processes video sequentially, maintaining Instance Track IDs (track_id) for Particle Filters.
    Returns dict: frame_idx → {'overlap': float, 'z_instr': float, 'z_roi': float}
    """
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results  = {}
    pf_tracks = {}  # track_id -> ParticleFilterBBox
    next_track_id = 1
    roi_pf = None

    print(f"  Processing {Path(video_path).name} ({total} frames, sampling {len(gt_frame_indices)} GT frames) …")

    gt_set = set(gt_frame_indices)
    max_gt_frame = max(gt_frame_indices) if gt_frame_indices else 0

    for frame_idx in range(max_gt_frame + 1):
        ret, frame = cap.read()
        if not ret:
            if frame_idx in gt_set:
                results[frame_idx] = {"overlap": 0.0, "z_instr": 32.5, "z_roi": 32.5}
            continue

        if frame_idx % 200 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            preds = model(frame, verbose=False)
        bboxes = preds[0].boxes

        roi_det = None
        instr_dets = []
        if bboxes is not None and len(bboxes):
            cls_ids = bboxes.cls.cpu().numpy().astype(int)
            xyxy    = bboxes.xyxy.cpu().numpy()
            for cid, box in zip(cls_ids, xyxy):
                if cid == ROI_CLASS_ID and roi_det is None:
                    roi_det = box
                elif cid != ROI_CLASS_ID:
                    instr_dets.append(box)

        # 1. Update ROI Particle Filter
        if roi_det is not None:
            if roi_pf is None: roi_pf = ParticleFilterBBox(NUM_PARTICLES, roi_det)
            else: roi_pf.predict(); roi_pf.update(roi_det); roi_pf.resample()
        elif roi_pf is not None:
            roi_pf.predict(); roi_pf.resample()
            
        roi_box_est = roi_pf.estimate() if roi_pf is not None else roi_det

        # 2. Update Instrument Instance Particle Filters (Hungarian track_id Assignment)
        active_tids = list(pf_tracks.keys())
        if active_tids and instr_dets:
            prev_ests = [pf_tracks[tid].estimate() for tid in active_tids]
            cost_mat  = np.zeros((len(prev_ests), len(instr_dets)))
            for i, pe in enumerate(prev_ests):
                for j, cd in enumerate(instr_dets):
                    cost_mat[i, j] = np.linalg.norm(pe[:2] - cd[:2])
            r_ind, c_ind = linear_sum_assignment(cost_mat)
            assigned_c = set()
            for r, c in zip(r_ind, c_ind):
                if cost_mat[r, c] < 150.0:
                    tid = active_tids[r]
                    pf_tracks[tid].predict()
                    pf_tracks[tid].update(instr_dets[c])
                    pf_tracks[tid].resample()
                    assigned_c.add(c)
            for c, cd in enumerate(instr_dets):
                if c not in assigned_c:
                    pf_tracks[next_track_id] = ParticleFilterBBox(NUM_PARTICLES, cd)
                    next_track_id += 1
        elif instr_dets:
            for cd in instr_dets:
                pf_tracks[next_track_id] = ParticleFilterBBox(NUM_PARTICLES, cd)
                next_track_id += 1
        else:
            for tid in list(pf_tracks.keys()):
                pf_tracks[tid].predict()
                pf_tracks[tid].resample()

        # Record result for ground-truth evaluation frames
        if frame_idx in gt_set:
            instr_estimates = [pf_tracks[tid].estimate() for tid in pf_tracks]
            max_overlap = max([intersection_over_roi(ib, roi_box_est) for ib in instr_estimates]) if (instr_estimates and roi_box_est is not None) else 0.0

            depth_map = compute_frame_depth_map(frame_rgb, depth_model, transform, device) if depth_model else None
            z_roi = sample_bbox_depth(depth_map, roi_box_est)
            z_instr = min([sample_bbox_depth(depth_map, ib) for ib in instr_estimates]) if (instr_estimates and depth_map is not None) else 32.5

            results[frame_idx] = {"overlap": max_overlap, "z_instr": z_instr, "z_roi": z_roi}

    cap.release()
    return results, fps


# ─────────────────────────────────────────────────────────────────────
#  CLASSIFICATION & METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────

def classify_frame(res, delta):
    """
    Hybrid 3D Depth-Aware Decision Rule:
    Triggers occlusion if 2D ROI spatial overlap >= delta AND instrument relative depth z_instr <= z_roi + DEPTH_MARGIN
    """
    ov = res.get("overlap", 0.0)
    zi = res.get("z_instr", 32.5)
    zr = res.get("z_roi", 32.5)
    return 1 if (ov >= delta and zi <= zr + DEPTH_MARGIN) else 0

def compute_metrics(y_true, y_pred):
    tp = sum(1 for gt, pr in zip(y_true, y_pred) if gt == 1 and pr == 1)
    fp = sum(1 for gt, pr in zip(y_true, y_pred) if gt == 0 and pr == 1)
    tn = sum(1 for gt, pr in zip(y_true, y_pred) if gt == 0 and pr == 0)
    fn = sum(1 for gt, pr in zip(y_true, y_pred) if gt == 1 and pr == 0)

    acc  = (tp + tn) / len(y_true) if y_true else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


# ─────────────────────────────────────────────────────────────────────
#  MAIN EVALUATION & ARTIFACT GENERATION
# ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  OCCLUSION ANALYSIS — Steps 2.2 / 2.3 / 2.4")
    print("=" * 65)

    # 1. Load Ground Truth
    gt_by_video = defaultdict(dict)
    with open(GT_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_by_video[row['video_name']][int(row['frame_index'])] = int(row['label'])

    total_gt = sum(len(v) for v in gt_by_video.values())
    print(f"  Loaded GT: {total_gt} frames across {len(gt_by_video)} videos")

    # 2. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print("  Loading YOLOv10 …")
    model = YOLO(str(MODEL_PATH))

    print("  Loading MiDaS Monocular Depth Model …")
    try:
        depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False).eval().to(device)
        transform   = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False).small_transform
        print("  MiDaS loaded successfully.")
    except Exception as e:
        print(f"  MiDaS load failed ({e}); falling back to 2D spatial overlap.")
        depth_model, transform = None, None

    # 3. Run Occlusion Engine
    all_results = {}
    fps_dict    = {}

    for vname, gt_frames in gt_by_video.items():
        vpath = BASE_DIR / vname
        if not vpath.exists():
            print(f"  [WARN] Video {vname} not found; skipping.")
            continue

        res, fps = run_occlusion_algorithm(
            vpath, list(gt_frames.keys()), model, depth_model, transform, device
        )
        all_results[vname] = res
        fps_dict[vname]    = fps

    # 4. Flatten GT and Results
    flat_gt      = []
    flat_results = []
    meta_info    = []

    for vname, gt_frames in gt_by_video.items():
        if vname not in all_results: continue
        for f_idx in sorted(gt_frames.keys()):
            flat_gt.append(gt_frames[f_idx])
            flat_results.append(all_results[vname].get(f_idx, {"overlap": 0.0, "z_instr": 32.5, "z_roi": 32.5}))
            meta_info.append((vname, f_idx))

    # 5. Step 2.3 — Threshold Delta Sweep
    metrics_by_delta = {}
    for delta in SWEEP_DELTAS:
        preds = [classify_frame(res, delta) for res in flat_results]
        m = compute_metrics(flat_gt, preds)
        m['delta'] = delta
        metrics_by_delta[delta] = m

    best_delta = max(SWEEP_DELTAS, key=lambda d: metrics_by_delta[d]['F1'])
    best_m     = metrics_by_delta[best_delta]

    print("\n" + "=" * 65)
    print("  THRESHOLD DELTA SWEEP (Step 2.3)")
    print("=" * 65)
    print(f"  {'δ':>8}  {'TP':>5}  {'FP':>5}  {'TN':>5}  {'FN':>5}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    print("  " + "─" * 62)
    for delta in SWEEP_DELTAS:
        m = metrics_by_delta[delta]
        star = " ★" if delta == best_delta else ""
        print(f"  {delta:8.4f}  {m['TP']:5d}  {m['FP']:5d}  {m['TN']:5d}  {m['FN']:5d}  "
              f"{m['Accuracy']:7.4f}  {m['Precision']:7.4f}  {m['Recall']:7.4f}  {m['F1']:7.4f}{star}")

    print(f"\n  Optimal Threshold: δ* = {best_delta}")

    # 6. Export occlusion_predictions.csv using UNIFIED criterion
    pred_path = OUTPUT_DIR / "occlusion_predictions.csv"
    with open(pred_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["video_name", "frame_index", "gt_label", "roi_overlap", "z_instr", "z_roi"] + [f"pred_delta_{d}" for d in SWEEP_DELTAS]
        writer.writerow(header)

        for (vname, f_idx), gt, res in zip(meta_info, flat_gt, flat_results):
            row = [vname, f_idx, gt, f"{res['overlap']:.6f}", f"{res['z_instr']:.2f}", f"{res['z_roi']:.2f}"]
            for d in SWEEP_DELTAS:
                row.append(classify_frame(res, d))
            writer.writerow(row)
    print(f"  [SAVED] {pred_path.name}")

    # 7. Export occlusion_metrics.csv & occlusion_best_metrics.csv
    with open(OUTPUT_DIR / "occlusion_metrics.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["delta", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"])
        writer.writeheader()
        for d in SWEEP_DELTAS:
            writer.writerow(metrics_by_delta[d])

    with open(OUTPUT_DIR / "occlusion_best_metrics.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["delta", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"])
        writer.writeheader()
        writer.writerow(best_m)

    # 8. Step 2.4 — Detection Delay (Latency) with Video-Specific FPS
    events = []
    for vname, gt_frames in gt_by_video.items():
        sorted_frames = sorted(gt_frames.keys())
        in_event = False
        start_frame = None
        prev_fi = sorted_frames[0] if sorted_frames else 0

        for fi in sorted_frames:
            lbl = gt_frames[fi]
            if lbl == 1 and not in_event:
                in_event = True
                start_frame = fi
            elif lbl == 0 and in_event:
                events.append((vname, start_frame, prev_fi))
                in_event = False
            prev_fi = fi
        if in_event:
            events.append((vname, start_frame, sorted_frames[-1]))

    latency_rows = []

    for idx, (vname, estart, eend) in enumerate(events, 1):
        v_fps = fps_dict.get(vname, 30.0)
        v_ms_per_frame = 1000.0 / v_fps if v_fps > 0 else 33.3

        detected_frame = None
        # Efficient lookup restricted to evaluation frames within event bounds
        avail_frames = sorted([f for f in all_results.get(vname, {}) if estart <= f <= eend])
        for fi in avail_frames:
            if classify_frame(all_results[vname][fi], best_delta) == 1:
                detected_frame = fi
                break

        if detected_frame is not None:
            delay_frames = detected_frame - estart
            delay_ms     = delay_frames * v_ms_per_frame
            status       = "Detected"
        else:
            delay_frames = -1
            delay_ms     = -1.0
            status       = "Missed"

        latency_rows.append({
            "event_id": f"E{idx}",
            "video_name": vname,
            "gt_start_frame": estart,
            "gt_end_frame": eend,
            "detected_frame": detected_frame if detected_frame is not None else -1,
            "delay_frames": delay_frames if delay_frames is not None else -1,
            "delay_ms": round(delay_ms, 2) if delay_ms is not None else -1.0,
            "status": status
        })

    with open(OUTPUT_DIR / "occlusion_latency.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(latency_rows[0].keys()))
        writer.writeheader()
        writer.writerows(latency_rows)

    # 9. Plot PR Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.suptitle("Step 2.3 — Threshold Sweep Analysis (ROI Overlap Threshold θ)", fontsize=13, fontweight="bold")

    recalls    = [metrics_by_delta[d]['Recall'] for d in SWEEP_DELTAS]
    precisions = [metrics_by_delta[d]['Precision'] for d in SWEEP_DELTAS]
    f1s        = [metrics_by_delta[d]['F1'] for d in SWEEP_DELTAS]

    ax1 = axes[0]
    ax1.plot(recalls, precisions, 'o-', color='#3498db', lw=2, ms=7, label='Threshold Sweep PR')
    ax1.scatter([best_m['Recall']], [best_m['Precision']], color='#2ecc71', s=250, zorder=5, edgecolors='black',
                label=f"Optimal θ* = {best_delta}\\n(P = {best_m['Precision']:.3f}, R = {best_m['Recall']:.3f})")
    ax1.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax1.set_title("Precision–Recall Across Overlap Thresholds θ", fontsize=12, fontweight="bold")
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="lower left", fontsize=10)

    ax2 = axes[1]
    ax2.plot(SWEEP_DELTAS, f1s, 'o-', color='#e74c3c', lw=2, ms=7, label='F1-Score')
    ax2.axvline(best_delta, color='#2ecc71', linestyle='--', lw=2, label=f'Optimal θ* = {best_delta}')
    ax2.set_xscale("log")
    ax2.set_xlabel("ROI Overlap Ratio Threshold (θ)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax2.set_title("F1-Score vs. Overlap Ratio Threshold", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "occlusion_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 10. Plot Confusion Matrix
    cm = np.array([[best_m['TN'], best_m['FP']],
                   [best_m['FN'], best_m['TP']]])
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Not Occluded (0)", "Occluded (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ground Truth Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix (θ* = {best_delta})\\nAccuracy = {best_m['Accuracy']:.3f} | F1 = {best_m['F1']:.3f}", fontsize=11, fontweight="bold")

    labels_cm = [[f"TN = {best_m['TN']}", f"FP = {best_m['FP']}"],
                 [f"FN = {best_m['FN']}", f"TP = {best_m['TP']}"]]
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, labels_cm[i][j], ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "occlusion_confusion.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== OCCLUSION ANALYSIS EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()
'''

with open("E:/Antigravity/TMRB/occlusion_analysis.py", "w", encoding="utf-8") as f:
    f.write(CODE)

print("[UPDATED] E:/Antigravity/TMRB/occlusion_analysis.py refactored into journal-ready standard!")
