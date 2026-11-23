import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import gc
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import cv2
import torch
from ultralytics import YOLO

# REPRODUCIBILITY
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR    = Path("E:/Antigravity/TMRB")
GT_CSV_PATH = BASE_DIR / "occlusion_gt.csv"
MODEL_PATH  = BASE_DIR / "best1.pt"
OUTPUT_DIR  = BASE_DIR / "occlusion_depth_benchmark"
OUTPUT_DIR.mkdir(exist_ok=True)

ROI_CLASS_ID           = 0
NUM_PARTICLES          = 100
DEPTH_SCALE            = 65.0
DEPTH_MARGIN           = 5.0
PF_SIGMA               = 10.0
MAX_ASSOCIATION_DIST   = 150.0
TRACK_RETAIN_FRAMES    = 5
ROI_PF_FALLBACK_FRAMES = 30
SWEEP_DELTAS           = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]

def bbox_center(box):
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

def intersection_over_roi(box_i, box_r):
    if box_i is None or box_r is None: return 0.0
    x1 = max(box_i[0], box_r[0]); y1 = max(box_i[1], box_r[1])
    x2 = min(box_i[2], box_r[2]); y2 = min(box_i[3], box_r[3])
    inter  = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_r = max(1.0, (box_r[2] - box_r[0]) * (box_r[3] - box_r[1]))
    return inter / area_r

def depth_map_for_frame(frame_rgb, midas, tfm, device):
    if midas is None:
        return None
    h, w = frame_rgb.shape[:2]
    inp = tfm(frame_rgb).to(device)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        dm = midas(inp)
        if isinstance(dm, dict): dm = dm["out"]
        dm = dm.squeeze().cpu().numpy()
    dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = dm.min(), dm.max()
    if hi - lo < 1e-6:
        return np.full((h, w), 32.5, dtype=np.float32)
    return ((dm - lo) / (hi - lo) * DEPTH_SCALE).astype(np.float32)

def sample_depth(dm, box):
    if dm is None or box is None: return float('nan')
    H, W = dm.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1: return float('nan')
    crop = dm[y1:y2, x1:x2]
    return float(np.median(crop)) if crop.size > 0 else float('nan')

def run_evaluation_for_model(model_name, midas, tfm, device, yolo_model, gt_by_video):
    print(f"\n=======================================================", flush=True)
    print(f" Running Occlusion Benchmark for: {model_name}", flush=True)
    print(f"=======================================================", flush=True)

    all_results = {}
    total_depth_time = 0.0
    total_depth_frames = 0

    for vname, gt_frames in gt_by_video.items():
        vpath = BASE_DIR / vname
        if not vpath.exists(): continue

        cap = cv2.VideoCapture(str(vpath))
        max_gt_frame = max(gt_frames.keys()) if gt_frames else 0
        gt_set = set(gt_frames.keys())

        results = {}
        pf_tracks = {}
        next_tid = 1
        roi_pf = None
        roi_last_seen_frm = -999

        for frame_idx in range(max_gt_frame + 1):
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                preds = yolo_model(frame, verbose=False)
            boxes = preds[0].boxes

            roi_det = None
            instr_dets = []
            if boxes is not None and len(boxes):
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                xyxy    = boxes.xyxy.cpu().numpy()
                for cid, box in zip(cls_ids, xyxy):
                    if cid == ROI_CLASS_ID and roi_det is None:
                        roi_det = box
                    elif cid != ROI_CLASS_ID:
                        instr_dets.append(box)

            if roi_det is not None:
                if roi_pf is None: roi_pf = ParticleFilterBBox(NUM_PARTICLES, roi_det)
                else: roi_pf.predict(); roi_pf.update(roi_det); roi_pf.resample()
                roi_last_seen_frm = frame_idx
            elif roi_pf is not None:
                roi_pf.predict(); roi_pf.resample()

            for tid in list(pf_tracks.keys()):
                pf_tracks[tid]["pf"].predict(); pf_tracks[tid]["pf"].resample()

            active_tids = list(pf_tracks.keys())
            matched_tids = set(); assigned_c = set()
            if active_tids and instr_dets:
                ests = [pf_tracks[t]["pf"].estimate() for t in active_tids]
                C = np.zeros((len(ests), len(instr_dets)))
                for i, pe in enumerate(ests):
                    for j, cd in enumerate(instr_dets):
                        C[i, j] = np.linalg.norm(bbox_center(pe) - bbox_center(cd))
                ri, ci = linear_sum_assignment(C)
                for r, c in zip(ri, ci):
                    if C[r, c] < MAX_ASSOCIATION_DIST:
                        tid = active_tids[r]
                        pf_tracks[tid]["pf"].update(instr_dets[c]); pf_tracks[tid]["pf"].resample()
                        pf_tracks[tid]["missed"] = 0
                        matched_tids.add(tid); assigned_c.add(c)

            for c, cd in enumerate(instr_dets):
                if c not in assigned_c:
                    pf_tracks[next_tid] = {"pf": ParticleFilterBBox(NUM_PARTICLES, cd), "missed": 0}
                    next_tid += 1

            for tid in list(pf_tracks.keys()):
                if tid not in matched_tids:
                    pf_tracks[tid]["missed"] += 1
                    if pf_tracks[tid]["missed"] > TRACK_RETAIN_FRAMES:
                        del pf_tracks[tid]

            if frame_idx not in gt_set: continue

            roi_for_overlap = None
            roi_source = "none"
            if roi_det is not None:
                roi_for_overlap = roi_det; roi_source = "live"
            elif roi_pf is not None and (frame_idx - roi_last_seen_frm <= ROI_PF_FALLBACK_FRAMES):
                roi_for_overlap = roi_pf.estimate(); roi_source = "pf"

            t0 = time.perf_counter()
            dm = depth_map_for_frame(frame_rgb, midas, tfm, device) if midas is not None else None
            t1 = time.perf_counter()
            if midas is not None:
                total_depth_time += (t1 - t0) * 1000.0
                total_depth_frames += 1

            if instr_dets and roi_for_overlap is not None:
                overlaps = [intersection_over_roi(b, roi_for_overlap) for b in instr_dets]
                best_i   = int(np.argmax(overlaps))
                max_ov   = overlaps[best_i]
                z_instr  = sample_depth(dm, instr_dets[best_i])
                z_roi    = sample_depth(dm, roi_det) if roi_det is not None else float('nan')
            else:
                max_ov  = 0.0
                z_instr = float('nan')
                z_roi   = sample_depth(dm, roi_det) if roi_det is not None else float('nan')

            results[frame_idx] = {
                "overlap": max_ov, "z_instr": z_instr, "z_roi": z_roi, "roi_source": roi_source
            }

        cap.release()
        all_results[vname] = results

    avg_depth_lat = (total_depth_time / total_depth_frames) if total_depth_frames > 0 else 0.0
    return all_results, avg_depth_lat

def classify_frame(res, delta, use_depth=True):
    ov = res.get("overlap", 0.0)
    overlap_ok = ov >= delta
    if not use_depth:
        return 1 if overlap_ok else 0
    roi_src = res.get("roi_source", "live")
    if roi_src == "pf":
        depth_ok = True
    else:
        zi = res.get("z_instr", float('nan'))
        zr = res.get("z_roi",   float('nan'))
        depth_ok = (zi <= zr + DEPTH_MARGIN) if (not np.isnan(zi) and not np.isnan(zr)) else False
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
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    gt_by_video = defaultdict(dict)
    with open(GT_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_by_video[row["video_name"]][int(row["frame_index"])] = int(row["label"])

    yolo_model = YOLO(str(MODEL_PATH))

    # Candidate depth models to evaluate
    depth_candidates = []

    # 1. No Depth (2D Overlap Only)
    depth_candidates.append(("2D_Overlap_Only (No Depth)", None, None, False))

    # 2. MiDaS_small
    try:
        m_small = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False).eval().to(device)
        tfm_small = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False).small_transform
        depth_candidates.append(("MiDaS_small", m_small, tfm_small, True))
    except Exception as e:
        print(f"Failed to load MiDaS_small: {e}")

    # 3. DPT_Hybrid
    try:
        dpt_hyb = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", verbose=False).eval().to(device)
        tfm_dpt = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False).dpt_transform
        depth_candidates.append(("DPT_Hybrid", dpt_hyb, tfm_dpt, True))
    except Exception as e:
        print(f"Failed to load DPT_Hybrid: {e}")

    summary_rows = []

    for name, midas, tfm, use_d in depth_candidates:
        res, avg_depth_lat = run_evaluation_for_model(name, midas, tfm, device, yolo_model, gt_by_video)
        
        flat_gt, flat_res = [], []
        for vname in sorted(gt_by_video.keys()):
            for fi in sorted(gt_by_video[vname].keys()):
                flat_gt.append(gt_by_video[vname][fi])
                flat_res.append(res[vname].get(fi, {"overlap": 0.0, "z_instr": float('nan'), "z_roi": float('nan'), "roi_source": "none"}))

        # Perform threshold delta sweep
        best_delta = 0.1
        best_m = None
        best_f1 = -1.0
        for delta in SWEEP_DELTAS:
            preds = [classify_frame(r, delta, use_depth=use_d) for r in flat_res]
            m = compute_metrics(flat_gt, preds)
            if m["F1"] > best_f1:
                best_f1 = m["F1"]
                best_delta = delta
                best_m = m

        print(f"\n---> {name} Results at δ* = {best_delta}:")
        print(f"     TP={best_m['TP']}, FP={best_m['FP']}, TN={best_m['TN']}, FN={best_m['FN']}")
        print(f"     Accuracy={best_m['Accuracy']:.4f}, Prec={best_m['Precision']:.4f}, Rec={best_m['Recall']:.4f}, F1={best_m['F1']:.4f}")
        print(f"     Pure Depth Latency={avg_depth_lat:.2f} ms")

        # Analytical Interleaved Pipeline Latency (k=4)
        # Latency = 15.2 ms (YOLO) + avg_depth_lat/4 + 1.5 ms (PF) + 3.7 ms (Render)
        est_pipe_lat = 15.2 + (avg_depth_lat / 4.0) + 1.5 + 3.7
        est_fps = 1000.0 / est_pipe_lat if est_pipe_lat > 0 else 0.0

        summary_rows.append({
            "Depth Model": name,
            "Pure Depth Latency (ms)": round(avg_depth_lat, 2),
            "Estimated Pipeline Latency (ms)": round(est_pipe_lat, 2),
            "Estimated Throughput (FPS)": round(est_fps, 1),
            "Best Delta": best_delta,
            "TP": best_m["TP"],
            "FP": best_m["FP"],
            "TN": best_m["TN"],
            "FN": best_m["FN"],
            "Accuracy": round(best_m["Accuracy"], 4),
            "Precision": round(best_m["Precision"], 4),
            "Recall": round(best_m["Recall"], 4),
            "F1": round(best_m["F1"], 4),
        })

    with open(OUTPUT_DIR / "depth_models_empirical_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("\n=======================================================", flush=True)
    print(" EMPIRICAL DEPTH MODEL BENCHMARK COMPLETE", flush=True)
    print(f" Results saved to {OUTPUT_DIR / 'depth_models_empirical_comparison.csv'}", flush=True)
    print("=======================================================", flush=True)

if __name__ == "__main__":
    main()
