"""
Recompute all metrics from the existing predictions CSV (no model re-run needed).
Uses diagonal-normalised 3D distance.
"""
import csv, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path("E:/Antigravity/TMRB/occlusion_analysis")

# ── Load predictions ──
rows = list(csv.DictReader(open(OUT / "occlusion_predictions.csv")))

# ── Video dims + FPS ──
VID_INFO = {}
for vn, vp in [("video_1.mp4", "E:/Antigravity/TMRB/video_1.mp4"),
               ("video_2.mp4", "E:/Antigravity/TMRB/video_2.mp4")]:
    cap = cv2.VideoCapture(vp)
    VID_INFO[vn] = {
        "w"  : int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h"  : int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()

# ── Build aligned lists with normalised distances ──
gt_list, dist_list, vname_list, fidx_list = [], [], [], []
for r in rows:
    gt = int(r["ground_truth"])
    if gt == -1:
        continue
    vn   = r["video_name"]
    w, h = VID_INFO[vn]["w"], VID_INFO[vn]["h"]
    diag = float(np.sqrt(w**2 + h**2))
    d    = r["distance"]
    nd   = float(d) / diag if d != "nan" else np.nan
    gt_list.append(gt)
    dist_list.append(nd)
    vname_list.append(vn)
    fidx_list.append(int(r["frame_index"]))

# ── Metrics sweep ──
DELTAS = [0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.28, 0.30, 0.35, 0.40, 0.50, 0.60]
delta_metrics = {}

print("\n  delta     TP    FP    TN    FN     Acc    Prec   Rec    F1")
print("  " + "-"*62)

for delta in DELTAS:
    preds = [1 if (not np.isnan(d) and d < delta) else 0 for d in dist_list]
    TP    = sum(1 for g, p in zip(gt_list, preds) if g == 1 and p == 1)
    FP    = sum(1 for g, p in zip(gt_list, preds) if g == 0 and p == 1)
    TN    = sum(1 for g, p in zip(gt_list, preds) if g == 0 and p == 0)
    FN    = sum(1 for g, p in zip(gt_list, preds) if g == 1 and p == 0)
    total = TP + FP + TN + FN
    acc   = (TP + TN) / total if total > 0 else 0
    prec  = TP / (TP + FP)   if (TP + FP) > 0 else 0
    rec   = TP / (TP + FN)   if (TP + FN) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    delta_metrics[delta] = dict(TP=TP, FP=FP, TN=TN, FN=FN,
                                 Accuracy=acc, Precision=prec,
                                 Recall=rec, F1=f1)
    print(f"  {delta:.2f}    {TP:>4}  {FP:>4}  {TN:>4}  {FN:>4}  "
          f" {acc:.3f}  {prec:.3f}  {rec:.3f}  {f1:.3f}")

best_delta = max(delta_metrics, key=lambda d: delta_metrics[d]["F1"])
bm = delta_metrics[best_delta]
print(f"\n  BEST delta = {best_delta:.2f}   F1={bm['F1']:.4f}  "
      f"P={bm['Precision']:.4f}  R={bm['Recall']:.4f}  Acc={bm['Accuracy']:.4f}")

# ── Save CSVs ──
fields = ["delta"] + list(bm.keys())
with open(OUT / "occlusion_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for d, m in delta_metrics.items():
        w.writerow({"delta": d, **m})

with open(OUT / "occlusion_best_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow({"delta": best_delta, **bm})
print("\n  [SAVED] occlusion_metrics.csv + occlusion_best_metrics.csv")

# ── PR Curve ──
precisions = [delta_metrics[d]["Precision"] for d in DELTAS]
recalls    = [delta_metrics[d]["Recall"]    for d in DELTAS]
f1s        = [delta_metrics[d]["F1"]        for d in DELTAS]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Step 2.3 - Threshold (delta) Analysis  [Normalised 3D Distance]",
             fontsize=13, fontweight="bold")

ax = axes[0]
sc = ax.scatter(recalls, precisions, c=DELTAS, cmap="plasma", s=120,
                zorder=3, edgecolors="white")
ax.plot(recalls, precisions, "k--", lw=0.8, alpha=0.5)
for d, p, r in zip(DELTAS, precisions, recalls):
    ax.annotate(f"d={d:.2f}", (r, p), textcoords="offset points",
                xytext=(5, 4), fontsize=7)
ax.scatter([bm["Recall"]], [bm["Precision"]], c="lime", s=220, zorder=5,
           edgecolors="black", lw=1.5,
           label=f"Best d={best_delta:.2f}  F1={bm['F1']:.3f}")
ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve across delta values")
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
plt.colorbar(sc, ax=ax, label="delta value")

ax2 = axes[1]
ax2.plot(DELTAS, f1s, "o-", color="#4c72b0", lw=2, ms=8)
ax2.axvline(best_delta, color="lime", ls="--", lw=2, label=f"Best d={best_delta:.2f}")
ax2.fill_between(DELTAS, f1s, alpha=0.15, color="#4c72b0")
ax2.set_xlabel("delta (normalised distance threshold)", fontsize=12)
ax2.set_ylabel("F1-Score", fontsize=12)
ax2.set_title("F1-Score vs delta")
ax2.grid(True, alpha=0.3); ax2.legend(); ax2.set_ylim(0, 0.5)
plt.tight_layout()
plt.savefig(OUT / "occlusion_pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] occlusion_pr_curve.png")

# ── Confusion Matrix ──
cm = np.array([[bm["TN"], bm["FP"]], [bm["FN"], bm["TP"]]])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
classes = ["Not Occluded (0)", "Occluded (1)"]
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(classes, fontsize=10)
ax.set_yticklabels(classes, fontsize=10)
ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Ground Truth", fontsize=12)
ax.set_title(
    f"Confusion Matrix  (delta = {best_delta:.2f})\n"
    f"Acc={bm['Accuracy']:.3f}  P={bm['Precision']:.3f}  "
    f"R={bm['Recall']:.3f}  F1={bm['F1']:.3f}",
    fontsize=11)
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig(OUT / "occlusion_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] occlusion_confusion.png")

# ── Detection Delay (Step 2.4) using best delta ──
print("\n  --- STEP 2.4 Detection Delay ---")
best_preds = [1 if (not np.isnan(d) and d < best_delta) else 0 for d in dist_list]

# Group by video
by_video = {}
for gt, pred, vn, fi in zip(gt_list, best_preds, vname_list, fidx_list):
    by_video.setdefault(vn, []).append((fi, gt, pred))

all_delays = []
for vn, data in by_video.items():
    fps = VID_INFO[vn]["fps"]
    data.sort(key=lambda x: x[0])
    frame_indices = [x[0] for x in data]
    gts           = [x[1] for x in data]
    preds         = [x[2] for x in data]

    in_event = False
    event_id = 0
    event_start_ptr = None
    event_start_frame = None

    for i, (fi, gt, pred) in enumerate(data):
        if not in_event and gt == 1:
            in_event = True
            event_id += 1
            event_start_ptr = i
            event_start_frame = fi
        elif in_event and gt == 0:
            # find first prediction=1 in event
            detect_ptr = None
            for j in range(event_start_ptr, i):
                if preds[j] == 1:
                    detect_ptr = j
                    break
            if detect_ptr is not None:
                delay_fr = detect_ptr - event_start_ptr
                detected = True
            else:
                delay_fr = i - event_start_ptr
                detected = False
            all_delays.append({
                "video"         : vn,
                "event_id"      : event_id,
                "gt_start_frame": event_start_frame,
                "detect_frame"  : data[detect_ptr][0] if detected else data[i-1][0],
                "delay_frames"  : delay_fr,
                "delay_ms"      : round(delay_fr * 1000 / fps, 1),
                "detected"      : detected,
            })
            in_event = False

    # Event at end of video
    if in_event:
        detect_ptr = None
        for j in range(event_start_ptr, len(data)):
            if preds[j] == 1:
                detect_ptr = j
                break
        delay_fr = (detect_ptr - event_start_ptr) if detect_ptr else (len(data) - event_start_ptr)
        all_delays.append({
            "video"         : vn,
            "event_id"      : event_id,
            "gt_start_frame": event_start_frame,
            "detect_frame"  : data[detect_ptr][0] if detect_ptr else data[-1][0],
            "delay_frames"  : delay_fr,
            "delay_ms"      : round(delay_fr * 1000 / fps, 1),
            "detected"      : detect_ptr is not None,
        })

if all_delays:
    delay_frames = [d["delay_frames"] for d in all_delays]
    delay_ms     = [d["delay_ms"]     for d in all_delays]
    detected     = [d["detected"]     for d in all_delays]

    print(f"\n  Total events    : {len(all_delays)}")
    print(f"  Detected        : {sum(detected)}")
    print(f"  Missed          : {sum(1 for x in detected if not x)}")
    print(f"  Mean delay (fr) : {np.mean(delay_frames):.2f}")
    print(f"  Mean delay (ms) : {np.mean(delay_ms):.1f}")
    print(f"  Min  delay (ms) : {np.min(delay_ms):.1f}")
    print(f"  Max  delay (ms) : {np.max(delay_ms):.1f}")
    print()
    print(f"  {'Video':<14} {'Event':>5} {'GT Start':>9} {'Delay(fr)':>10} {'Delay(ms)':>10} {'Status':>9}")
    print("  " + "-"*62)
    for d in all_delays:
        status = "DETECTED" if d["detected"] else "MISSED"
        print(f"  {d['video']:<14} {d['event_id']:>5} {d['gt_start_frame']:>9} "
              f"{d['delay_frames']:>10} {d['delay_ms']:>10} {status:>9}")

    with open(OUT / "occlusion_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_delays[0].keys()))
        w.writeheader(); w.writerows(all_delays)
    print(f"\n  [SAVED] occlusion_latency.csv")

    # Latency plot
    event_labels = [f"{d['video'][-5]}-E{d['event_id']}" for d in all_delays]
    colors = ["#2ecc71" if x else "#e74c3c" for x in detected]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Step 2.4 - Detection Delay per Occlusion Event",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.bar(event_labels, delay_frames, color=colors, edgecolor="white")
    ax.axhline(np.mean(delay_frames), color="orange", ls="--", lw=1.5,
               label=f"Mean={np.mean(delay_frames):.1f} frames")
    ax.set_xlabel("Occlusion Event"); ax.set_ylabel("Delay (frames)")
    ax.set_title("Detection Delay in Frames")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(event_labels, delay_ms, color=colors, edgecolor="white")
    ax2.axhline(np.mean(delay_ms), color="orange", ls="--", lw=1.5,
                label=f"Mean={np.mean(delay_ms):.1f} ms")
    ax2.set_xlabel("Occlusion Event"); ax2.set_ylabel("Delay (ms)")
    ax2.set_title("Detection Delay in Milliseconds")
    ax2.tick_params(axis="x", rotation=45); ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "occlusion_latency_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] occlusion_latency_plot.png")

print("\n  ===== COMPLETE =====")
print(f"  BEST delta = {best_delta:.2f}  (normalised 3D distance)")
print(f"  Acc={bm['Accuracy']:.3f}  P={bm['Precision']:.3f}  R={bm['Recall']:.3f}  F1={bm['F1']:.3f}")
print(f"  TP={bm['TP']}  FP={bm['FP']}  TN={bm['TN']}  FN={bm['FN']}")
