import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Outputs
OUTPUT_PATH = Path("E:/Antigravity/TMRB/sarka6.png")
ARTIFACT_PATH = Path("C:/Users/ANUSKA/.gemini/antigravity-ide/brain/2f9fa35d-b537-4d0e-91a9-b9c745eb4cb1/sarka6.png")

# Set up 2-panel figure with compact vertical height for squeezed bars
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 6.4), dpi=300, gridspec_kw={'height_ratios': [1.25, 1.0]})

# Adjust subplot layout margins
plt.subplots_adjust(top=0.92, bottom=0.10, hspace=0.45, left=0.33, right=0.96)

# Color & Hatch definitions
color_det   = "#f39c12"  # Warm Orange / Peach
color_depth = "#2ecc71"  # Emerald Green
color_pf    = "#3498db"  # Sky Blue
color_rend  = "#e74c3c"  # Coral Red

hatch_det   = "..."
hatch_depth = "ooo"
hatch_pf    = "xxx"
hatch_rend  = "///"

# -------------------------------------------------------------------------
# PANEL A: Sub-Module Latency Breakdown & Total Sequential Runtime
# -------------------------------------------------------------------------
modules = [
    "Tool/ROI Detection (YOLOv10)",
    "Depth Estimation (MiDaS)",
    "Particle Filtering (PF)",
    "Visualization & Rendering",
    "Total Sequential Runtime"
]

y1 = np.arange(len(modules))[::-1]

# Thicker bars squeezed close together
bh = 0.78

det_v   = 21.0
depth_v = 36.5
pf_v    = 1.5
rend_v  = 11.0
total_v = 70.0  # 21.0 + 36.5 + 1.5 + 11.0 = 70.0 ms

# Draw individual module bars
ax1.barh(y1[0], det_v,   height=bh, color=color_det,   edgecolor="black", linewidth=1.2, hatch=hatch_det)
ax1.barh(y1[1], depth_v, height=bh, color=color_depth, edgecolor="black", linewidth=1.2, hatch=hatch_depth)
ax1.barh(y1[2], pf_v,    height=bh, color=color_pf,    edgecolor="black", linewidth=1.2, hatch=hatch_pf)
ax1.barh(y1[3], rend_v,  height=bh, color=color_rend,  edgecolor="black", linewidth=1.2, hatch=hatch_rend)

# Draw Stacked Total Runtime bar
b1 = ax1.barh(y1[4], det_v,   height=bh, color=color_det,   edgecolor="black", linewidth=1.2, hatch=hatch_det, label="Tool/ROI Detection (21.0 ms)")
b2 = ax1.barh(y1[4], depth_v, left=det_v, height=bh, color=color_depth, edgecolor="black", linewidth=1.2, hatch=hatch_depth, label="Depth Estimation (36.5 ms)")
b3 = ax1.barh(y1[4], pf_v,    left=det_v+depth_v, height=bh, color=color_pf, edgecolor="black", linewidth=1.2, hatch=hatch_pf, label="Particle Filtering (1.5 ms)")
b4 = ax1.barh(y1[4], rend_v,  left=det_v+depth_v+pf_v, height=bh, color=color_rend, edgecolor="black", linewidth=1.2, hatch=hatch_rend, label="Visualization & Post-proc (11.0 ms)")

# Value annotations (Large & Crisp)
ax1.text(det_v + 1.5,   y1[0], f"{det_v:.1f} ms",   va="center", fontsize=12.0, fontweight="bold")
ax1.text(depth_v + 1.5, y1[1], f"{depth_v:.1f} ms", va="center", fontsize=12.0, fontweight="bold")
ax1.text(pf_v + 1.5,    y1[2], f"{pf_v:.1f} ms",    va="center", fontsize=12.0, fontweight="bold")
ax1.text(rend_v + 1.5,  y1[3], f"{rend_v:.1f} ms",  va="center", fontsize=12.0, fontweight="bold")
ax1.text(total_v + 2.0, y1[4], f"{total_v:.1f} ms (14.3 FPS)", va="center", fontsize=12.5, fontweight="bold", color="#c0392b")

# Threshold lines
l_clin = ax1.axvline(100.0, color="#d32f2f", linestyle="--", linewidth=2.4, label="Clinical Threshold (100 ms)")
l_rt   = ax1.axvline(33.3,  color="#d35400", linestyle=":",  linewidth=2.2, label="Real-Time Target (33.3 ms = 30 FPS)")

ax1.set_yticks(y1)
ax1.set_yticklabels(modules, fontsize=12.5, fontweight="bold")
ax1.set_xlabel("Latency (ms)", fontsize=13.0, fontweight="bold", labelpad=4)
ax1.set_title("(a) Sub-Module Latency Breakdown & Total Sequential Runtime (70.0 ms)", fontsize=13.5, fontweight="bold", pad=10)
ax1.set_xlim(0, 165)
ax1.set_ylim(-0.6, 4.6)
ax1.tick_params(axis='x', labelsize=12.0)
ax1.grid(True, axis="x", alpha=0.3, linestyle="--")

# Clean, spacious Legend placed inside the empty upper-right region of Panel A (no squeezing)
ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.95, edgecolor="gray")

# -------------------------------------------------------------------------
# PANEL B: Operating Modes Latency & Throughput Comparison
# -------------------------------------------------------------------------
modes = [
    "Sequential Mode (Full Depth)",
    "Interleaved Mode (Sampling 1/4x)",
    "Tracking-Only Mode (No Depth)"
]

y2 = np.arange(len(modes))[::-1]

det_m   = np.array([21.0, 15.2, 12.0])
depth_m = np.array([36.5,  5.5,  0.0])
pf_m    = np.array([ 1.5,  1.5,  1.5])
rend_m  = np.array([11.0,  3.7,  4.3])

totals_m = det_m + depth_m + pf_m + rend_m  # [70.0, 25.9, 17.8]
fps_m    = [1000.0 / t for t in totals_m]    # [14.3, 38.7, 56.1]

ax2.barh(y2, det_m,   height=bh, color=color_det,   edgecolor="black", linewidth=1.2, hatch=hatch_det)
ax2.barh(y2, depth_m, left=det_m, height=bh, color=color_depth, edgecolor="black", linewidth=1.2, hatch=hatch_depth)
ax2.barh(y2, pf_m,    left=det_m+depth_m, height=bh, color=color_pf, edgecolor="black", linewidth=1.2, hatch=hatch_pf)
ax2.barh(y2, rend_m,  left=det_m+depth_m+pf_m, height=bh, color=color_rend, edgecolor="black", linewidth=1.2, hatch=hatch_rend)

for i, (y, tot, fps_v) in enumerate(zip(y2, totals_m, fps_m)):
    tag = " (PASS >30 FPS)" if fps_v >= 30 else " (FAIL <30 FPS)"
    color_tag = "#27ae60" if fps_v >= 30 else "#c0392b"
    ax2.text(tot + 2.0, y, f"{tot:.1f} ms | {fps_v:.1f} FPS{tag}", va="center", ha="left", fontsize=12.0, fontweight="bold", color=color_tag)

ax2.axvline(100.0, color="#d32f2f", linestyle="--", linewidth=2.4)
ax2.axvline(33.3,  color="#d35400", linestyle=":",  linewidth=2.2)

ax2.set_yticks(y2)
ax2.set_yticklabels(modes, fontsize=12.5, fontweight="bold")
ax2.set_xlabel("Processing Time per Frame (ms)", fontsize=13.0, fontweight="bold", labelpad=4)
ax2.set_title("(b) Operating Modes Latency & Throughput Comparison", fontsize=13.5, fontweight="bold", pad=10)
ax2.set_xlim(0, 165)
ax2.set_ylim(-0.6, 2.6)
ax2.tick_params(axis='x', labelsize=12.0)
ax2.grid(True, axis="x", alpha=0.3, linestyle="--")

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.savefig(ARTIFACT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"[SUCCESS] Saved figure with squeezed bars and clean legend to {OUTPUT_PATH} and {ARTIFACT_PATH}")
