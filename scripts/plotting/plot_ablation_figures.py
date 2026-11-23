"""
Generate exact high-resolution PNG plots for pf_analysis matching updated ablation table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("E:/Antigravity/TMRB/pf_analysis")
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Ablation Plot ──
configs = [
    "YOLOv10\nOnly",
    "YOLOv10\n+ PF",
    "YOLOv10 + PF\n+ MiDaS (Interleaved)",
    "YOLOv10 + PF\n+ MiDaS (Full)"
]

tiou = [0.4079, 0.6134, 0.6134, 0.6134]
conf = [0.6680, 0.8339, 0.8339, 0.8339]

x = np.arange(len(configs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
rects1 = ax.bar(x - width/2, tiou, width, label="Temporal IoU", color="#3498db", edgecolor="black", linewidth=0.8)
rects2 = ax.bar(x + width/2, conf, width, label="Mean Confidence", color="#2ecc71", edgecolor="black", linewidth=0.8)

ax.set_ylabel("Metric Score", fontsize=12, fontweight="bold")
ax.set_title("3-Way Ablation Study: Tracking Consistency & Confidence", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.set_ylim(0, 1.05)
ax.grid(True, axis="y", alpha=0.3, linestyle="--")

# Value annotations
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "ablation_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("[GENERATED] pf_analysis/ablation_plot.png")


# ── 2. Runtime Breakdown Plot ──
modes = ["YOLOv10\nOnly", "YOLOv10\n+ PF", "Proposed\nInterleaved 4th", "Proposed\nSequential Full"]
fps_vals = [54.6, 53.8, 38.7, 11.9]
latency_vals = [18.3, 18.6, 25.9, 84.3]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
fig.suptitle("Step 3.2 — Pipeline Latency & Throughput Comparison", fontsize=13, fontweight="bold")

# FPS
ax1 = axes[0]
bars1 = ax1.bar(modes, fps_vals, color=["#9b59b6", "#3498db", "#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.8)
ax1.axhline(30, color="red", linestyle="--", linewidth=1.5, label="Real-Time Target (30 FPS)")
ax1.set_ylabel("Throughput (FPS)", fontsize=11, fontweight="bold")
ax1.set_title("Frame Rate Across Pipeline Configurations", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 65)
ax1.grid(True, axis="y", alpha=0.3)
ax1.legend(loc="upper right", fontsize=10)

for bar in bars1:
    h = bar.get_height()
    ax1.annotate(f"{h:.1f} FPS", (bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Latency (ms)
ax2 = axes[1]
bars2 = ax2.bar(modes, latency_vals, color=["#9b59b6", "#3498db", "#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.8)
ax2.axhline(100, color="red", linestyle="--", linewidth=1.5, label="Clinical Max Latency (100 ms)")
ax2.set_ylabel("Pipeline Latency (ms)", fontsize=11, fontweight="bold")
ax2.set_title("Per-Frame Processing Latency", fontsize=12, fontweight="bold")
ax2.set_ylim(0, 110)
ax2.grid(True, axis="y", alpha=0.3)
ax2.legend(loc="upper left", fontsize=10)

for bar in bars2:
    h = bar.get_height()
    ax2.annotate(f"{h:.1f} ms", (bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "runtime_breakdown.png", dpi=300, bbox_inches="tight")
plt.close()
print("[GENERATED] pf_analysis/runtime_breakdown.png")

print("\n=== PF ANALYSIS PLOTS GENERATED AND SAVED PERFECTLY ===")
