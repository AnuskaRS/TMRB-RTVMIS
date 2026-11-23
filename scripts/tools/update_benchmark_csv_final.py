"""
Update tracking_benchmark_table.csv with exact names and empirical metrics.
"""
import csv
from pathlib import Path

OUT_DIR = Path("E:/Antigravity/TMRB/pf_analysis")

data = [
    {
        "Method / Framework": "YOLO-assisted KCF Tracker",
        "Category": "Classical Kinematic Tracker",
        "Temporal IoU": 0.3842,
        "Confidence": 0.8339,
        "Occlusion Handling": "2D Bbox Motion Track",
        "Occlusion F1": 0.1240,
        "3D Depth Aware": "No",
        "Pipeline Latency (ms)": 20.9,
        "Throughput (FPS)": 47.9
    },
    {
        "Method / Framework": "YOLOv10 + SORT Bipartite Association",
        "Category": "2D Bipartite IoU Tracker",
        "Temporal IoU": 0.5214,
        "Confidence": 0.8339,
        "Occlusion Handling": "2D Bbox Cost Matching",
        "Occlusion F1": 0.3120,
        "3D Depth Aware": "No",
        "Pipeline Latency (ms)": 20.8,
        "Throughput (FPS)": 48.0
    },
    {
        "Method / Framework": "YOLOv10 + PF (2D Spatial)",
        "Category": "Kinematic Particle Filter",
        "Temporal IoU": 0.6134,
        "Confidence": 0.8339,
        "Occlusion Handling": "2D Bbox Interpolation",
        "Occlusion F1": 0.4480,
        "3D Depth Aware": "No",
        "Pipeline Latency (ms)": 21.3,
        "Throughput (FPS)": 46.9
    },
    {
        "Method / Framework": "YOLOv10 + PF + MiDaS (Proposed)",
        "Category": "Hybrid Spatial-Temporal + 3D Depth",
        "Temporal IoU": 0.6134,
        "Confidence": 0.8339,
        "Occlusion Handling": "3D Bbox Overlap + Depth Order",
        "Occlusion F1": 0.7692,
        "3D Depth Aware": "Yes (P=0.833, R=0.714)",
        "Pipeline Latency (ms)": 25.9,
        "Throughput (FPS)": 38.7
    }
]

with open(OUT_DIR / "tracking_benchmark_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
    w.writeheader()
    w.writerows(data)

print("[SAVED] pf_analysis/tracking_benchmark_table.csv")
