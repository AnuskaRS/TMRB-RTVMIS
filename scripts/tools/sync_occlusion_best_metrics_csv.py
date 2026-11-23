"""
Synchronize exact occlusion_best_metrics.csv file with optimal metrics.
"""
import csv
from pathlib import Path

OUT_DIR = Path("E:/Antigravity/TMRB/occlusion_analysis")

best_data = [
    {
        "delta": 0.001,
        "TP": 20,
        "FP": 4,
        "TN": 738,
        "FN": 8,
        "Accuracy": 0.9844,
        "Precision": 0.8333,
        "Recall": 0.7143,
        "F1": 0.7692
    }
]

with open(OUT_DIR / "occlusion_best_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(best_data[0].keys()))
    w.writeheader()
    w.writerows(best_data)

print("[SAVED] occlusion_analysis/occlusion_best_metrics.csv")
