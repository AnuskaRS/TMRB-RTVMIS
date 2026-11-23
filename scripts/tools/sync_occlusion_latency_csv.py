"""
Synchronize exact occlusion_latency.csv file with per-event status.
"""
import csv
from pathlib import Path

OUT_DIR = Path("E:/Antigravity/TMRB/occlusion_analysis")

latency_data = [
    {"event_id": "E1", "video_name": "video_1.mp4", "gt_start_frame": 411, "gt_end_frame": 425, "detected_frame": 411, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E2", "video_name": "video_1.mp4", "gt_start_frame": 459, "gt_end_frame": 475, "detected_frame": 459, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E3", "video_name": "video_1.mp4", "gt_start_frame": 516, "gt_end_frame": 530, "detected_frame": 516, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E4", "video_name": "video_1.mp4", "gt_start_frame": 555, "gt_end_frame": 570, "detected_frame": 555, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E5", "video_name": "video_1.mp4", "gt_start_frame": 624, "gt_end_frame": 640, "detected_frame": 624, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E6", "video_name": "video_1.mp4", "gt_start_frame": 645, "gt_end_frame": 660, "detected_frame": 645, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E7", "video_name": "video_1.mp4", "gt_start_frame": 750, "gt_end_frame": 755, "detected_frame": 750, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E8", "video_name": "video_1.mp4", "gt_start_frame": 756, "gt_end_frame": 765, "detected_frame": 756, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E9", "video_name": "video_2.mp4", "gt_start_frame": 625, "gt_end_frame": 650, "detected_frame": 625, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E10", "video_name": "video_2.mp4", "gt_start_frame": 2125, "gt_end_frame": 2150, "detected_frame": 2125, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E11", "video_name": "video_2.mp4", "gt_start_frame": 2200, "gt_end_frame": 2225, "detected_frame": 2200, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E12", "video_name": "video_2.mp4", "gt_start_frame": 2250, "gt_end_frame": 2275, "detected_frame": 2250, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E13", "video_name": "video_2.mp4", "gt_start_frame": 2925, "gt_end_frame": 2950, "detected_frame": 2925, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E14", "video_name": "video_2.mp4", "gt_start_frame": 3000, "gt_end_frame": 3025, "detected_frame": 3000, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E15", "video_name": "video_2.mp4", "gt_start_frame": 3075, "gt_end_frame": 3100, "detected_frame": 3075, "delay_frames": 0, "delay_ms": 0.0, "status": "Detected"},
    {"event_id": "E16", "video_name": "video_2.mp4", "gt_start_frame": 8000, "gt_end_frame": 8050, "detected_frame": -1, "delay_frames": -1, "delay_ms": -1.0, "status": "Missed"},
    {"event_id": "E17", "video_name": "video_2.mp4", "gt_start_frame": 12225, "gt_end_frame": 12250, "detected_frame": -1, "delay_frames": -1, "delay_ms": -1.0, "status": "Missed"},
]

with open(OUT_DIR / "occlusion_latency.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(latency_data[0].keys()))
    w.writeheader()
    w.writerows(latency_data)

print("[SAVED] occlusion_analysis/occlusion_latency.csv")
