import csv
import numpy as np
from collections import defaultdict

with open('E:/Antigravity/TMRB/occlusion_analysis/occlusion_predictions.csv') as f:
    rows = list(csv.DictReader(f))

for vname in ['video_1.mp4', 'video_2.mp4']:
    vrows = [r for r in rows if r['video_name'] == vname]
    overlaps = [float(r['roi_overlap']) for r in vrows]
    gt_labels = [int(r['gt_label']) for r in vrows]
    pos_overlaps = [float(r['roi_overlap']) for r in vrows if r['gt_label']=='1']
    neg_overlaps = [float(r['roi_overlap']) for r in vrows if r['gt_label']=='0']
    print(f"=== {vname} ===")
    print(f"  GT rows: {len(vrows)}, Positive: {sum(gt_labels)}, Negative: {len(vrows)-sum(gt_labels)}")
    print(f"  Overlap mean={np.mean(overlaps):.4f}, max={np.max(overlaps):.4f}")
    print(f"  Overlap on GT=1 frames: {[round(o,4) for o in pos_overlaps]}")
    print(f"  Neg frames with overlap>0.001: {sum(1 for o in neg_overlaps if o>0.001)}")
    print(f"  z_instr on GT=1 frames: {[round(float(r['z_instr']),2) for r in vrows if r['gt_label']=='1'][:10]}")
    print(f"  z_roi on GT=1 frames:   {[round(float(r['z_roi']),2) for r in vrows if r['gt_label']=='1'][:10]}")
    print()
