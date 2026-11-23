import csv, numpy as np
from collections import Counter

print('=== BEST METRICS (occlusion_best_metrics.csv) ===')
with open('E:/Antigravity/TMRB/occlusion_analysis/occlusion_best_metrics.csv') as f:
    for r in csv.DictReader(f):
        d = dict(r)
        print(f"  delta={d['delta']}  TP={d['TP']}  FP={d['FP']}  TN={d['TN']}  FN={d['FN']}")
        print(f"  Accuracy={float(d['Accuracy']):.4f}  Precision={float(d['Precision']):.4f}  Recall={float(d['Recall']):.4f}  F1={float(d['F1']):.4f}")

print()
print('=== FULL DELTA SWEEP (occlusion_metrics.csv) ===')
with open('E:/Antigravity/TMRB/occlusion_analysis/occlusion_metrics.csv') as f:
    rows = list(csv.DictReader(f))
print(f"  {'delta':>8}  {'TP':>4}  {'FP':>5}  {'TN':>5}  {'FN':>4}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
for r in rows:
    print(f"  {float(r['delta']):8.4f}  {r['TP']:>4}  {r['FP']:>5}  {r['TN']:>5}  {r['FN']:>4}  "
          f"{float(r['Accuracy']):7.4f}  {float(r['Precision']):7.4f}  {float(r['Recall']):7.4f}  {float(r['F1']):7.4f}")

print()
print('=== LATENCY (occlusion_latency.csv) ===')
with open('E:/Antigravity/TMRB/occlusion_analysis/occlusion_latency.csv') as f:
    lat = list(csv.DictReader(f))

det  = [r for r in lat if r['status'] == 'Detected']
miss = [r for r in lat if r['status'] == 'Missed']
print(f"  Total events: {len(lat)}  |  Detected: {len(det)}  |  Missed: {len(miss)}")
delays = [float(r['delay_ms']) for r in det]
if delays:
    print(f"  Latency (detected) — mean={np.mean(delays):.1f} ms  median={np.median(delays):.1f} ms  std={np.std(delays):.1f} ms")

print()
for r in lat:
    flag = 'OK' if r['status'] == 'Detected' else 'XX'
    d_str = f"{float(r['delay_ms']):.1f} ms" if r['status'] == 'Detected' else 'Missed'
    print(f"  [{flag}] {r['event_id']} ({r['video_name']}) "
          f"frames {r['gt_start_frame']}-{r['gt_end_frame']}  delay={d_str}")

print()
print('=== PREDICTIONS SUMMARY (occlusion_predictions.csv) ===')
with open('E:/Antigravity/TMRB/occlusion_analysis/occlusion_predictions.csv') as f:
    preds = list(csv.DictReader(f))
print(f"  Total GT frames evaluated: {len(preds)}")
print(f"  GT label distribution:  {Counter(r['gt_label'] for r in preds)}")
overlaps = [float(r['roi_overlap']) for r in preds]
print(f"  Overlap range: min={min(overlaps):.4f}  max={max(overlaps):.4f}  mean={np.mean(overlaps):.4f}")
pos_overlaps = [float(r['roi_overlap']) for r in preds if r['gt_label']=='1']
print(f"  Overlap on GT=1 frames: {[round(o,3) for o in pos_overlaps]}")
