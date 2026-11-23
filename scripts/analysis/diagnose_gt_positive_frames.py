"""
Diagnostic: for every GT=1 frame, run YOLO and show exactly what it detects.
This tells us WHY overlap = 0.0 on occluded frames.
"""
import cv2, csv, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

BASE_DIR     = Path("E:/Antigravity/TMRB")
GT_CSV_PATH  = BASE_DIR / "occlusion_gt.csv"
MODEL_PATH   = BASE_DIR / "best1.pt"
ROI_CLASS_ID = 0

model = YOLO(str(MODEL_PATH))

gt_by_video = defaultdict(dict)
with open(GT_CSV_PATH) as f:
    for row in csv.DictReader(f):
        gt_by_video[row['video_name']][int(row['frame_index'])] = int(row['label'])

for vname in sorted(gt_by_video.keys()):
    gt_frames = gt_by_video[vname]
    pos_frames = sorted(fi for fi, lbl in gt_frames.items() if lbl == 1)
    if not pos_frames:
        continue

    cap = cv2.VideoCapture(str(BASE_DIR / vname))
    print(f"\n{'='*65}")
    print(f"  {vname}  — {len(pos_frames)} occluded frames")
    print(f"{'='*65}")
    print(f"  {'frame':>6}  {'ROI_det':>8}  {'n_instr':>7}  {'IoR':>6}  {'note'}")
    print(f"  {'─'*55}")

    for fi in pos_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"  {fi:6d}  [FRAME READ FAILED]")
            continue

        with torch.no_grad():
            preds = model(frame, verbose=False)
        boxes = preds[0].boxes

        roi_box    = None
        instr_dets = []

        if boxes is not None and len(boxes):
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs   = boxes.conf.cpu().numpy()
            xyxy    = boxes.xyxy.cpu().numpy()
            for cid, conf, box in zip(cls_ids, confs, xyxy):
                if cid == ROI_CLASS_ID and roi_box is None:
                    roi_box = (box, conf)
                elif cid != ROI_CLASS_ID:
                    instr_dets.append((box, cid, conf))

        # Compute IoR if both present
        ior = 0.0
        if roi_box and instr_dets:
            rb = roi_box[0]
            for ib, _, _ in instr_dets:
                x1 = max(ib[0], rb[0]); y1 = max(ib[1], rb[1])
                x2 = min(ib[2], rb[2]); y2 = min(ib[3], rb[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                area  = max(1, (rb[2]-rb[0])*(rb[3]-rb[1]))
                ior = max(ior, inter/area)

        roi_str   = f"{roi_box[1]:.2f}" if roi_box else "NONE"
        instr_str = str(len(instr_dets))
        instr_detail = ""
        if instr_dets:
            instr_detail = " | " + ", ".join(f"cls{cid}({cf:.2f})" for _, cid, cf in instr_dets[:3])
        note = ""
        if roi_box is None and instr_dets:
            note = "<-- ROI MISSING, instrument present"
        elif roi_box is not None and not instr_dets:
            note = "<-- ROI ok, NO instrument"
        elif roi_box is None and not instr_dets:
            note = "<-- BOTH MISSING"

        print(f"  {fi:6d}  {roi_str:>8}  {instr_str:>7}  {ior:6.3f}  {note}{instr_detail}")

    cap.release()

print("\nDone.")
