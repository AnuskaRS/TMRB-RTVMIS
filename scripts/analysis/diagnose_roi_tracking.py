"""
Deep diagnostic: for every GT=1 frame, show:
1. When the ROI was last seen before this frame
2. What the PF estimate is
3. What overlap that gives with each instrument
4. Whether the PF fallback would fire under ROI_PF_FALLBACK_FRAMES=30
"""
import cv2, csv, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

BASE_DIR            = Path("E:/Antigravity/TMRB")
GT_CSV_PATH         = BASE_DIR / "occlusion_gt.csv"
MODEL_PATH          = BASE_DIR / "best1.pt"
ROI_CLASS_ID        = 0
NUM_PARTICLES       = 100
PF_SIGMA            = 10.0
ROI_PF_FALLBACK_FRAMES = 30
np.random.seed(42); torch.manual_seed(42)

model = YOLO(str(MODEL_PATH))

gt_by_video = defaultdict(dict)
with open(GT_CSV_PATH) as f:
    for row in csv.DictReader(f):
        gt_by_video[row['video_name']][int(row['frame_index'])] = int(row['label'])

def bbox_center(b):
    return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

class PF:
    def __init__(self, n, b):
        self.n=n; self.pts=np.random.randn(n,4)+np.array(b); self.w=np.ones(n)/n
    def predict(self):
        self.pts += np.random.randn(self.n,4)*2.0
    def update(self, b):
        d=np.linalg.norm(self.pts-np.array(b),axis=1)
        self.w=np.exp(-d/PF_SIGMA); s=self.w.sum()
        self.w=self.w/s if s>0 else np.ones(self.n)/self.n
    def resample(self):
        if 1.0/(self.w**2).sum()<self.n/2:
            idx=np.random.choice(self.n,self.n,p=self.w)
            self.pts=self.pts[idx]; self.w=np.ones(self.n)/self.n
    def estimate(self): return np.average(self.pts,weights=self.w,axis=0)

def ior(bi, br):
    if bi is None or br is None: return 0.0
    x1=max(bi[0],br[0]); y1=max(bi[1],br[1])
    x2=min(bi[2],br[2]); y2=min(bi[3],br[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area=max(1,(br[2]-br[0])*(br[3]-br[1]))
    return inter/area

# Only run on video_1 for speed
vname = 'video_1.mp4'
gt_frames = gt_by_video[vname]
pos_set   = set(fi for fi, lbl in gt_frames.items() if lbl == 1)
max_fi    = max(gt_frames.keys())

cap = cv2.VideoCapture(str(BASE_DIR / vname))
roi_pf            = None
roi_last_seen_frm = -999
roi_ever_detected = False
roi_det_count     = 0

print(f"Scanning video_1.mp4 — {max_fi+1} frames")
print("First scan: when is ROI (class 0) detected?")

# First pass: count ROI detections up to first GT=1 frame
first_pos = min(pos_set)
roi_detections_before_first_occ = []

for frame_idx in range(min(first_pos+1, max_fi+1)):
    ret, frame = cap.read()
    if not ret: break
    with torch.no_grad():
        preds = model(frame, verbose=False)
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
        roi_detections_before_first_occ.append(frame_idx)
        roi_ever_detected = True
        roi_det_count += 1

print(f"\nROI class 0 detected on {roi_det_count} frames before frame {first_pos}")
if roi_detections_before_first_occ:
    print(f"Last ROI detection before first occlusion: frame {roi_detections_before_first_occ[-1]}")
    print(f"Distance to first GT=1 frame ({first_pos}): {first_pos - roi_detections_before_first_occ[-1]} frames")
    print(f"PF fallback would fire: {first_pos - roi_detections_before_first_occ[-1] <= ROI_PF_FALLBACK_FRAMES}")
else:
    print("ROI was NEVER detected before the first occlusion event!")

cap.release()
