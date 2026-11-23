"""
Deep diagnostic: run the FULL tracking loop on video_1 up through the first
occlusion event and print exactly what the PF estimate is and what IoR
it gives against each instrument on each GT=1 frame.
"""
import cv2, csv, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

BASE_DIR              = Path("E:/Antigravity/TMRB")
GT_CSV_PATH           = BASE_DIR / "occlusion_gt.csv"
MODEL_PATH            = BASE_DIR / "best1.pt"
ROI_CLASS_ID          = 0
NUM_PARTICLES         = 100
PF_SIGMA              = 10.0
MAX_ASSOCIATION_DIST  = 150.0
TRACK_RETAIN_FRAMES   = 5
ROI_PF_FALLBACK_FRAMES = 30
np.random.seed(42); torch.manual_seed(42)

model = YOLO(str(MODEL_PATH))

gt_by_video = defaultdict(dict)
with open(GT_CSV_PATH) as f:
    for row in csv.DictReader(f):
        gt_by_video[row['video_name']][int(row['frame_index'])] = int(row['label'])


def bbox_center(b):
    return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])


class PFBBox:
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
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    area=max(1.0,(br[2]-br[0])*(br[3]-br[1]))
    return inter/area


# Run on video_1, first event: frames 411-420
vname    = 'video_1.mp4'
gt_fms   = gt_by_video[vname]
pos_set  = set(fi for fi, lbl in gt_fms.items() if lbl == 1)
# Only go up to frame 420 for speed
max_scan = 425

cap = cv2.VideoCapture(str(BASE_DIR / vname))
roi_pf            = None
roi_last_seen_frm = -999
pf_tracks         = {}
next_tid          = 1

print(f"Running tracking loop on video_1 up to frame {max_scan} ...")
print()

for frame_idx in range(max_scan + 1):
    ret, frame = cap.read()
    if not ret: break

    roi_det    = None
    instr_dets = []
    with torch.no_grad():
        preds = model(frame, verbose=False)
    boxes = preds[0].boxes
    if boxes is not None and len(boxes):
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs   = boxes.conf.cpu().numpy()
        xyxy    = boxes.xyxy.cpu().numpy()
        for cid, conf, box in zip(cls_ids, confs, xyxy):
            if cid == ROI_CLASS_ID and roi_det is None:
                roi_det = box
            elif cid != ROI_CLASS_ID:
                instr_dets.append((box, cid, conf))

    # ROI PF
    if roi_det is not None:
        if roi_pf is None:
            roi_pf = PFBBox(NUM_PARTICLES, roi_det)
        else:
            roi_pf.predict(); roi_pf.update(roi_det); roi_pf.resample()
        roi_last_seen_frm = frame_idx
    elif roi_pf is not None:
        roi_pf.predict(); roi_pf.resample()

    # Instrument PF predict
    for tid in list(pf_tracks.keys()):
        pf_tracks[tid]["pf"].predict()
        pf_tracks[tid]["pf"].resample()

    # Hungarian
    active = list(pf_tracks.keys())
    matched_tids = set(); assigned_c = set()
    instr_boxes = [b for b, _, _ in instr_dets]
    if active and instr_boxes:
        ests = [pf_tracks[t]["pf"].estimate() for t in active]
        C = np.zeros((len(ests), len(instr_boxes)))
        for i, pe in enumerate(ests):
            for j, cd in enumerate(instr_boxes):
                C[i,j] = np.linalg.norm(bbox_center(pe) - bbox_center(cd))
        ri, ci = linear_sum_assignment(C)
        for r, c in zip(ri, ci):
            if C[r,c] < MAX_ASSOCIATION_DIST:
                tid = active[r]
                pf_tracks[tid]["pf"].update(instr_boxes[c])
                pf_tracks[tid]["pf"].resample()
                pf_tracks[tid]["missed"] = 0
                matched_tids.add(tid); assigned_c.add(c)
    new_tids = []
    for c, cd in enumerate(instr_boxes):
        if c not in assigned_c:
            pf_tracks[next_tid] = {"pf": PFBBox(NUM_PARTICLES, cd), "missed": 0}
            new_tids.append(next_tid); next_tid += 1
    for tid in list(pf_tracks.keys()):
        if tid not in matched_tids and tid not in new_tids:
            pf_tracks[tid]["missed"] += 1
            if pf_tracks[tid]["missed"] > TRACK_RETAIN_FRAMES:
                del pf_tracks[tid]

    if frame_idx not in pos_set:
        continue

    # GT=1 frame — show detailed breakdown
    frm_dist = frame_idx - roi_last_seen_frm
    pf_fires = (roi_pf is not None and frm_dist <= ROI_PF_FALLBACK_FRAMES)
    roi_for_overlap = None
    roi_src = "none"
    if roi_det is not None:
        roi_for_overlap = roi_det; roi_src = "live"
    elif pf_fires:
        roi_for_overlap = roi_pf.estimate(); roi_src = "pf"

    print(f"Frame {frame_idx} [GT=1]")
    print(f"  roi_det     : {'box '+str([round(x,1) for x in roi_det.tolist()]) if roi_det is not None else 'NONE'}")
    print(f"  roi_last_seen: frame {roi_last_seen_frm}  (dist={frm_dist}  fallback={'YES' if pf_fires else 'NO (too stale)'})")
    if roi_pf is not None:
        est = roi_pf.estimate()
        print(f"  pf_estimate : [{est[0]:.1f}, {est[1]:.1f}, {est[2]:.1f}, {est[3]:.1f}]")
    print(f"  roi_source  : {roi_src}")
    print(f"  n_instr     : {len(instr_dets)}")
    for i, (box, cid, conf) in enumerate(instr_dets):
        overlap = ior(box, roi_for_overlap) if roi_for_overlap is not None else 0.0
        print(f"    instr[{i}] cls{cid} conf={conf:.2f} box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]  IoR={overlap:.4f}")
    if roi_for_overlap is not None:
        print(f"  roi_for_overlap: [{roi_for_overlap[0]:.0f},{roi_for_overlap[1]:.0f},{roi_for_overlap[2]:.0f},{roi_for_overlap[3]:.0f}]")
    print()

cap.release()
print("Done.")
