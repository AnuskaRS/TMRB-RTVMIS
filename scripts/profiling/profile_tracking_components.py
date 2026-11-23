import cv2
import time
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Profiling on device: {device}")

model = YOLO("E:/Antigravity/TMRB/best1.pt")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False).eval().to(device)
transform = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False).small_transform

cap = cv2.VideoCapture("E:/Antigravity/TMRB/video_1.mp4")
frames = []
for _ in range(100):
    ret, f = cap.read()
    if ret:
        frames.append(f)
cap.release()

# Particle Filter class
NUM_PARTICLES = 100
PF_SIGMA = 10.0

class ParticleFilterBBox:
    def __init__(self, n, init_bbox):
        self.n = n
        self.pts = np.random.randn(n, 4) + np.array(init_bbox)
        self.w = np.ones(n) / n

    def predict(self, noise=2.0):
        self.pts += np.random.randn(self.n, 4) * noise

    def update(self, det):
        d = np.linalg.norm(self.pts - np.array(det), axis=1)
        self.w = np.exp(-d / PF_SIGMA)
        s = self.w.sum()
        self.w = self.w / s if s > 0 else np.ones(self.n) / self.n

    def resample(self):
        if 1.0 / (self.w ** 2).sum() < self.n / 2:
            idx = np.random.choice(self.n, self.n, p=self.w)
            self.pts = self.pts[idx]
            self.w = np.ones(self.n) / self.n

    def estimate(self):
        return np.average(self.pts, weights=self.w, axis=0)

# Warmup
for f in frames[:5]:
    _ = model(f, verbose=False)
    inp = transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).to(device)
    if inp.dim() == 3: inp = inp.unsqueeze(0)
    with torch.no_grad():
        _ = midas(inp)

# Detailed per-module profiling
t_yolo = 0.0
t_pf = 0.0
t_midas = 0.0

pf_tracks = {}
next_tid = 1

for f in frames:
    frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    
    # 1. YOLO Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(f, verbose=False)
    t1 = time.perf_counter()
    t_yolo += (t1 - t0)
    
    boxes = preds[0].boxes
    instr_dets = []
    if boxes is not None and len(boxes):
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        for cid, box in zip(cls_ids, xyxy):
            if cid != 0:
                instr_dets.append(box)
                
    # 2. Particle Filtering & Hungarian Association
    t0 = time.perf_counter()
    for tid in list(pf_tracks.keys()):
        pf_tracks[tid].predict()
        pf_tracks[tid].resample()
        
    active_tids = list(pf_tracks.keys())
    matched_tids = set()
    assigned_c = set()
    if active_tids and instr_dets:
        ests = [pf_tracks[t].estimate() for t in active_tids]
        C = np.zeros((len(ests), len(instr_dets)))
        for i, pe in enumerate(ests):
            pe_c = np.array([(pe[0]+pe[2])/2, (pe[1]+pe[3])/2])
            for j, cd in enumerate(instr_dets):
                cd_c = np.array([(cd[0]+cd[2])/2, (cd[1]+cd[3])/2])
                C[i, j] = np.linalg.norm(pe_c - cd_c)
        ri, ci = linear_sum_assignment(C)
        for r, c in zip(ri, ci):
            if C[r, c] < 150.0:
                tid = active_tids[r]
                pf_tracks[tid].update(instr_dets[c])
                pf_tracks[tid].resample()
                matched_tids.add(tid)
                assigned_c.add(c)
                
    for c, cd in enumerate(instr_dets):
        if c not in assigned_c:
            pf_tracks[next_tid] = ParticleFilterBBox(NUM_PARTICLES, cd)
            next_tid += 1
    t1 = time.perf_counter()
    t_pf += (t1 - t0)
    
    # 3. MiDaS Monocular Depth Estimation
    t0 = time.perf_counter()
    inp = transform(frame_rgb).to(device)
    if inp.dim() == 3: inp = inp.unsqueeze(0)
    with torch.no_grad():
        _ = midas(inp)
    t1 = time.perf_counter()
    t_midas += (t1 - t0)

n = len(frames)
ms_yolo = (t_yolo * 1000) / n
ms_pf = (t_pf * 1000) / n
ms_midas = (t_midas * 1000) / n
ms_total = ms_yolo + ms_pf + ms_midas

print("\n" + "="*60)
print("  PER-MODULE EXECUTION TIME BREAKDOWN (100 frames)")
print("="*60)
print(f"  YOLOv10 Object Detection     : {ms_yolo:6.2f} ms ({ms_yolo/ms_total*100:5.1f}%)")
print(f"  Particle Filter Tracking (N=100): {ms_pf:6.2f} ms ({ms_pf/ms_total*100:5.1f}%)")
print(f"  MiDaS Monocular Depth        : {ms_midas:6.2f} ms ({ms_midas/ms_total*100:5.1f}%)")
print("  " + "─"*56)
print(f"  Total Sequential Pipeline    : {ms_total:6.2f} ms ({1000/ms_total:5.1f} FPS)")
print("="*60)
