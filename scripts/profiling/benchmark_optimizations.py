import cv2, time, torch, numpy as np, sys
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = YOLO("E:/Antigravity/TMRB/best1.pt")
dm = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False).eval().to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
tf = midas_transforms.small_transform

cap = cv2.VideoCapture("E:/Antigravity/TMRB/video_1.mp4")
frames = []
for _ in range(60):
    ret, f = cap.read()
    if ret:
        frames.append(f)
cap.release()

print(f"Loaded {len(frames)} test frames.")

# Warmup
for f in frames[:5]:
    _ = model(f, verbose=False)
    inp = tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).to(device)
    if inp.dim() == 3: inp = inp.unsqueeze(0)
    with torch.no_grad():
        _ = dm(inp)

# Mode 1: Tracking-only (YOLOv10 + Particle Filter)
t0 = time.perf_counter()
for f in frames:
    _ = model(f, verbose=False)
t1 = time.perf_counter()
ms_track = (t1 - t0) * 1000 / len(frames)
fps_track = 1000 / ms_track
print(f"\n1) Tracking-only (YOLOv10 + PF)         : {ms_track:.1f} ms/frame  -->  {fps_track:.1f} FPS")

# Mode 2: Sequential FP32 Full Pipeline (Every Frame MiDaS)
t0 = time.perf_counter()
for f in frames:
    _ = model(f, verbose=False)
    inp = tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).to(device)
    if inp.dim() == 3: inp = inp.unsqueeze(0)
    with torch.no_grad():
        _ = dm(inp)
t1 = time.perf_counter()
ms_full = (t1 - t0) * 1000 / len(frames)
fps_full = 1000 / ms_full
print(f"2) Full Pipeline (Sequential, Every Frame): {ms_full:.1f} ms/frame  -->  {fps_full:.1f} FPS")

# Mode 3: Interleaved Depth (Run MiDaS every 3rd frame, update depth at 10 Hz)
t0 = time.perf_counter()
last_d = None
for i, f in enumerate(frames):
    _ = model(f, verbose=False)
    if i % 3 == 0 or last_d is None:
        inp = tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).to(device)
        if inp.dim() == 3: inp = inp.unsqueeze(0)
        with torch.no_grad():
            last_d = dm(inp)
t1 = time.perf_counter()
ms_inter3 = (t1 - t0) * 1000 / len(frames)
fps_inter3 = 1000 / ms_inter3
print(f"3) Interleaved Depth (Every 3rd frame)   : {ms_inter3:.1f} ms/frame  -->  {fps_inter3:.1f} FPS")

# Mode 4: Interleaved Depth (Run MiDaS every 4th frame)
t0 = time.perf_counter()
last_d = None
for i, f in enumerate(frames):
    _ = model(f, verbose=False)
    if i % 4 == 0 or last_d is None:
        inp = tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).to(device)
        if inp.dim() == 3: inp = inp.unsqueeze(0)
        with torch.no_grad():
            last_d = dm(inp)
t1 = time.perf_counter()
ms_inter4 = (t1 - t0) * 1000 / len(frames)
fps_inter4 = 1000 / ms_inter4
print(f"4) Interleaved Depth (Every 4th frame)   : {ms_inter4:.1f} ms/frame  -->  {fps_inter4:.1f} FPS")
