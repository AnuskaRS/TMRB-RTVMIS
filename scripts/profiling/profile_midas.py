import time, torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False).eval().to(device)
    params = sum(p.numel() for p in midas.parameters()) / 1e6
    inp = torch.randn(1, 3, 384, 384).to(device)
    for _ in range(5): _ = midas(inp)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            _ = midas(inp)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()
    lat = (t1 - t0) * 1000.0 / 50.0
    print(f"MiDaS_small: {params:.2f}M params, {lat:.2f} ms per forward pass")
except Exception as e:
    print("Error:", e)
