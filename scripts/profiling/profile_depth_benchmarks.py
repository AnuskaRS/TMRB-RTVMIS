import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Profiling Depth Models on Device: {device}")

models_to_test = [
    ("MiDaS_small", "intel-isl/MiDaS", "MiDaS_small"),
    ("DPT_Hybrid",  "intel-isl/MiDaS", "DPT_Hybrid"),
]

results = []

for name, repo, model_type in models_to_test:
    try:
        print(f"\nLoading {name} ...", flush=True)
        model = torch.hub.load(repo, model_type, verbose=False).eval().to(device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        dummy_in = torch.randn(1, 3, 384, 384).to(device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_in)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Time 50 forward passes
        t0 = time.perf_counter()
        for _ in range(50):
            with torch.no_grad():
                _ = model(dummy_in)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        lat_ms = (t1 - t0) * 1000.0 / 50.0
        fps = 1000.0 / lat_ms
        print(f"  {name}: Params = {params:.2f}M | Latency = {lat_ms:.2f} ms | Pure Model Throughput = {fps:.1f} FPS")
        results.append({"name": name, "params_M": round(params, 2), "lat_ms": round(lat_ms, 2)})
    except Exception as e:
        print(f"  {name} load error: {e}")

print("\nProfiling Complete.")
