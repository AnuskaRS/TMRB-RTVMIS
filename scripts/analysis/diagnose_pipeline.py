import os
import glob
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = r"e:\Antigravity\TMRB\Depth calibration"
DIAG_DIR = os.path.join(BASE_DIR, "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading MiDaS on {device}...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

def diagnose_sample(trial_name, sample_idx):
    path = os.path.join(BASE_DIR, trial_name)
    idx_str = f"{sample_idx:03d}"
    
    rgb_path = os.path.join(path, f"rgb_{idx_str}.png")
    raw_path = os.path.join(path, f"depth_raw_{idx_str}.png")
    vis_path = os.path.join(path, f"depth_vis_{idx_str}.png")
    info_path = os.path.join(path, f"depth_info_{idx_str}.txt")
    
    if not os.path.exists(rgb_path):
        print(f"File not found: {rgb_path}")
        return
        
    with open(info_path, 'r') as f:
        content = f.read().strip()
        
    d_match = re.search(r"Marker Depth \(mm\):\s*(\d+)", content)
    c_match = re.search(r"Marker Center:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", content)
    
    info_depth_mm = int(d_match.group(1)) if d_match else 0
    cx = int(c_match.group(1)) if c_match else 0
    cy = int(c_match.group(2)) if c_match else 0
    
    # Load RGB image
    img_bgr = cv2.imread(rgb_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    # Run OpenCV ArUco detector on RGB
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(img_bgr)
    
    rgb_marked = img_rgb.copy()
    detected_center = None
    if ids is not None and len(ids) > 0:
        c_corners = corners[0][0]
        dc_x = int(c_corners[:, 0].mean())
        dc_y = int(c_corners[:, 1].mean())
        detected_center = (dc_x, dc_y)
        # Draw ArUco box in green
        cv2.polylines(rgb_marked, [c_corners.astype(np.int32)], True, (0, 255, 0), 4)
        cv2.circle(rgb_marked, (dc_x, dc_y), 8, (0, 255, 0), -1)
        
    # Mark info center in Red cross
    cv2.drawMarker(rgb_marked, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 25, 3)
    cv2.circle(rgb_marked, (cx, cy), 15, (255, 0, 0), 2)
    
    # Load raw OAK-D depth
    raw_depth = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    raw_val_at_c = raw_depth[cy, cx] if raw_depth is not None else 0
    
    # Patch statistics around (cx, cy)
    if raw_depth is not None:
        patch = raw_depth[max(0, cy-10):cy+11, max(0, cx-10):cx+11]
        valid_patch = patch[(patch > 0) & (patch < 30000)]
        patch_median = np.median(valid_patch) if len(valid_patch) > 0 else 0
    else:
        patch_median = 0
        
    # Run MiDaS
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False
        ).squeeze()
    midas_map = prediction.cpu().numpy()
    midas_val_at_c = midas_map[cy, cx]
    
    # Prepare visualizations
    plt.figure(figsize=(18, 6))
    
    # Panel 1: RGB Image with Marked Center
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_marked)
    plt.plot(cx, cy, 'rx', markersize=12, markeredgewidth=3, label=f"Info Center ({cx},{cy})")
    if detected_center:
        plt.plot(detected_center[0], detected_center[1], 'go', markersize=8, label=f"ArUco Detected ({detected_center[0]},{detected_center[1]})")
    plt.title(f"{trial_name} Frame {idx_str}: RGB Image\nOAK-D Info Depth = {info_depth_mm} mm", fontsize=11, fontweight='bold')
    plt.legend(loc="lower right")
    plt.axis("off")
    
    # Panel 2: MiDaS Depth Map
    plt.subplot(1, 3, 2)
    plt.imshow(midas_map, cmap="inferno")
    plt.colorbar(label="MiDaS Relative Disparity")
    plt.plot(cx, cy, 'w+', markersize=14, markeredgewidth=3)
    plt.circle = plt.Circle((cx, cy), 15, color='w', fill=False, linewidth=2)
    plt.gca().add_patch(plt.circle)
    plt.title(f"MiDaS Depth Map\nVal at ({cx},{cy}) = {midas_val_at_c:.1f}", fontsize=11, fontweight='bold')
    plt.axis("off")
    
    # Panel 3: OAK-D Depth Map
    plt.subplot(1, 3, 3)
    oak_vis = raw_depth.copy().astype(np.float32)
    oak_vis[oak_vis == 0] = np.nan
    oak_vis[oak_vis > 30000] = np.nan
    plt.imshow(oak_vis, cmap="viridis")
    plt.colorbar(label="OAK-D Depth (mm)")
    plt.plot(cx, cy, 'r+', markersize=14, markeredgewidth=3)
    plt.title(f"OAK-D Raw Depth Map\nVal at ({cx},{cy}) = {raw_val_at_c} mm | Patch Med = {patch_median:.0f} mm", fontsize=11, fontweight='bold')
    plt.axis("off")
    
    plt.tight_layout()
    out_fn = os.path.join(DIAG_DIR, f"{trial_name}_frame_{idx_str}_diagnostic.png")
    plt.savefig(out_fn, dpi=200)
    plt.close()
    
    print(f"\n================ {trial_name} Frame {idx_str} ================")
    print(f"Info Center (cx, cy)  : ({cx}, {cy})")
    print(f"Info Marker Depth     : {info_depth_mm} mm")
    print(f"OAK-D Raw [cy, cx]    : {raw_val_at_c} mm")
    print(f"OAK-D 21x21 Patch Med : {patch_median:.1f} mm")
    print(f"MiDaS Disparity at C  : {midas_val_at_c:.2f}")
    if detected_center:
        print(f"Detected ArUco Center : ({detected_center[0]}, {detected_center[1]})")
        print(f"Center Shift (pixels) : dx={detected_center[0]-cx}, dy={detected_center[1]-cy}")
    else:
        print(f"Detected ArUco Center : None (Auto-detect failed without CLAHE/params)")
    print(f"Saved diagnostic figure to {out_fn}")

# Diagnose key frames (including outliers identified by user: trial1 frame 4, 9 and trial2 frame 9)
diagnose_sample("trial1", 1)
diagnose_sample("trial1", 4)
diagnose_sample("trial1", 5)
diagnose_sample("trial1", 9)
diagnose_sample("trial2", 3)
diagnose_sample("trial2", 8)
diagnose_sample("trial2", 9)
