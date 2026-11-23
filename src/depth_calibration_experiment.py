import os
import glob
import re
import json
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Base directory
BASE_DIR = r"e:\Antigravity\TMRB\Depth calibration"
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select computing device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load MiDaS_small model
print("Loading MiDaS_small model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

def fit_scale_and_shift(x_pred_raw, y_gt):
    """
    Affine alignment: Z_pred = scale * (1 / (x_pred_raw + eps)) + shift
    or Z_pred = s * x_inv + t via linear least squares.
    """
    x_inv = 1.0 / (x_pred_raw + 1e-6)
    A = np.vstack([x_inv, np.ones(len(x_inv))]).T
    scale, shift = np.linalg.lstsq(A, y_gt, rcond=None)[0]
    pred_aligned = scale * x_inv + shift
    return pred_aligned, scale, shift

def compute_metrics(y_true, y_pred):
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mre = np.mean(abs_errors / y_true) * 100.0  # Percentage
    
    # R2 Score
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(errors ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "MAE_mm": float(mae),
        "RMSE_mm": float(rmse),
        "MAE_cm": float(mae / 10.0),
        "RMSE_cm": float(rmse / 10.0),
        "MRE_pct": float(mre),
        "R2": float(r2),
        "Max_Error_mm": float(np.max(abs_errors)),
        "Std_Error_mm": float(np.std(errors))
    }

def process_trial(trial_name):
    path = os.path.join(BASE_DIR, trial_name)
    rgb_files = sorted(glob.glob(os.path.join(path, "rgb_*.png")))
    
    records = []
    
    for rgb_path in rgb_files:
        idx_str = re.search(r"\d+", os.path.basename(rgb_path)).group(0)
        idx_num = int(idx_str)
        info_path = os.path.join(path, f"depth_info_{idx_str}.txt")
        raw_path = os.path.join(path, f"depth_raw_{idx_str}.png")
        vis_path = os.path.join(path, f"depth_vis_{idx_str}.png")
        
        # Read info file
        with open(info_path, 'r') as f:
            content = f.read()
            
        d_match = re.search(r"Marker Depth \(mm\):\s*(\d+)", content)
        c_match = re.search(r"Marker Center:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", content)
        
        gt_depth_mm = float(d_match.group(1)) if d_match else 0.0
        cx = int(c_match.group(1)) if c_match else 0
        cy = int(c_match.group(2)) if c_match else 0
        
        # Read raw depth image (uint16 in mm)
        raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        val_at_center = float(raw_img[cy, cx]) if raw_img is not None else 0.0
        
        # Calculate robust median in 21x21 neighborhood
        if raw_img is not None:
            patch = raw_img[max(0, cy-10):min(raw_img.shape[0], cy+11), max(0, cx-10):min(raw_img.shape[1], cx+11)]
            valid = patch[(patch > 0) & (patch < 30000)]
            oak_patch_median = float(np.median(valid)) if len(valid) > 0 else val_at_center
        else:
            oak_patch_median = val_at_center
            
        # OAK-D reference depth choice: use Marker Depth recorded by OAK-D software
        oak_ref_depth = gt_depth_mm
        
        # Load RGB for MiDaS
        img_bgr = cv2.imread(rgb_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
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
        
        # Sample MiDaS relative disparity at marker center (median of 5x5 patch)
        m_patch = midas_map[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)]
        midas_disp_val = float(np.median(m_patch))
        
        records.append({
            "Trial": trial_name,
            "Position": idx_num,
            "Filename": os.path.basename(rgb_path),
            "CenterX": cx,
            "CenterY": cy,
            "OAK_Info_Depth_mm": gt_depth_mm,
            "OAK_Raw_Center_mm": val_at_center,
            "OAK_Raw_Patch_Median_mm": oak_patch_median,
            "MiDaS_Relative_Disparity": midas_disp_val,
            "MiDaS_Inv_Disparity": 1.0 / (midas_disp_val + 1e-6)
        })
        
    return records

print("Processing Trial 1...")
rec1 = process_trial("trial1")
print("Processing Trial 2...")
rec2 = process_trial("trial2")

all_records = rec1 + rec2
df_all = pd.DataFrame(all_records)

# Fit alignment models for Trial 1, Trial 2, and Combined
def analyze_dataframe(df_sub, label):
    gt = df_sub["OAK_Info_Depth_mm"].values
    disp = df_sub["MiDaS_Relative_Disparity"].values
    
    pred_aligned, scale, shift = fit_scale_and_shift(disp, gt)
    df_sub["MiDaS_Predicted_Depth_mm"] = pred_aligned
    df_sub["Absolute_Error_mm"] = np.abs(pred_aligned - gt)
    
    metrics = compute_metrics(gt, pred_aligned)
    metrics["Scale"] = float(scale)
    metrics["Shift"] = float(shift)
    metrics["Sample_Count"] = len(gt)
    
    print(f"\n================ {label} (N={len(gt)}) ================")
    print(f"Scale: {scale:.4f}, Shift: {shift:.4f}")
    print(f"MAE:  {metrics['MAE_mm']:.2f} mm ({metrics['MAE_cm']:.2f} cm)")
    print(f"RMSE: {metrics['RMSE_mm']:.2f} mm ({metrics['RMSE_cm']:.2f} cm)")
    print(f"MRE:  {metrics['MRE_pct']:.2f} %")
    print(f"R²:   {metrics['R2']:.4f}")
    print(f"Max Error: {metrics['Max_Error_mm']:.2f} mm")
    
    return df_sub, metrics

df_t1 = df_all[df_all["Trial"] == "trial1"].copy()
df_t2 = df_all[df_all["Trial"] == "trial2"].copy()

df_t1_res, metrics_t1 = analyze_dataframe(df_t1, "Trial 1 (Position 1-9)")
df_t2_res, metrics_t2 = analyze_dataframe(df_t2, "Trial 2 (Position 1-12)")
df_all_res, metrics_all = analyze_dataframe(df_all.copy(), "Combined Dataset (Trial 1 + Trial 2)")

# Combine into detailed final DataFrame
df_final = pd.concat([df_t1_res, df_t2_res], ignore_index=True)
csv_path = os.path.join(OUTPUT_DIR, "depth_calibration_comparison_results.csv")
df_final.to_csv(csv_path, index=False)
print(f"\nSaved detailed comparison CSV to {csv_path}")

# Save summary metrics JSON
summary_metrics = {
    "Trial_1": metrics_t1,
    "Trial_2": metrics_t2,
    "Combined": metrics_all
}

json_path = os.path.join(OUTPUT_DIR, "depth_calibration_summary_metrics.json")
with open(json_path, 'w') as f:
    json.dump(summary_metrics, f, indent=4)
print(f"Saved summary metrics JSON to {json_path}")

# Generate high quality plots
plt.figure(figsize=(14, 5))

# Plot 1: OAK-D vs MiDaS Depth per Position
plt.subplot(1, 2, 1)
x1 = df_t1_res["Position"]
plt.plot(x1, df_t1_res["OAK_Info_Depth_mm"], 'o-', color='#1f77b4', linewidth=2, label="OAK-D Depth (Trial 1)")
plt.plot(x1, df_t1_res["MiDaS_Predicted_Depth_mm"], 's--', color='#ff7f0e', linewidth=2, label="MiDaS Depth (Trial 1)")

x2 = df_t2_res["Position"]
plt.plot(x2, df_t2_res["OAK_Info_Depth_mm"], '^--', color='#2ca02c', linewidth=2, label="OAK-D Depth (Trial 2)")
plt.plot(x2, df_t2_res["MiDaS_Predicted_Depth_mm"], 'd:', color='#d62728', linewidth=2, label="MiDaS Depth (Trial 2)")

plt.xlabel("Phantom Position / Frame Index", fontsize=11, fontweight='bold')
plt.ylabel("Depth (mm)", fontsize=11, fontweight='bold')
plt.title("OAK-D Depth vs MiDaS Estimated Depth", fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=9)

# Plot 2: Correlation Scatter Plot
plt.subplot(1, 2, 2)
plt.scatter(df_t1_res["OAK_Info_Depth_mm"], df_t1_res["MiDaS_Predicted_Depth_mm"], color='blue', s=60, alpha=0.8, label=f"Trial 1 (R² = {metrics_t1['R2']:.3f})")
plt.scatter(df_t2_res["OAK_Info_Depth_mm"], df_t2_res["MiDaS_Predicted_Depth_mm"], color='green', s=60, alpha=0.8, label=f"Trial 2 (R² = {metrics_t2['R2']:.3f})")

# Ideal diagonal
all_gt = df_all_res["OAK_Info_Depth_mm"].values
min_val, max_val = min(all_gt) - 100, max(all_gt) + 100
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Ideal Identity (1:1)")

plt.xlabel("OAK-D Ground Truth Depth (mm)", fontsize=11, fontweight='bold')
plt.ylabel("MiDaS Calibrated Depth (mm)", fontsize=11, fontweight='bold')
plt.title("Correlation: OAK-D vs MiDaS", fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=9)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "oak_vs_midas_depth_comparison.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Saved comparison plot to {plot_path}")

print("Experiment script completed successfully!")
