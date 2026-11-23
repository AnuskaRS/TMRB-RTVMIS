import os
import shutil
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from ultralytics import YOLO

def setup_no_roi_dataset(src_dir, dest_dir):
    print("Setting up dataset without ROI...")
    src_images = os.path.join(src_dir, "images")
    src_labels = os.path.join(src_dir, "labels")
    
    dest_images = os.path.join(dest_dir, "images")
    dest_labels = os.path.join(dest_dir, "labels")
    
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)
    
    image_files = os.listdir(src_images)
    for img_file in image_files:
        src_img_path = os.path.join(src_images, img_file)
        dest_img_path = os.path.join(dest_images, img_file)
        if not os.path.exists(dest_img_path):
            try:
                os.link(src_img_path, dest_img_path)
            except OSError:
                shutil.copy2(src_img_path, dest_img_path)
                
    label_files = os.listdir(src_labels)
    for label_file in label_files:
        src_label_path = os.path.join(src_labels, label_file)
        dest_label_path = os.path.join(dest_labels, label_file)
        
        with open(src_label_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id == 0: # Skip ROI class
                continue
            new_class_id = class_id - 1
            new_lines.append(f"{new_class_id} " + " ".join(parts[1:]) + "\n")
            
        with open(dest_label_path, 'w') as f:
            f.writelines(new_lines)
    print("Dataset setup complete.")

def main():
    base_dir = r"e:\Antigravity\TMRB\cross-validation"
    src_dir = os.path.join(base_dir, "chole_data")
    dest_dir = os.path.join(base_dir, "chole_data_no_roi")
    
    # 1. Setup dataset (remove ROI)
    setup_no_roi_dataset(src_dir, dest_dir)
    
    image_paths = sorted(glob.glob(os.path.join(dest_dir, "images", "*.jpg")))
    
    # 2. Configure Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    class_names = [
        "AtraumaticGrasper_Head", "AtraumaticGrasper_Body", "Electrichook",
        "CurvedAtraumaticGrasper_Head", "CurvedAtraumaticGrasper_Body",
        "Suction-Irrigation", "ClipApplier(Metal)_Head", "ClipApplier(Metal)_Body",
        "Scissors_Head", "Scissors_Body", "Overholt_Head", "Overholt_Body",
        "Ligasure_Head", "Ligasure_Body", "ClipApplier(Hem-O-Lok)_Head",
        "ClipApplier(Hem-O-Lok)_Body", "Specimanbag"
    ]
    
    all_metrics = []
    confusion_matrices = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths), 1):
        print(f"\n--- Starting Fold {fold} ---")
        train_images = [image_paths[i] for i in train_idx]
        val_images = [image_paths[i] for i in val_idx]
        
        fold_dir = os.path.join(base_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_txt = os.path.join(fold_dir, "train.txt")
        val_txt = os.path.join(fold_dir, "val.txt")
        
        with open(train_txt, 'w') as f:
            f.writelines(f"{img}\n" for img in train_images)
        with open(val_txt, 'w') as f:
            f.writelines(f"{img}\n" for img in val_images)
            
        dataset_yaml_path = os.path.join(fold_dir, "dataset.yaml")
        yaml_content = {
            'train': train_txt,
            'val': val_txt,
            'nc': len(class_names),
            'names': class_names
        }
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
            
        # 3. Train YOLOv10n
        model = YOLO("yolov10n.pt")
        model.train(
            data=dataset_yaml_path,
            epochs=50,
            batch=8,
            imgsz=960,
            lr0=0.001,
            optimizer="AdamW",
            patience=5,
            amp=True,
            project=base_dir,
            name=f"fold_{fold}_run",
            exist_ok=True,
            # Disable Data Augmentation completely
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
            scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
            bgr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0
        )
        
        # 4. Evaluate and Extract Metrics
        best_model = YOLO(os.path.join(base_dir, f"fold_{fold}_run", "weights", "best.pt"))
        metrics = best_model.val(data=dataset_yaml_path, batch=8, imgsz=960)
        
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        
        all_metrics.append({
            'Fold': fold,
            'mAP@50': map50,
            'mAP@50-95': map50_95
        })
        print(f"Fold {fold} - mAP@50: {map50:.4f}, mAP@50-95: {map50_95:.4f}")
        
        if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
            confusion_matrices.append(metrics.confusion_matrix.matrix)
            
    # 5. Generate Reports and Plots
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(base_dir, "all_folds_metrics.csv"), index=False)
    
    mean_map50 = metrics_df['mAP@50'].mean()
    sd_map50 = metrics_df['mAP@50'].std()
    mean_map50_95 = metrics_df['mAP@50-95'].mean()
    sd_map50_95 = metrics_df['mAP@50-95'].std()
    
    summary_df = pd.DataFrame([
        {'Metric': 'mAP@50', 'Mean': mean_map50, 'SD': sd_map50, 'Mean ± SD': f"{mean_map50:.4f} ± {sd_map50:.4f}"},
        {'Metric': 'mAP@50-95', 'Mean': mean_map50_95, 'SD': sd_map50_95, 'Mean ± SD': f"{mean_map50_95:.4f} ± {sd_map50_95:.4f}"}
    ])
    summary_df.to_csv(os.path.join(base_dir, "summary_statistics.csv"), index=False)
    
    # Boxplots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=metrics_df['mAP@50'], color='skyblue')
    sns.stripplot(y=metrics_df['mAP@50'], color='black', alpha=0.5)
    plt.title("mAP@50 across Folds")
    plt.ylabel("mAP@50")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=metrics_df['mAP@50-95'], color='lightgreen')
    sns.stripplot(y=metrics_df['mAP@50-95'], color='black', alpha=0.5)
    plt.title("mAP@50-95 across Folds")
    plt.ylabel("mAP@50-95")
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "boxplots.png"), dpi=300)
    plt.close()
    
    # Training Curves
    plt.figure(figsize=(10, 6))
    for fold in range(1, 6):
        results_csv = os.path.join(base_dir, f"fold_{fold}_run", "results.csv")
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            if 'metrics/mAP50(B)' in df.columns:
                plt.plot(df['epoch'], df['metrics/mAP50(B)'], label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50")
    plt.title("Training Curves (mAP@50)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(base_dir, "training_curves_mAP50.png"), dpi=300)
    plt.close()

    # Mean Confusion Matrix
    if len(confusion_matrices) == 5:
        mean_cm = np.mean(confusion_matrices, axis=0)
        # Normalize it for better visualization (optional, usually matrix is raw counts)
        # We will plot the raw counts to be consistent with YOLO defaults
        plt.figure(figsize=(14, 12))
        sns.heatmap(mean_cm, annot=False, cmap='Blues', fmt='.2f')
        if mean_cm.shape[0] >= len(class_names):
            tick_labels = class_names + ["Background"][:mean_cm.shape[0]-len(class_names)]
            plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=90)
            plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=0)
        
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Mean Confusion Matrix (Across 5 Folds)")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "mean_confusion_matrix.png"), dpi=300)
        plt.close()
    
    print("\n--- Pipeline Completed Successfully ---")
    print(summary_df)

if __name__ == "__main__":
    main()
