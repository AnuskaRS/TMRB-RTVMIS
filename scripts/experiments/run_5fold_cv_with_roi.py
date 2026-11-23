import os
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from ultralytics import YOLO

def main():
    base_dir = r"e:\Antigravity\TMRB\cross-validation"
    src_dir = os.path.join(base_dir, "chole_data")
    output_dir = os.path.join(base_dir, "with_roi_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(src_dir, "images", "*.jpg")))
    
    # Configure Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    class_names = [
        "ROI",
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
        
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
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
            project=output_dir,
            name=f"fold_{fold}_run",
            exist_ok=True,
            # Disable Data Augmentation completely
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
            scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
            bgr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0
        )
        
        # 4. Evaluate and Extract Metrics
        best_model = YOLO(os.path.join(output_dir, f"fold_{fold}_run", "weights", "best.pt"))
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
    metrics_df.to_csv(os.path.join(output_dir, "all_folds_metrics_with_roi.csv"), index=False)
    
    mean_map50 = metrics_df['mAP@50'].mean()
    sd_map50 = metrics_df['mAP@50'].std()
    mean_map50_95 = metrics_df['mAP@50-95'].mean()
    sd_map50_95 = metrics_df['mAP@50-95'].std()
    
    summary_df = pd.DataFrame([
        {'Metric': 'mAP@50', 'Mean': mean_map50, 'SD': sd_map50, 'Mean ± SD': f"{mean_map50:.4f} ± {sd_map50:.4f}"},
        {'Metric': 'mAP@50-95', 'Mean': mean_map50_95, 'SD': sd_map50_95, 'Mean ± SD': f"{mean_map50_95:.4f} ± {sd_map50_95:.4f}"}
    ])
    summary_df.to_csv(os.path.join(output_dir, "summary_statistics_with_roi.csv"), index=False)
    
    # Boxplots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=metrics_df['mAP@50'], color='skyblue')
    sns.stripplot(y=metrics_df['mAP@50'], color='black', alpha=0.5)
    plt.title("mAP@50 across Folds (With ROI)")
    plt.ylabel("mAP@50")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=metrics_df['mAP@50-95'], color='lightgreen')
    sns.stripplot(y=metrics_df['mAP@50-95'], color='black', alpha=0.5)
    plt.title("mAP@50-95 across Folds (With ROI)")
    plt.ylabel("mAP@50-95")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplots_with_roi.png"), dpi=300)
    plt.close()
    
    # Training Curves
    plt.figure(figsize=(10, 6))
    for fold in range(1, 6):
        results_csv = os.path.join(output_dir, f"fold_{fold}_run", "results.csv")
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            if 'metrics/mAP50(B)' in df.columns:
                plt.plot(df['epoch'], df['metrics/mAP50(B)'], label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50")
    plt.title("Training Curves (mAP@50 - With ROI)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "training_curves_mAP50_with_roi.png"), dpi=300)
    plt.close()

    # Mean Confusion Matrix
    if len(confusion_matrices) == 5:
        mean_cm = np.mean(confusion_matrices, axis=0)
        plt.figure(figsize=(14, 12))
        sns.heatmap(mean_cm, annot=False, cmap='Blues', fmt='.2f')
        if mean_cm.shape[0] >= len(class_names):
            tick_labels = class_names + ["Background"][:mean_cm.shape[0]-len(class_names)]
            plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=90)
            plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=0)
        
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Mean Confusion Matrix (With ROI)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mean_confusion_matrix_with_roi.png"), dpi=300)
        plt.close()
    
    print("\n--- Pipeline Completed Successfully ---")
    print(summary_df)

if __name__ == "__main__":
    main()
