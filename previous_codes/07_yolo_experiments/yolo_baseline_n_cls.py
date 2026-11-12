# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/yolo_baseline.py
# Purpose       :
#   YOLOv8n-cls baseline, 30 epochs, 224px.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
from ultralytics import YOLO

def run_baseline_training():
    # 1. Initialize a pretrained YOLOv8 Nano Classification model
    # This automatically downloads 'yolov8n-cls.pt' if it's not in your directory
    model = YOLO('yolov8n-cls.pt')

    # 2. Define the absolute root path to your classification task folder
    # Ultralytics expects this folder to contain 'train' and 'val' subdirectories
    dataset_root = r"E:\Datasets\Medical\Brisc\brisc2025\classification_task"

    # Quick structural sanity check before kicking off weights allocation
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Could not locate dataset root at: {dataset_root}")

    print("Starting native YOLOv8n-cls baseline training...")
    print(f"Dataset target: {dataset_root}")

    # 3. Execute training using matching hyperparameters from your custom runs
    results = model.train(
        data=dataset_root,      # Points to the parent folder containing train/val
        epochs=30,              # Set to 30 to match your current multi-epoch timeline
        imgsz=224,              # Native target size matching your transformations
        batch=16,               # Keeps the batch size stable relative to your current runs
        workers=4,              # DataLoader subprocess workers
        device=0,               # Enforces execution on CUDA device 0
        optimizer='AdamW',      # Matches the custom script optimizer
        lr0=1e-4,               # Matching your initial learning rate
        project='BRISC_YOLO_Baseline', # Top-level save directory
        name='yolov8n_pure_cls',       # Name of this specific training run
        save=True,              # Saves model checkpoints automatically
        plots=True              # Generates confusion matrices and loss curves
    )

    print("\nBaseline training completed!")
    print(f"Results, metrics, and weights saved to: {results.save_dir}")

if __name__ == "__main__":
    run_baseline_training()