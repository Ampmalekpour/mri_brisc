# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/yolo_baseline_2.py
# Purpose       :
#   YOLOv8m-cls, 50 epochs, 512px.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
from ultralytics import YOLO


def run_advanced_training():
    # 1. Initialize a higher-capacity architecture
    # yolov8m-cls scales parameters up to ~15.8M, giving it deep spatial awareness
    print("Initializing YOLOv8-Medium Classification Model...")
    model = YOLO('yolov8m-cls.pt')

    # 2. Kick off the advanced training run
    # Point directly to your local dataset path on the E: drive
    model.train(
        data=r"E:\Datasets\Medical\Brisc\brisc2025\classification_task",
        epochs=50,  # Slightly more runway for a larger model to converge
        imgsz=512,  # Massive resolution increase to preserve tumor boundaries
        batch=32,  # Perfect fit for 512px on an 8GB RTX 4060
        workers=4,  # Safe worker count for Windows stability
        optimizer='AdamW',  # Keeps convergence smooth and stable
        lr0=0.0001,  # Maintain your stable baseline learning rate

        # --- Context & Boundary Augmentations ---
        hsv_v=0.4,  # High value/brightness variation (breaks intensity dependency)
        scale=0.2,  # Slight scaling shifts to teach global size context
        fliplr=0.5,  # Left-right reflections for anatomical variance
        flipud=0.5,  # Up-down reflections to maximize orientation robustness

        # --- Environment Configurations ---
        project="D:/projects/EyePass/AdaFace/Train_Yolo_Tumor/BRISC_YOLO_Advanced",
        name="yolov8m_boundary_focused",
        exist_ok=True
    )


if __name__ == '__main__':
    run_advanced_training()