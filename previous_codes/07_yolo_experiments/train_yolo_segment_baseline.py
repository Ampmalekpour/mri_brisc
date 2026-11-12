# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/train_yolo_segment_basline.py
# Purpose       :
#   Converts masks to polygons and trains YOLOv8n-seg.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO


def convert_masks_to_yolo_polygons(base_dir):
    """
    Crawls masks, extracts contours, normalizes coordinates,
    and writes YOLO segmentation txt files.
    """
    print("Starting pixel-mask to polygon vector conversion...")

    # BRISC 2025 uses shorthand sub-tokens inside the filenames: _gl_, _me_, _pi_
    class_map = {"_gl_": 0, "_me_": 1, "_pi_": 2}

    # Clear old YOLO cache files to force a complete re-scan of the new labels
    print("Evicting old dataset caches to ensure a clean re-index...")
    for cache_path in base_dir.rglob("*.cache"):
        try:
            cache_path.unlink()
            print(f"Removed stale cache file: {cache_path.name}")
        except Exception as e:
            print(f"Could not remove {cache_path.name}: {e}")

    # Process both train and test splits
    splits = ['train', 'test']
    for split in splits:
        mask_dir = base_dir / split / 'masks'
        label_dir = base_dir / split / 'labels'
        label_dir.mkdir(parents=True, exist_ok=True)

        if not mask_dir.exists():
            print(f"Skipping split '{split}': directory not found.")
            continue

        mask_files = list(mask_dir.glob('*.png'))
        print(f"Processing {len(mask_files)} masks inside '{split}/masks'...")

        for mask_path in mask_files:
            # 1. Determine class ID using the dataset's specific abbreviation tokens
            filename_lower = mask_path.name.lower()
            class_id = None
            for class_keyword, idx in class_map.items():
                if class_keyword in filename_lower:
                    class_id = idx
                    break

            # Target text file path
            txt_path = label_dir / f"{mask_path.stem}.txt"

            # If healthy slice (_nt_) or unregistered code, leave file empty as a background image
            if class_id is None or "_nt_" in filename_lower:
                open(txt_path, 'w').close()
                continue

            # 2. Extract boundary edges from mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Find contours (external boundaries only)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lines = []
            h, w = mask.shape[:2]

            for contour in contours:
                # Simplify contour paths to prevent massive coordinate arrays
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) < 3:
                    continue  # Ignore artifacts that can't form a polygon

                # Flatten and normalize coordinate vectors
                flat_coords = approx.flatten()
                normalized_coords = []
                for i, coord in enumerate(flat_coords):
                    if i % 2 == 0:
                        normalized_coords.append(str(coord / w))  # Normalized X
                    else:
                        normalized_coords.append(str(coord / h))  # Normalized Y

                # Format: class_id x1 y1 x2 y2 x3 y3...
                line_str = f"{class_id} " + " ".join(normalized_coords)
                lines.append(line_str)

            # Save configurations to file
            with open(txt_path, 'w') as f:
                f.write("\n".join(lines))

    print("Data conversion successfully completed!\n")


def run_segmentation_baseline():
    # 1. Setup absolute paths
    dataset_root = Path(r"E:\Datasets\Medical\Brisc\brisc2025\segmentation_task")
    project_root = r"D:\projects\EyePass\AdaFace\Train_Yolo_Tumor\BRISC_YOLO_Segmentation"

    # 2. Run the internal asset converter
    convert_masks_to_yolo_polygons(dataset_root)

    # 3. Create the data.yaml tracking profile dynamically
    yaml_content = {
        'path': str(dataset_root),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('test', 'images'),  # Point val to test split for direct baseline metrics
        'names': {
            0: 'glioma',
            1: 'meningioma',
            2: 'pituitary'
        }
    }

    yaml_path = dataset_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Generated environment configuration at: {yaml_path}")

    # 4. Initialize and launch training inside safe multi-threaded environment
    print("Loading pre-trained YOLOv8n-seg weights...")
    model = YOLO("yolov8n-seg.pt")

    print("Commencing baseline segmentation run...")
    model.train(
        data=str(yaml_path),
        epochs=30,
        imgsz=640,  # Default YOLOv8 spatial envelope
        batch=32,  # Highly stable footprint for your local GPU
        workers=4,  # Windows multiprocessing guard limit
        project=project_root,
        name="yolov8n_seg_baseline",
        exist_ok=True
    )


if __name__ == '__main__':
    run_segmentation_baseline()