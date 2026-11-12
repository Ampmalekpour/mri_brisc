# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/test_yolo_baseline.py
# Purpose       :
#   Per-class accuracy / confusions for a YOLO classifier.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


def run_detailed_inference():
    # 1. Define local paths
    model_path = r"D:\Image\EyePlate\Train_Yolo_Tumor\BRISC_YOLO_Baseline\yolov8n_pure_cls\weights\best.pt"
    test_dataset_dir = Path(r"D:\Datasets\Medical\brisc2025\classification_task\test")

    # 2. Load model & mapping
    print("Loading model weights...")
    model = YOLO(model_path)
    class_names = model.names
    print(f"Model Class Mapping: {class_names}\n")

    # 3. Initialize metrics structures
    class_totals = defaultdict(int)
    class_correct = defaultdict(int)
    # Nested dictionary tracking: [True Class][Wrong Predicted Class] = Count
    class_mistakes = defaultdict(lambda: defaultdict(int))

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    total_processed = 0
    overall_correct = 0

    print("Running inference across test split splits...")

    # 4. Process directories
    for class_folder in test_dataset_dir.iterdir():
        if class_folder.is_dir():
            ground_truth_label = class_folder.name

            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in valid_extensions:

                    # Core prediction
                    outputs = model(str(img_path), verbose=False)
                    pred_index = outputs[0].probs.top1
                    predicted_label = class_names[pred_index]

                    # Normalize names for comparison metrics
                    gt_lower = ground_truth_label.lower()
                    pred_lower = predicted_label.lower()

                    # Update counters
                    class_totals[ground_truth_label] += 1
                    total_processed += 1

                    if gt_lower == pred_lower:
                        class_correct[ground_truth_label] += 1
                        overall_correct += 1
                    else:
                        # Log specific misclassification error
                        class_mistakes[ground_truth_label][predicted_label] += 1

    # 5. Render Per-Class Analytics Report
    print("\n" + "=" * 50)
    print("      PER-CLASS PERFORMANCE & ACCURACY REPORT      ")
    print("=" * 50)

    for real_class in sorted(class_totals.keys()):
        total = class_totals[real_class]
        correct = class_correct[real_class]
        accuracy = (correct / total) * 100 if total > 0 else 0

        print(f"\n📁 Class: {real_class.upper()}")
        print(f"   ↳ Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

        # Pull and rank mistakes for this class
        mistakes = class_mistakes[real_class]
        if mistakes:
            print("   ↳ Most Common Misclassifications:")
            # Sort mistakes by count descending
            sorted_mistakes = sorted(mistakes.items(), key=lambda item: item[1], reverse=True)
            for wrong_pred, count in sorted_mistakes:
                print(f"      ▪ Confused as '{wrong_pred}': {count} times")
        else:
            print("   ↳ Most Common Misclassifications: None 🎉 (Perfect Classification!)")

    # 6. Global Summary
    print("\n" + "=" * 50)
    print("                GLOBAL SUMMARY                 ")
    print("=" * 50)
    global_accuracy = (overall_correct / total_processed) * 100 if total_processed > 0 else 0
    print(f"Total Test Images Run : {total_processed}")
    print(f"Total Correct Passes  : {overall_correct}")
    print(f"Overall Test Accuracy : {global_accuracy:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    run_detailed_inference()