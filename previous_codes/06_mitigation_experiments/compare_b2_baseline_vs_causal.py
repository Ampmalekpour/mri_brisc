# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : EfficientNets/test_b2_compare_normal_causal.py
# Purpose       :
#   Baseline vs causally-trained B2: clean / zero-out accuracy, drop, accuracy-based per-class SRS.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress warnings for clean terminal output
warnings.filterwarnings("ignore")


def evaluate_model(model, dataloader, device, class_names):
    """Runs inference and returns overall accuracy and class-wise accuracy."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    # Extract class-wise recall (accuracy per class)
    class_acc = {cls: report[cls]['recall'] for cls in class_names}
    return acc, class_acc


def main():
    print("\n" + "=" * 80)
    print(" === INITIALIZING COMPARATIVE CAUSAL AUDIT SUITE (EFFICIENTNET-B2) ===")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Core Hardware: {device}\n")

    # 1. Directories Configuration
    # ---------------------------------------------------------
    base_dir = r"D:\Datasets\Medical\brisc2025\classification_task"
    val_clean_dir = os.path.join(base_dir, "val")
    val_zero_dir = os.path.join(base_dir, "val_mask_zero")

    weights_baseline = 'brisc_effnet_b2_v1.pth'
    weights_causal = 'brisc_effnet_b2_causal_v2.pth'
    batch_size = 32

    # 2. Data Loading
    # ---------------------------------------------------------
    data_transforms = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_clean = datasets.ImageFolder(val_clean_dir, data_transforms)
    dataset_zero = datasets.ImageFolder(val_zero_dir, data_transforms)

    loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_zero = DataLoader(dataset_zero, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = dataset_clean.classes

    # 3. Model Architecture Setup
    # ---------------------------------------------------------
    def get_model(weights_path):
        model = models.efficientnet_b2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return model.to(device)

    # 4. Execution & Auditing
    # ---------------------------------------------------------
    results = {}

    print(f"Auditing Baseline Model: {weights_baseline} ...")
    model_base = get_model(weights_baseline)
    base_clean_acc, base_clean_cls = evaluate_model(model_base, loader_clean, device, class_names)
    base_zero_acc, base_zero_cls = evaluate_model(model_base, loader_zero, device, class_names)

    print(f"Auditing Causal Defense Model: {weights_causal} ...")
    model_causal = get_model(weights_causal)
    causal_clean_acc, causal_clean_cls = evaluate_model(model_causal, loader_clean, device, class_names)
    causal_zero_acc, causal_zero_cls = evaluate_model(model_causal, loader_zero, device, class_names)

    # 5. Global Scorecard Generation
    # ---------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("                    FINAL DEFENSE COMPARISON REPORT")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Baseline (Standard)':<22} | {'Defense (Causal)':<20}")
    print("-" * 80)
    print(f"{'Clean Accuracy':<25} | {base_clean_acc:<22.4f} | {causal_clean_acc:<20.4f}")
    print(f"{'Zero-Out Accuracy':<25} | {base_zero_acc:<22.4f} | {causal_zero_acc:<20.4f}")

    # Calculate Macro Global Reliance (Drop in Accuracy)
    base_drop = base_clean_acc - base_zero_acc
    causal_drop = causal_clean_acc - causal_zero_acc
    print(f"{'Global Accuracy Drop':<25} | {base_drop:<22.4f} | {causal_drop:<20.4f}")
    print("=" * 80)

    # 6. Class-Level Shortcut Scorecard
    # ---------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("              CLASS-LEVEL SHORTCUT RELIANCE SCORES (SRS)")
    print("              (Calculated as Zero-Out Acc / Clean Acc)")
    print("=" * 80)
    print(f"{'Tumor Class':<15} | {'Baseline SRS':<20} | {'Causal SRS (Lower is Better)':<30}")
    print("-" * 80)

    for cls in class_names:
        # Prevent division by zero mathematically
        base_srs = base_zero_cls[cls] / max(base_clean_cls[cls], 0.0001)
        causal_srs = causal_zero_cls[cls] / max(causal_clean_cls[cls], 0.0001)

        # Format the visual indicator for improvement
        diff = base_srs - causal_srs
        trend = f"(Improved by {diff:.4f})" if diff > 0 else ""
        if cls == "no_tumor":
            trend = "(Control Variable)"

        print(f"{cls:<15} | {base_srs:<20.4f} | {causal_srs:<15.4f} {trend}")

    print("=" * 80)
    print("Audit Notes: A successfully defended model should show severe accuracy collapse (low SRS)")
    print("when the pathology is erased, proving it no longer relies on the background.")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()