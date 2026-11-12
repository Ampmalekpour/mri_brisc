# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Dual-Branch Siamese Network/test.py
# Purpose       :
#   Audits the Siamese model on clean / zero-out / Gaussian sets.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =====================================================================
# 1. MODEL ARCHITECTURE RECONSTRUCTION (Must match training footprint)
# =====================================================================
class DualBranchNet(nn.Module):
    def __init__(self, num_classes=4, feat_dim=128):
        super().__init__()
        # Shared Feature Extractor Backbone
        resnet = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Outputs [B, 2048, 1, 1]

        # Re-added to match the saved checkpoint dictionary keys exactly
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )

        # Final Linear Probe Classifier Head
        self.classifier_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward_encoder(self, x):
        features = self.encoder(x)
        return torch.flatten(features, 1)

    def forward_classify(self, x):
        feat = self.forward_encoder(x)
        return self.classifier_head(feat)

# =====================================================================
# 2. EVALUATION CORE EXECUTION LOOP
# =====================================================================
def evaluate_model_on_variant(model, test_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Route inputs directly through the classification pipeline branch
            outputs = model.forward_classify(inputs)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# =====================================================================
# 3. MAIN AUDIT SUITE CONTROLLER
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INITIALIZING SIAMESE DUAL-BRANCH CAUSAL AUDIT SUITE ===")
    print(f"Execution Core Hardware: {device}\n")

    # Base Path Configurations
    base_data_dir = r"D:\Datasets\Medical\brisc2025\classification_task"
    weights_path = 'brisc_dual_branch_model.pth'
    resolution = 224  # Standard ResNet50 input resolution constraint

    variants = {
        "Clean Baseline": "val",
        "Zero Out (Pathology Erased)": "val_mask_zero",
        "Gaussian Noise (Pathology Scrambled)": "val_mask_gaussian"
    }

    # Extract target class names from validation baseline folder
    baseline_val_path = os.path.join(base_data_dir, "val")
    if not os.path.exists(baseline_val_path):
        print(f"[CRITICAL ERROR] Baseline validation directory missing at: {baseline_val_path}")
        return

    sample_dataset = datasets.ImageFolder(baseline_val_path)
    class_names = sample_dataset.classes
    num_classes = len(class_names)
    print(f"Detected Target Classes: {class_names}")

    # Initialize model structure
    model = DualBranchNet(num_classes=num_classes)

    if not os.path.exists(weights_path):
        print(f"[CRITICAL ERROR] Target Siamese weights missing at: {weights_path}. Aborting.")
        return

    print(f"Loading local parameter maps from: {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    # Dynamic Data Transform matching model architecture configuration
    data_transforms = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Score tracking dictionary
    master_summary_table = {}

    # Iterate through each dataset slice variant
    for variant_display_name, folder_name in variants.items():
        variant_path = os.path.join(base_data_dir, folder_name)

        if not os.path.exists(variant_path):
            print(f"\n  [Warning] Dataset Variant Folder '{folder_name}' missing at {variant_path}. Skipping.")
            continue

        # Instantiate specific ImageFolder loader configuration
        dataset = datasets.ImageFolder(variant_path, data_transforms)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        # Compute predictions
        labels, preds = evaluate_model_on_variant(model, loader, device)
        acc = accuracy_score(labels, preds)

        # Archive score metric
        master_summary_table[variant_display_name] = acc

        # Print detailed matrix layout reports per folder variant
        print("\n" + "-" * 60)
        print(f" TARGET MATRIX: {variant_display_name} ({folder_name})")
        print(f" Resolution Constraints: {resolution}x{resolution}")
        print("-" * 60)
        print(f"Overall Set Target Accuracy: {acc:.4f}\n")

        print("Classification Metrics Distribution:")
        print(classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0))

        print("Confusion Matrix Layout:")
        print(f"Labels Order: {class_names}")
        print(confusion_matrix(labels, preds))
        print("-" * 60)

    # =====================================================================
    # 4. FINAL GLOBAL CAUSAL AUDIT SCORECARD REPORT
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("                    FINAL GLOBAL CAUSAL AUDIT SCORECARD REPORT")
    print("=" * 80)
    print(f"{'Model Architecture':<25} | {'Clean Baseline':<16} | {'Zero Out':<14} | {'Gaussian Noise':<14}")
    print("-" * 80)

    b_acc = master_summary_table.get("Clean Baseline", 0.0)
    z_acc = master_summary_table.get("Zero Out (Pathology Erased)", 0.0)
    g_acc = master_summary_table.get("Gaussian Noise (Pathology Scrambled)", 0.0)

    print(f"{'Siamese ResNet50':<25} | {b_acc:<16.4f} | {z_acc:<14.4f} | {g_acc:<14.4f}")
    print("=" * 80)

    print("Interpretation Checklist for Manuscript Analysis:")
    print("1. Perfect Siamese Integration: 'Clean Baseline' accuracy stays near 100%.")
    print("2. Causal Validation: If 'Zero Out' and 'Gaussian Noise' performance drops sharply down to")
    print("   chance thresholds (~0.2500) or misclassifies tumor instances into 'no_tumor', it confirms")
    print("   the model's representations are strictly linked to true biological boundaries, not shortcuts.\n")


if __name__ == '__main__':
    main()