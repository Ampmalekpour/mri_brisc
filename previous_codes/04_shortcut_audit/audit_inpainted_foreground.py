# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : test_all_models_tumor_mask_2.py
# Purpose       :
#   Evaluates all 7 CNNs on clean vs inpainted vs foreground-only variants of the TRAIN split.
#   OUTPUT: miccai_diagnostic_audit_report.txt (kept next to this file).
#   CAVEAT: evaluated on the training split, so the 'clean' accuracy (~99.9%) is not held-out accuracy.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import timm


class Logger(object):
    """
    Dual-stream logger that simultaneously outputs to stdout terminal
    and writes to a log file on disk.
    """

    def __init__(self, filename="miccai_diagnostic_audit_report.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_model_config(model_name, num_classes):
    """
    Returns the initialized architecture, resolution, and weights file name
    tailored precisely to how each model was compiled.
    """
    weights_map = {
        'EfficientNet-B0': ('EfficientNets', 'brisc_effnet_b0_v1.pth', 224),
        'EfficientNet-B1': ('EfficientNets', 'brisc_effnet_b1_v1.pth', 240),
        'EfficientNet-B2': ('EfficientNets', 'brisc_effnet_b2_v1.pth', 260),
        'Xception': ('Xceptions', 'brisc_xception_v1.pth', 299),
        'VGG16': ('Vgg', 'brisc_vgg16_v1.pth', 224),
        'VGG19': ('Vgg', 'brisc_vgg19_v1.pth', 224),
        'InceptionV3': ('Inception', 'brisc_inceptionv3_v1.pth', 299)
    }

    folder, file_name, resolution = weights_map[model_name]
    weights_path = os.path.join(folder, file_name)

    if model_name == 'EfficientNet-B0':
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'EfficientNet-B1':
        model = models.efficientnet_b1(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'EfficientNet-B2':
        model = models.efficientnet_b2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'Xception':
        model = timm.create_model('xception', pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'VGG16':
        model = models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'VGG19':
        model = models.vgg19(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'InceptionV3':
        model = models.inception_v3(weights=None, aux_logits=True, transform_input=True)
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model, resolution, weights_path


def evaluate_model_on_variant(model, test_loader, device, model_name):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            if (isinstance(outputs, tuple) or hasattr(outputs, 'logits')) and model_name == 'InceptionV3':
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs.logits

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    report_file = "miccai_diagnostic_audit_report.txt"
    sys.stdout = Logger(report_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INITIALIZING MICCAI REPRESENTATION AUDIT SUITE ===")
    print(f"Execution Core Hardware: {device}")
    print(f"Output Audit File: {report_file}\n")

    base_data_dir = r"D:\Datasets\Medical\brisc2025\classification_task"

    # Target diagnostic variants generated by the MICCAI script
    variants = {
        "Clean Baseline": "train",
        "Anatomically Inpainted (Healthy Brain)": "train_mask_inpainted",
        "Foreground Isolated (Pathology Only)": "train_foreground_only"
    }

    models_to_test = [
        'EfficientNet-B0',
        'EfficientNet-B1',
        'EfficientNet-B2',
        'Xception',
        'VGG16',
        'VGG19',
        'InceptionV3'
    ]

    baseline_folder = os.path.join(base_data_dir, variants["Clean Baseline"])
    if not os.path.exists(baseline_folder):
        print(f"[CRITICAL ERROR] Baseline folder not found at {baseline_folder}.")
        return

    sample_dataset = datasets.ImageFolder(baseline_folder)
    class_names = sample_dataset.classes
    num_classes = len(class_names)

    master_summary_table = {m: {} for m in models_to_test}

    for model_name in models_to_test:
        print("\n" + "#" * 70)
        print(f" ARCHITECTURE AUDITING STAGE: {model_name}")
        print("#" * 70)

        model, resolution, weights_relative_path = get_model_config(model_name, num_classes)

        if not os.path.exists(weights_relative_path):
            print(f"[CRITICAL ERROR] Weights missing at location: {weights_relative_path}. Skipping model.")
            continue

        print(f"Loading local parameter maps from: {weights_relative_path}...")
        model.load_state_dict(torch.load(weights_relative_path, map_location=device))
        model = model.to(device)
        model.eval()

        data_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for variant_display_name, folder_name in variants.items():
            variant_path = os.path.join(base_data_dir, folder_name)

            if not os.path.exists(variant_path):
                print(f"  [Warning] Dataset Variant Folder '{folder_name}' missing. Skipping variant.")
                continue

            dataset = datasets.ImageFolder(variant_path, data_transforms)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            labels, preds = evaluate_model_on_variant(model, loader, device, model_name)
            acc = accuracy_score(labels, preds)

            master_summary_table[model_name][variant_display_name] = acc

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

    print("\n\n" + "=" * 85)
    print("                    FINAL MICCAI DIAGNOSTIC AUDIT SCORECARD")
    print("=" * 85)
    print(
        f"{'Model Architecture':<20} | {'Clean Baseline':<16} | {'Inpainted (Healthy)':<20} | {'Foreground Only':<16}")
    print("-" * 85)

    for model_name, results in master_summary_table.items():
        if not results:
            continue
        b_acc = results.get("Clean Baseline", 0.0)
        i_acc = results.get("Anatomically Inpainted (Healthy Brain)", 0.0)
        f_acc = results.get("Foreground Isolated (Pathology Only)", 0.0)
        print(f"{model_name:<20} | {b_acc:<16.4f} | {i_acc:<20.4f} | {f_acc:<16.4f}")

    print("=" * 85)
    print("\nMICCAI Diagnostic Interpretation Checklist:")
    print("1. Inpainted High Accuracy (>0.75): Confirms model relies heavily on background/contextual shortcuts.")
    print("2. Inpainted Low Accuracy (<0.30): Indicates model heavily depends on internal lesion features.")
    print("3. Foreground Only Drop: Quantifies accuracy loss when surrounding anatomy is removed.")
    print(f"\nAll logs successfully saved to: {os.path.abspath(report_file)}\n")


if __name__ == '__main__':
    main()