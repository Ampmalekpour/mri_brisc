# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : test_all_models_cdm_srs.py
# Purpose       :
#   Computes Shortcut Reliance Score (SRS = mean P(true class | masked) / mean P(true class | clean)) per class
#   and a Counterfactual Discrepancy Map (CDM = mean |feat_clean - feat_masked|) for ONE validation image per model.
#   OUTPUT PLOTS: cdm_output_plots/*.png
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import timm


# -------------------------------------------------------------------------
# 1. CORE CDM HOOK INFRASTRUCTURE
# -------------------------------------------------------------------------
class FeatureExtractorHook:
    """Intercerts and retains target convolutional layer feature tensors."""

    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        # Handle cases where output might be wrapped in specific model types
        if isinstance(output, tuple):
            self.features = output[0]
        else:
            self.features = output


def get_final_conv_layer(model, model_name):
    """
    Locates the absolute last convolutional module before pooling layers
    across all targeted architectures.
    """
    if 'EfficientNet' in model_name:
        # The last conv layer is the final block in the features sequential box
        return model.features[-1]
    elif 'VGG' in model_name:
        # The final Conv2d layer before adaptive pooling
        return model.features[-2]
    elif model_name == 'Xception':
        # The absolute last element block of the feature stream
        return model.conv4
    elif model_name == 'InceptionV3':
        # Conv2d_4a_3x3 or Mixed_7c depending on version; Mixed_7c is the final block
        return model.Mixed_7c
    raise ValueError(f"Target layer extraction mapping not registered for: {model_name}")


def generate_cdm_heatmap(clean_features, masked_features):
    """
    Calculates the 2D Counterfactual Discrepancy Map matrix (D)
    by analyzing feature space divergence.
    """
    # Compute absolute difference along channel space: Mean(|A - A_tilde|)
    diff = torch.abs(clean_features - masked_features)
    discrepancy_map = torch.mean(diff, dim=1).squeeze(0).cpu().numpy()

    # Min-Max Normalization normalization safeguard
    denom = (discrepancy_map.max() - discrepancy_map.min()) + 1e-8
    normalized_map = (discrepancy_map - discrepancy_map.min()) / denom
    return normalized_map


# -------------------------------------------------------------------------
# 2. RUNTIME RECONSTRUCTIONS
# -------------------------------------------------------------------------
def get_model_config(model_name, num_classes):
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
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B1':
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B2':
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'Xception':
        model = timm.create_model('xception', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'VGG16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'VGG19':
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'InceptionV3':
        model = models.inception_v3(weights=None, aux_logits=True, transform_input=True)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model, resolution, weights_path


def evaluate_model_on_variant(model, test_loader, device, model_name):
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if model_name == 'InceptionV3' and (isinstance(outputs, tuple) or hasattr(outputs, 'logits')):
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs.logits

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# -------------------------------------------------------------------------
# 3. MASTER EXECUTION ENGINE
# -------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INITIALIZING MASTER CAUSAL AUDIT + XAI SUITE ===")
    print(f"Execution Core Hardware: {device}\n")

    base_data_dir = r"D:\Datasets\Medical\brisc2025\classification_task"
    variants = {
        "Clean Baseline": "val",
        "Zero Out (Pathology Erased)": "val_mask_zero",
        "Gaussian Noise (Pathology Scrambled)": "val_mask_gaussian"
    }

    models_to_test = ['EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'Xception', 'VGG16', 'VGG19',
                      'InceptionV3']

    sample_dataset = datasets.ImageFolder(os.path.join(base_data_dir, "val"))
    class_names = sample_dataset.classes
    num_classes = len(class_names)

    master_summary_table = {m: {} for m in models_to_test}
    srs_summary_table = {m: {} for m in models_to_test}

    os.makedirs("cdm_output_plots", exist_ok=True)

    for model_name in models_to_test:
        print("\n" + "#" * 70)
        print(f" ARCHITECTURE AUDITING STAGE: {model_name}")
        print("#" * 70)

        model, resolution, weights_relative_path = get_model_config(model_name, num_classes)

        if not os.path.exists(weights_relative_path):
            print(f"[CRITICAL ERROR] Weights missing at location: {weights_relative_path}. Skipping model.")
            continue

        model.load_state_dict(torch.load(weights_relative_path, map_location=device))
        model = model.to(device)
        model.eval()

        data_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Extract data vectors across variants for tracking
        variant_probs = {}

        for variant_display_name, folder_name in variants.items():
            variant_path = os.path.join(base_data_dir, folder_name)
            if not os.path.exists(variant_path):
                continue

            dataset = datasets.ImageFolder(variant_path, data_transforms)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            labels, preds, probs = evaluate_model_on_variant(model, loader, device, model_name)
            acc = accuracy_score(labels, preds)

            master_summary_table[model_name][variant_display_name] = acc
            variant_probs[variant_display_name] = (labels, probs)

            print(f"  -> Successfully computed tracking performance map on: {variant_display_name} | Acc: {acc:.4f}")

        # Compute Class-Level Shortcut Reliance Scores (SRS)
        if "Clean Baseline" in variant_probs and "Zero Out (Pathology Erased)" in variant_probs:
            labels, clean_p = variant_probs["Clean Baseline"]
            _, zero_p = variant_probs["Zero Out (Pathology Erased)"]
            _, gauss_p = variant_probs["Gaussian Noise (Pathology Scrambled)"]

            srs_summary_table[model_name] = {}
            for idx, class_label in enumerate(class_names):
                class_mask = (labels == idx)
                if np.sum(class_mask) == 0:
                    continue

                # SRS = P(y|x_mask) / P(y|x)
                avg_clean_prob = np.mean(clean_p[class_mask, idx])
                avg_zero_prob = np.mean(zero_p[class_mask, idx])
                avg_gauss_prob = np.mean(gauss_p[class_mask, idx])

                srs_zero = avg_zero_prob / (avg_clean_prob + 1e-8)
                srs_gauss = avg_gauss_prob / (avg_clean_prob + 1e-8)

                srs_summary_table[model_name][class_label] = {"Zero-Out SRS": srs_zero, "Gaussian SRS": srs_gauss}

        # -----------------------------------------------------------------
        # 4. VISUAL CDM GENERATOR LOOP (Per Model Execution Example)
        # -----------------------------------------------------------------
        try:
            target_layer = get_final_conv_layer(model, model_name)
            extractor = FeatureExtractorHook()
            hook_handle = target_layer.register_forward_hook(extractor)

            # Pull exactly 1 validation visual tracking sample pair from baseline and zero_out sets
            clean_ds = datasets.ImageFolder(os.path.join(base_data_dir, "val"), data_transforms)
            zero_ds = datasets.ImageFolder(os.path.join(base_data_dir, "val_mask_zero"), data_transforms)

            # Select an arbitrary early sample (e.g., sample index 10)
            sample_idx = min(10, len(clean_ds) - 1)
            img_clean, lbl = clean_ds[sample_idx]
            img_zero, _ = zero_ds[sample_idx]

            # Shape transformation additions [1, C, H, W]
            tensor_clean = img_clean.unsqueeze(0).to(device)
            tensor_zero = img_zero.unsqueeze(0).to(device)

            # Pass forward passes sequentially to harvest hooks
            with torch.no_grad():
                model(tensor_clean)
                features_clean = extractor.features.clone()

                model(tensor_zero)
                features_zero = extractor.features.clone()

            hook_handle.remove()  # Clean up hook resource footprint

            # Process map calculation
            cdm_map = generate_cdm_heatmap(features_clean, features_zero)

            # Save the diagnostic visualization
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1)
            plt.title("Original Input Scan (x)")
            plt.imshow(np.transpose(img_clean.numpy(), (1, 2, 0)).clip(0, 1))
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Counterfactual Intervention")
            plt.imshow(np.transpose(img_zero.numpy(), (1, 2, 0)).clip(0, 1))
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title(f"CDM Map ({model_name})")
            # Resize CDM map matching native input window frame metrics
            plt.imshow(cdm_map, cmap='inferno', interpolation='bilinear')
            plt.colorbar()
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"cdm_output_plots/{model_name}_cdm_report.png")
            plt.close()
            print(f"  [XAI INFO] Generated and saved CDM counterfactual map for {model_name}.")

        except Exception as e:
            print(f"  [XAI WARNING] Could not run CDM generation hook array loop on {model_name}: {e}")

    # --- FINAL MASTER METRIC COMPARISON SUMMARY ---
    print("\n\n" + "=" * 80)
    print("                    FINAL GLOBAL CAUSAL AUDIT SCORECARD REPORT")
    print("=" * 80)
    print(f"{'Model Architecture':<20} | {'Clean Baseline':<16} | {'Zero Out':<14} | {'Gaussian Noise':<14}")
    print("-" * 80)
    for model_name, results in master_summary_table.items():
        if not results: continue
        print(
            f"{model_name:<20} | {results.get('Clean Baseline', 0.0):<16.4f} | {results.get('Zero Out (Pathology Erased)', 0.0):<14.4f} | {results.get('Gaussian Noise (Pathology Scrambled)', 0.0):<14.4f}")
    print("=" * 80)

    # --- NOVEL COMPONENT SUBMISSION REPORT: SRS SCORE MATRIX ---
    print("\n\n" + "=" * 80)
    print("                 XAI REGULATORY SUITE: SHORTCUT RELIANCE SCORES (SRS)")
    print("=" * 80)
    print(f"{'Model Name':<18} | {'Class Categorization':<20} | {'Zero-Out SRS':<16} | {'Gaussian SRS':<14}")
    print("-" * 80)
    for model_name, class_metrics in srs_summary_table.items():
        for cls_name, metrics in class_metrics.items():
            # Fixed keys here to match the dictionary definition exactly
            print(
                f"{model_name:<18} | {cls_name:<20} | {metrics['Zero-Out SRS']:<16.4f} | {metrics['Gaussian SRS']:<14.4f}")
    print("=" * 80)
    print("SRS Audit Metric Notes: Values approaching 1.0 indicate severe shortcut cheating dependencies.\n")

if __name__ == '__main__':
    main()