# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : test_all_models_mask_tumor_visual.py
# Purpose       :
#   Grad-CAM + ROI overlap analysis for the 7 CNNs on clean / masked inputs.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import timm


# =====================================================================
# 1. GRAD-CAM, ROI, & VISUALIZATION ENGINE
# =====================================================================

class GradCAM:
    """
    Hook-based PyTorch Grad-CAM engine for extracting visual attribution maps.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register PyTorch hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if isinstance(output, tuple):
            output = output[0]

        score = output[0, target_class]
        score.backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]

        # Global average pooling on gradients
        weights = torch.mean(grads, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * acts, dim=0)
        cam = F.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode='bilinear', align_corners=False)
        return cam.squeeze()


def get_last_conv_layer(model, model_name):
    if model_name == 'InceptionV3':
        return model.Mixed_7c

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def compute_energy_roi(cam_map, roi_mask):
    cam_sum = torch.sum(cam_map)
    if cam_sum == 0:
        return 0.0
    roi_energy = torch.sum(cam_map * roi_mask.float())
    return (roi_energy / cam_sum).item()


def save_gradcam_plot(img_tensor, cam_tensor, mask_tensor, save_path, true_class, pred_class, class_names):
    """
    Generates a 3-panel figure: Original | Grad-CAM Overlay | ROI Mask
    """
    # 1. Un-normalize the image for visualization
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # 2. Extract CAM and Mask arrays
    cam = cam_tensor.cpu().numpy()
    mask = mask_tensor.cpu().numpy()

    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original
    axes[0].imshow(img)
    axes[0].set_title(f"Input Image\nTrue: {class_names[true_class]} | Pred: {class_names[pred_class]}")
    axes[0].axis('off')

    # Panel 2: Grad-CAM Overlay
    axes[1].imshow(img)
    axes[1].imshow(cam, cmap='jet', alpha=0.5)  # Alpha blends the heatmap
    axes[1].set_title("Grad-CAM Attention Map")
    axes[1].axis('off')

    # Panel 3: ROI Mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Extracted Pathology ROI")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# =====================================================================
# 2. MODEL CONFIGURATION FACTORY
# =====================================================================

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


# =====================================================================
# 3. EVALUATION & EXPLAINABILITY PIPELINE
# =====================================================================

def evaluate_model_with_explainability(model, test_loader, clean_loader, zero_loader, device,
                                       model_name, variant_name, class_names):
    all_preds = []
    all_labels = []
    e_roi_scores = []

    # Setup directory for saving visual outputs
    vis_dir = os.path.join("cam_visualizations", model_name, variant_name.replace(" ", "_"))
    os.makedirs(vis_dir, exist_ok=True)
    images_saved = 0
    MAX_VISUALS_TO_SAVE = 5  # Adjust this to save more or fewer images per variant

    target_layer = get_last_conv_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    # Note: We zip test_loader, clean_loader, and zero_loader together!
    for (inputs, labels), (clean_inputs, _), (zero_inputs, _) in zip(test_loader, clean_loader, zero_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        clean_inputs, zero_inputs = clean_inputs.to(device), zero_inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            if (isinstance(outputs, tuple) or hasattr(outputs, 'logits')) and model_name == 'InceptionV3':
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # FIX: Mask is ALWAYS derived by comparing Clean Baseline against Zero Out variant.
        roi_masks = torch.abs(clean_inputs - zero_inputs).sum(dim=1) > 1e-2

        for i in range(inputs.size(0)):
            single_input = inputs[i:i + 1]
            predicted_class = preds[i].item()
            true_class = labels[i].item()
            roi_mask = roi_masks[i]

            # Generate CAM
            cam_map = grad_cam.generate_cam(single_input, predicted_class)

            # Compute Metric
            e_roi = compute_energy_roi(cam_map, roi_mask)
            e_roi_scores.append(e_roi)

            # Save Visualizations for the first few images
            if images_saved < MAX_VISUALS_TO_SAVE:
                save_path = os.path.join(vis_dir,
                                         f"sample_{images_saved}_true_{class_names[true_class]}_pred_{class_names[predicted_class]}.png")
                save_gradcam_plot(inputs[i], cam_map, roi_mask, save_path, true_class, predicted_class, class_names)
                images_saved += 1

    mean_acc = accuracy_score(all_labels, all_preds)
    mean_e_roi = float(np.mean(e_roi_scores))

    return np.array(all_labels), np.array(all_preds), mean_acc, mean_e_roi


# =====================================================================
# 4. MAIN AUDIT EXECUTION
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== INITIALIZING QUANTITATIVE CAUSAL & EXPLAINABILITY AUDIT ===")
    print(f"Execution Core Hardware: {device}\n")

    base_data_dir = r"D:\Datasets\Medical\brisc2025\classification_task"
    variants = {
        "Clean Baseline": "val",
        "Zero Out (Pathology Erased)": "val_mask_zero",
        "Gaussian Noise (Pathology Scrambled)": "val_mask_gaussian"
    }

    models_to_test = [
        'EfficientNet-B0',
        'EfficientNet-B1',
        # 'EfficientNet-B2', # Uncomment others as needed
        # 'Xception',
        # 'VGG16',
        # 'VGG19',
        # 'InceptionV3'
    ]

    sample_dataset = datasets.ImageFolder(os.path.join(base_data_dir, "val"))
    class_names = sample_dataset.classes
    num_classes = len(class_names)

    master_summary_table = {m: {} for m in models_to_test}

    for model_name in models_to_test:
        print("\n" + "#" * 75)
        print(f" ARCHITECTURE AUDITING & EXPLAINABILITY STAGE: {model_name}")
        print("#" * 75)

        model, resolution, weights_relative_path = get_model_config(model_name, num_classes)

        if not os.path.exists(weights_relative_path):
            print(f"[CRITICAL ERROR] Weights missing at: {weights_relative_path}. Skipping model.")
            continue

        model.load_state_dict(torch.load(weights_relative_path, map_location=device))
        model = model.to(device)
        model.eval()

        data_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load Reference Dataloaders for Masking Math
        clean_dataset = datasets.ImageFolder(os.path.join(base_data_dir, "val"), data_transforms)
        clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False, num_workers=0)

        zero_dataset = datasets.ImageFolder(os.path.join(base_data_dir, "val_mask_zero"), data_transforms)
        zero_loader = DataLoader(zero_dataset, batch_size=32, shuffle=False, num_workers=0)

        for variant_display_name, folder_name in variants.items():
            variant_path = os.path.join(base_data_dir, folder_name)

            if not os.path.exists(variant_path):
                print(f"  [Warning] Directory '{folder_name}' missing. Skipping.")
                continue

            # This is the loader being actively evaluated
            dataset = datasets.ImageFolder(variant_path, data_transforms)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            # Run evaluation (Passing clean_loader and zero_loader for mask calc)
            labels, preds, acc, e_roi = evaluate_model_with_explainability(
                model, loader, clean_loader, zero_loader, device,
                model_name, variant_display_name, class_names
            )

            master_summary_table[model_name][variant_display_name] = {'acc': acc, 'e_roi': e_roi}

            print("\n" + "-" * 60)
            print(f" TARGET MATRIX: {variant_display_name} ({folder_name})")
            print(f" Target Resolution: {resolution}x{resolution}")
            print("-" * 60)
            print(f" Overall Accuracy   : {acc:.4f}")
            print(f" Mean Energy-in-ROI : {e_roi:.4f} (E_ROI Ratio)")
            print("\nClassification Metrics:")
            print(classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0))

    # --- PUBLICATION SCORECARD TABLE ---
    print("\n\n" + "=" * 95)
    print("                 FINAL GLOBAL CAUSAL & EXPLAINABILITY AUDIT SCORECARD")
    print("=" * 95)
    print(f"{'Model Architecture':<18} | {'Clean Baseline':<18} | {'Zero Out (Erased)':<20} | {'Gaussian Noise':<20}")
    print(f"{'':<18} | {'Acc / E_ROI':<18} | {'Acc / E_ROI':<20} | {'Acc / E_ROI':<20}")
    print("-" * 95)

    for model_name, results in master_summary_table.items():
        if not results:
            continue
        b = results.get("Clean Baseline", {'acc': 0.0, 'e_roi': 0.0})
        z = results.get("Zero Out (Pathology Erased)", {'acc': 0.0, 'e_roi': 0.0})
        g = results.get("Gaussian Noise (Pathology Scrambled)", {'acc': 0.0, 'e_roi': 0.0})

        print(
            f"{model_name:<18} | {b['acc']:.4f} / {b['e_roi']:.4f} | {z['acc']:.4f} / {z['e_roi']:.4f} | {g['acc']:.4f} / {g['e_roi']:.4f}")


if __name__ == '__main__':
    main()