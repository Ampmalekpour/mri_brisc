# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : test_all_models_xai_1.py
# Purpose       :
#   Grad-CAM attribution gallery per architecture (note: uses timm 'legacy_xception', unlike the other audit scripts).
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import timm
from PIL import Image


# =====================================================================
# 1. MODEL ARCHITECTURE & CONFIGURATION
# =====================================================================
def get_model_config(model_name, num_classes):
    """
    Returns the initialized architecture, resolution, weights path, and
    the target layer required for XAI attribution visualization.
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
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer = model.features[-1]

    elif model_name == 'EfficientNet-B1':
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer = model.features[-1]

    elif model_name == 'EfficientNet-B2':
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer = model.features[-1]

    elif model_name == 'Xception':
        model = timm.create_model('legacy_xception', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.bn4

    elif model_name == 'VGG16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        target_layer = model.features[28]

    elif model_name == 'VGG19':
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        target_layer = model.features[34]

    elif model_name == 'InceptionV3':
        model = models.inception_v3(weights=None, aux_logits=True, transform_input=True)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.Mixed_7c

    return model, resolution, weights_path, target_layer


# =====================================================================
# 2. XAI ENGINE IMPLEMENTATIONS (GRAD-CAM, IG, SMOOTHGRAD)
# =====================================================================
class XAIEngine:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _forward_wrapper(self, input_tensor):
        outputs = self.model(input_tensor)
        if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        return outputs

    def generate_gradcam(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()

        outputs = self._forward_wrapper(input_tensor)

        one_hot = torch.zeros_like(outputs)
        one_hot[0][target_class] = 1.0
        outputs.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return self._normalize(cam)

    def generate_integrated_gradients(self, input_tensor, target_class, steps=30):
        self.model.eval()
        baseline = torch.zeros_like(input_tensor)
        scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]

        grads = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.clone().detach().requires_grad_(True)
            outputs = self._forward_wrapper(scaled_input)

            self.model.zero_grad()
            score = outputs[0, target_class]
            score.backward()
            grads.append(scaled_input.grad.detach().cpu().numpy())

        avg_grads = np.mean(np.array(grads), axis=0)
        delta = (input_tensor - baseline).detach().cpu().numpy()
        ig_map = np.abs(np.sum(avg_grads * delta, axis=1)).squeeze()
        return self._normalize(ig_map)

    def generate_smoothgrad(self, input_tensor, target_class, stdev_spread=0.15, n_samples=20):
        self.model.eval()
        stdev = stdev_spread * (input_tensor.max() - input_tensor.min()).item()
        total_gradients = np.zeros(input_tensor.shape[2:])

        for _ in range(n_samples):
            noise = torch.normal(0, stdev, size=input_tensor.size()).to(input_tensor.device)
            noisy_input = (input_tensor + noise).clone().detach().requires_grad_(True)

            outputs = self._forward_wrapper(noisy_input)
            self.model.zero_grad()
            score = outputs[0, target_class]
            score.backward()

            grad = noisy_input.grad.detach().cpu().numpy()
            total_gradients += np.sum(np.abs(grad), axis=1).squeeze()

        smooth_map = total_gradients / n_samples
        return self._normalize(smooth_map)

    @staticmethod
    def _normalize(cam):
        if np.max(cam) != np.min(cam):
            return (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return np.zeros_like(cam)


# =====================================================================
# 3. UTILITIES FOR VISUALIZATION
# =====================================================================
def unnormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)


def apply_colormap(heatmap_2d, rgb_img, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_2d), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = (1 - alpha) * rgb_img + alpha * heatmap
    return np.clip(overlay, 0, 1)


# =====================================================================
# 4. MAIN AUDIT & VISUALIZATION EXECUTION LOOP
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INITIALIZING FULL XAI SUITE (GRAD-CAM, IG, SMOOTHGRAD) ===")
    print(f"Execution Core Hardware: {device}\n")

    base_data_dir = r"D:\Datasets\Medical\brisc2025\classification_task\val"
    output_dir = r"D:\Datasets\Medical\brisc2025\xai_full_audit"
    os.makedirs(output_dir, exist_ok=True)

    models_to_test = [
        'EfficientNet-B0',
        'EfficientNet-B1',
        'EfficientNet-B2',
        'Xception',
        'VGG16',
        'VGG19',
        'InceptionV3'
    ]

    sample_dataset = datasets.ImageFolder(base_data_dir)
    class_names = sample_dataset.classes
    num_classes = len(class_names)

    # Pick 1 representative evaluation sample per class
    samples_per_class = {}
    for img_path, class_idx in sample_dataset.samples:
        class_name = class_names[class_idx]
        if class_name not in samples_per_class:
            samples_per_class[class_name] = img_path
        if len(samples_per_class) == num_classes:
            break

    print(f"Selected evaluation samples for classes: {list(samples_per_class.keys())}\n")

    for model_name in models_to_test:
        print(f"\nProcessing Multi-Method Attributions for: {model_name}...")
        model, resolution, weights_path, target_layer = get_model_config(model_name, num_classes)

        if not os.path.exists(weights_path):
            print(f"  [Skipped] Weights missing at: {weights_path}")
            continue

        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)

        # Recursively disable in-place operations across all modules
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

        # Force evaluation mode to prevent BatchNorm batch-size errors
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        xai_engine = XAIEngine(model, target_layer)

        # Setup figure grid: 4 columns per class (Original | Grad-CAM | Integrated Gradients | SmoothGrad)
        fig, axes = plt.subplots(len(samples_per_class), 4, figsize=(16, 4 * len(samples_per_class)))
        fig.suptitle(f"XAI Comparison Audit: {model_name}", fontsize=16, fontweight='bold')

        for idx, (gt_class, img_path) in enumerate(samples_per_class.items()):
            raw_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(raw_img).unsqueeze(0).to(device)

            # Forward pass for top prediction class
            outputs = xai_engine._forward_wrapper(input_tensor)
            pred_class_idx = torch.argmax(outputs, dim=1).item()
            pred_class = class_names[pred_class_idx]
            conf = torch.softmax(outputs, dim=1)[0, pred_class_idx].item()

            # Compute XAI Heatmaps
            gradcam_map = xai_engine.generate_gradcam(input_tensor, pred_class_idx)
            ig_map = xai_engine.generate_integrated_gradients(input_tensor, pred_class_idx)
            smooth_map = xai_engine.generate_smoothgrad(input_tensor, pred_class_idx)

            rgb_img = unnormalize_image(input_tensor)
            title_color = 'green' if gt_class == pred_class else 'red'

            # Column 1: Original Image
            axes[idx, 0].imshow(rgb_img)
            axes[idx, 0].set_title(f"GT: {gt_class}\nPred: {pred_class} ({conf:.1%})", color=title_color)
            axes[idx, 0].axis('off')

            # Column 2: Grad-CAM Overlay
            axes[idx, 1].imshow(apply_colormap(gradcam_map, rgb_img))
            axes[idx, 1].set_title("Grad-CAM (Coarse Regional)")
            axes[idx, 1].axis('off')

            # Column 3: Integrated Gradients Overlay
            axes[idx, 2].imshow(apply_colormap(ig_map, rgb_img))
            axes[idx, 2].set_title("Integrated Gradients (Pixel-wise)")
            axes[idx, 2].axis('off')

            # Column 4: SmoothGrad Overlay
            axes[idx, 3].imshow(apply_colormap(smooth_map, rgb_img))
            axes[idx, 3].set_title("SmoothGrad (Denoised Saliency)")
            axes[idx, 3].axis('off')

        plt.tight_layout()
        save_file = os.path.join(output_dir, f"{model_name}_full_xai_audit.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved full audit figure: {save_file}")

    print("\n=== FULL XAI AUDIT COMPLETE ===")
    print(f"All heatmap comparison grids saved to: {output_dir}")


if __name__ == '__main__':
    main()