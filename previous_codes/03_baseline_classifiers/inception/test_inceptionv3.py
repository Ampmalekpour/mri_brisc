# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Inception/test_v0.py
# Purpose       :
#   Evaluate InceptionV3.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = r"D:\Datasets\Medical\brisc2025\classification_task"
    val_dir = os.path.join(data_dir, "val")
    batch_size = 32
    weights_path = 'brisc_inceptionv3_v1.pth'

    # 1. Verification Step: Standard Inception V3 normalizations
    data_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(val_dir, data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = test_dataset.classes
    print(f"Classes found: {class_names}")

    # 2. FIX: Explicitly set transform_input=True to force match the training footprint
    model = models.inception_v3(weights=None, aux_logits=True, transform_input=True)

    num_ftrs_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs_aux, len(class_names))
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on the test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 3. EXTRA SAFEGUARDS FOR INCEPTION TEST EVAL
            outputs = model(inputs)

            # In some Torch versions, calling eval() still returns a named tuple for Inception.
            # We handle it defensively here:
            if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs.logits

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS (Inception V3)")
    print("=" * 50)

    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc:.4f}\n")

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)


if __name__ == '__main__':
    main()