# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Vgg/test_v16.py
# Purpose       :
#   Evaluate VGG16.
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
    weights_path = 'brisc_vgg16_v1.pth'

    # Matches training size of 224x224
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(val_dir, data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = test_dataset.classes
    print(f"Classes found: {class_names}")

    # Initialize structure matching weights
    model = models.vgg16(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, len(class_names))

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

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS (VGG16)")
    print("=" * 50)

    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc:.4f}\n")

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

if __name__ == '__main__':
    main()