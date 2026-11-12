# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : EfficientNets/test_b0.py
# Purpose       :
#   Evaluate EfficientNet-B0 on the test split.
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
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use your validation/test directory
    data_dir = r"D:\Datasets\Medical\brisc2025\classification_task"
    val_dir = os.path.join(data_dir, "val")

    batch_size = 32
    weights_path = 'brisc_effnet_b0_v1.pth'

    # 2. Data Transforms (Must match the validation transforms from training)
    # ---------------------------------------------------------
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Data Loader
    # ---------------------------------------------------------
    test_dataset = datasets.ImageFolder(val_dir, data_transforms)
    # num_workers=0 to prevent the Windows paging file crash
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = test_dataset.classes
    print(f"Classes found: {class_names}")

    # 4. Model Initialization
    # ---------------------------------------------------------
    # Recreate the exact architecture
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    # Load your trained weights
    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()  # CRITICAL: Set to evaluation mode

    # 5. Evaluation Loop
    # ---------------------------------------------------------
    all_preds = []
    all_labels = []

    print("Running inference on the test set...")
    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Move predictions and labels back to CPU for scikit-learn
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Calculate Metrics
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)

    # Calculate overall accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc:.4f}\n")

    # Generate the full classification report
    # This automatically computes Precision, Recall, F1 for each class,
    # plus the Macro and Weighted averages.
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)


if __name__ == '__main__':
    main()