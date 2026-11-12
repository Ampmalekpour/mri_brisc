# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/test_yolo_con_ada.py
# Purpose       :
#   Evaluates the contrastive adaptive-margin YOLO model.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
from ultralytics import YOLO


# =====================================================================
# ARCHITECTURE RECONSTRUCTION (Must match training definitions exactly)
# =====================================================================
class AdaAngularMarginHead(nn.Module):
    def __init__(self, in_features, num_classes, s=32.0, m=0.30):
        super(AdaAngularMarginHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, labels=None):
        normalized_features = F.normalize(embedding, p=2, dim=1)
        normalized_weights = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(normalized_features, normalized_weights)

        # During evaluation (labels=None), we bypass margin alteration entirely
        if labels is None:
            return cosine * self.s

        safe_cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        th = math.cos(math.pi - self.m)
        mm = math.sin(math.pi - self.m) * self.m

        sine = torch.sqrt(1.0 - torch.pow(safe_cosine, 2))
        phi = safe_cosine * cos_m - sine * sin_m
        phi = torch.where(safe_cosine > th, phi, safe_cosine - mm)

        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        marginal_logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return marginal_logits * self.s, cosine * self.s


class MedicalContrastiveNet(nn.Module):
    def __init__(self, base_weights='yolov8n-cls.pt', num_classes=4, embedding_dim=256):
        super(MedicalContrastiveNet, self).__init__()
        yolo_internal = YOLO(base_weights).model
        self.backbone = nn.Sequential(*list(yolo_internal.model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            backbone_out_dim = torch.flatten(self.pool(self.backbone(dummy_input)), start_dim=1).shape[1]

        self.projection_head = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU()
        )
        self.margin_head = AdaAngularMarginHead(in_features=embedding_dim, num_classes=num_classes, s=32.0, m=0.0)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        flat_features = torch.flatten(self.pool(features), start_dim=1)
        embeddings = self.projection_head(flat_features)

        if labels is not None:
            marginal_logits, raw_logits = self.margin_head(embeddings, labels)
            return marginal_logits, raw_logits, embeddings

        raw_logits = self.margin_head(embeddings)
        return raw_logits


# =====================================================================
# EVALUATION & GENERATE REPORT RUNTIME
# =====================================================================
def run_evaluation_report():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_root = r"E:\Datasets\Medical\Brisc\brisc2025\classification_task"
    weights_path = "brisc_nano_projection.pth"

    # Validation Transform Setup
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'val'), val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Class mapping recovery
    class_names = val_dataset.classes  # list ordered by index e.g. ['glioma', 'meningioma'...]
    num_classes = len(class_names)

    # Instantiate model and load state dictionary safely
    model = MedicalContrastiveNet(base_weights='yolov8n-cls.pt', num_classes=num_classes, embedding_dim=256)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing custom checkpoint file at: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Track classification specifics
    class_totals = defaultdict(int)
    class_corrects = defaultdict(int)
    misclassifications = defaultdict(lambda: defaultdict(int))

    total_images = 0
    total_correct = 0

    print("Running geometric feature evaluations across validation cohort...")

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Labels are omitted here -> returns raw cosine-scaled logits with no margin penalty
            raw_logits = model(inputs)
            _, preds = torch.max(raw_logits, 1)

            for true_label, pred_label in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                true_name = class_names[true_label].upper()
                pred_name = class_names[pred_label].upper()

                class_totals[true_name] += 1
                total_images += 1

                if true_label == pred_label:
                    class_corrects[true_name] += 1
                    total_correct += 1
                else:
                    misclassifications[true_name][pred_name] += 1

    # Print out report mapping perfectly to original formats
    print("\n==================================================")
    print("      PER-CLASS PERFORMANCE & ACCURACY REPORT      ")
    print("==================================================")

    for c_name in sorted(class_names):
        c_upper = c_name.upper()
        total = class_totals[c_upper]
        correct = class_corrects[c_upper]
        accuracy = (correct / total * 100) if total > 0 else 0.0

        print(f"\n📁 Class: {c_upper}")
        print(f"   ↳ Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

        if len(misclassifications[c_upper]) > 0:
            print("   ↳ Most Common Misclassifications:")
            # Sort errors descending to bubble up the worst systemic blindspots
            sorted_errors = sorted(misclassifications[c_upper].items(), key=lambda x: x[1], reverse=True)
            for mis_class, count in sorted_errors:
                print(f"      ▪ Confused as '{mis_class.lower()}': {count} times")
        else:
            print("   ↳ Most Common Misclassifications: None 🎉 (Perfect Classification!)")

    global_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0.0

    print("\n==================================================")
    print("                GLOBAL SUMMARY                    ")
    print("==================================================")
    print(f"Total Test Images Run : {total_images}")
    print(f"Total Correct Passes  : {total_correct}")
    print(f"Overall Test Accuracy : {global_accuracy:.2f}%")
    print("==================================================\n")


if __name__ == "__main__":
    run_evaluation_report()