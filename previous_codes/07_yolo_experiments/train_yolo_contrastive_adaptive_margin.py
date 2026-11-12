# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/train_yolo_cls_contrastive_ada.py
# Purpose       :
#   YOLO backbone + contrastive embedding + adaptive angular-margin head.
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
from ultralytics import YOLO


# =====================================================================
# 1. NUMERICALLY STABLE ANGULAR MARGIN HEAD
# =====================================================================
class AdaAngularMarginHead(nn.Module):
    """
    Implements an Additive Angular Margin classification head.
    Includes explicit numerical stability safeguards to prevent NaN underflows.
    """

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
        raw_logits = cosine * self.s

        if labels is None:
            return raw_logits

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
        marginal_logits *= self.s

        return marginal_logits, raw_logits


# =====================================================================
# 2. NANO ARCHITECTURE WITH DEDICATED PROJECTION HEAD
# =====================================================================
class MedicalContrastiveNet(nn.Module):
    def __init__(self, base_weights='yolov8n-cls.pt', num_classes=4, embedding_dim=256):
        super(MedicalContrastiveNet, self).__init__()
        yolo_internal = YOLO(base_weights).model

        # Isolate the Nano convolutional feature extractor
        self.backbone = nn.Sequential(*list(yolo_internal.model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Dynamic dimension mapping
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            dummy_pooled = self.pool(dummy_features)
            backbone_out_dim = torch.flatten(dummy_pooled, start_dim=1).shape[1]

        # THE BRIDGE: Maps Nano's raw feature map into a dedicated angular embedding space
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU()
        )

        self.margin_head = AdaAngularMarginHead(
            in_features=embedding_dim,
            num_classes=num_classes,
            s=32.0,
            m=0.0
        )

    def forward(self, x, labels=None):
        features = self.backbone(x)
        pooled_features = self.pool(features)
        flat_features = torch.flatten(pooled_features, start_dim=1)

        # Pass features through the projection adapter
        embeddings = self.projection_head(flat_features)

        if labels is not None:
            marginal_logits, raw_logits = self.margin_head(embeddings, labels)
            return marginal_logits, raw_logits, embeddings

        raw_logits = self.margin_head(embeddings)
        return raw_logits


# =====================================================================
# 3. RUNTIME TRAINING ENGINE
# =====================================================================
def train_margin_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_root = r"E:\Datasets\Medical\Brisc\brisc2025\classification_task"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # Added mild rotational invariance for medical variations
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_root, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
                   for x in ['train', 'val']}

    # Locked in strictly for YOLOv8n-cls
    model = MedicalContrastiveNet(base_weights='yolov8n-cls.pt', num_classes=4, embedding_dim=256)
    model = model.to(device)

    # Slightly higher learning rate since the projection head provides clean gradient paths
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

    epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Calibrated to 0.30 radians. 0.40 was over-forcing boundaries for a 4-class setup,
    # causing the optimization curve to stall.
    target_margin = 0.30
    warmup_epochs = 5

    print(f"Beginning training on {device} with Nano Backbone, Projection Bridge, and Cosine Decay...")

    for epoch in range(epochs):
        model.train()

        if epoch < warmup_epochs:
            current_margin = (epoch / warmup_epochs) * target_margin
        else:
            current_margin = target_margin

        model.margin_head.m = current_margin

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            marginal_logits, raw_logits, _ = model(inputs, labels)

            loss = criterion(marginal_logits, labels)
            loss.backward()

            # Prevents aggressive angular adjustment spikes from disrupting the backbone weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            _, preds = torch.max(raw_logits, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects.double() / len(image_datasets['train'])

        # --- VALIDATION PHASE ---
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for val_inputs, val_labels in dataloaders['val']:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_logits = model(val_inputs)
                _, val_preds = torch.max(val_logits, 1)
                val_corrects += torch.sum(val_preds == val_labels.data)

        val_acc = val_corrects.double() / len(image_datasets['val'])
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{epochs} [Margin: {current_margin:.2f} | LR: {current_lr:.6f}] -> "
              f"Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), "brisc_nano_projection.pth")
    print("Training complete. Upgraded Nano weights saved successfully.")


if __name__ == "__main__":
    train_margin_model()