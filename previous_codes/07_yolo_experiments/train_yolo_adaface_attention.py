# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/train_yolo_ada_log_2.py
# Purpose       :
#   AdaFace head + tumor attention head variant.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ==========================================
# 1. BRISC 2025 DATASET & PARSING LAYER
# ==========================================
class BRISCDataset(Dataset):
    """
    Parses filenames like: brisc2025_train_00470_gl_co_t1.jpg
    Classes: gl=0 (Glioma), me=1 (Meningioma), pi=2 (Pituitary), no=3 (Normal)
    Planes: co=Coronal, sa=Sagittal, ax=Axial
    """

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        if not os.path.exists(image_dir):
            # Fallback for testing/empty directories so code runs out of the box
            self.filenames = [
                f"brisc2025_train_{i:05d}_{random.choice(['gl', 'me', 'pi', 'no'])}_{random.choice(['co', 'sa', 'ax'])}_t1.jpg"
                for i in range(100)]
        else:
            self.filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        self.class_map = {'gl': 0, 'me': 1, 'pi': 2, 'no': 3}
        self.plane_map = {'co': 'Coronal', 'sa': 'Sagittal', 'ax': 'Axial'}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # Parse labels and metadata from the string structure
        parts = filename.split('_')
        class_str = parts[3]  # 'gl', 'me', 'pi', 'no'
        plane_str = parts[4]  # 'co', 'sa', 'ax'

        label = self.class_map.get(class_str, 3)
        plane = self.plane_map.get(plane_str, 'Unknown')

        img_path = os.path.join(self.image_dir, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.new('RGB', (224, 224),
                            color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        if self.transform:
            img = self.transform(img)

        return img, label, filename, plane


# ==========================================
# 2. ADAFACE & ATTENTION HEAD ARCHITECTURE
# ==========================================
class AdaFace(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=4, m=0.4, h=0.333, s=64.0):
        super(AdaFace, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)

    def forward(self, embeddings, target=None):
        norms = torch.norm(embeddings, 2, 1, keepdim=True)

        embeddings_normalized = self.batch_norm(embeddings)
        embeddings_normalized = F.normalize(embeddings_normalized, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings_normalized, weight_normalized)
        cosine = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)

        if not self.training or target is None:
            return cosine * self.s, norms.detach()

        safe_norms = torch.clamp(norms, min=0.001, max=100)
        norm_batch_avg = safe_norms.mean().detach()

        g = (safe_norms - norm_batch_avg) / (safe_norms.std().detach() + self.eps)
        g = torch.clamp(g, -1.0, 1.0)

        margin_m = self.m * g
        margin_h = self.h * g

        theta = cosine.acos()
        theta_m = torch.clamp(theta + margin_m, min=self.eps, max=math.pi - self.eps)
        cosine_m = theta_m.cos() - margin_h

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1.0)

        output = (one_hot * cosine_m) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output, safe_norms.detach()


class TumorAttentionHead(nn.Module):
    def __init__(self, in_channels=256, embedding_dim=512, num_classes=4, dropout_p=0.4):
        super(TumorAttentionHead, self).__init__()

        # Spatial Attention layer calibrated to YOLOv8n channel width (256)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 1, kernel_size=1, bias=False)
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(in_channels, embedding_dim)
        self.adaface = AdaFace(embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, x, target=None):
        B, C, H, W = x.shape
        attn_map = self.spatial_attention(x)
        attn_map_flat = attn_map.view(B, 1, -1)
        attn_weights = F.softmax(attn_map_flat, dim=2)
        attn_weights = attn_weights.view(B, 1, H, W)

        features = (x * attn_weights).sum(dim=(2, 3))
        features = self.dropout(features)
        embeddings = self.fc(features)

        return self.adaface(embeddings, target)


# ==========================================
# 3. BACKBONE COMPATIBILITY LAYER
# ==========================================
class YOLOFeatureExtractor(nn.Module):
    """
    Wraps YOLOv8n backbone to output raw [B, 256, 7, 7] feature maps.
    """

    def __init__(self, model_path='yolov8n.pt'):
        super().__init__()
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path).model
            self.has_yolo = True
        except ImportError:
            self.has_yolo = False
            # Fallback mimicking the exact dimensions of the Nano backbone
            self.mock_backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((7, 7))
            )

    def forward(self, x):
        if self.has_yolo:
            # Slices through layer 9 (SPPF) to extract features right before global pooling
            return self.yolo_model.model[:10](x)
        else:
            return self.mock_backbone(x)


# ==========================================
# 4. MAIN TRAINING & VALIDATION PIPELINE
# ==========================================
def run_pipeline():
    print("--- Starting BRISC 2025 Ada-YOLO Training Pipeline (Optimized) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = r"D:\projects\EyePass\AdaFace\Train_Yolo_Tumor\data\train"
    val_dir = r"D:\projects\EyePass\AdaFace\Train_Yolo_Tumor\data\val"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BRISCDataset(train_dir, transform=transform)
    val_dataset = BRISCDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    backbone = YOLOFeatureExtractor(model_path='yolov8n.pt').to(device)
    custom_head = TumorAttentionHead(in_channels=256, num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()

    # --- DIFFERENTIAL LEARNING RATES ---
    # Give the raw attention head room to sprint, keep the backbone on a tight leash
    optimizer = torch.optim.AdamW([
        {'params': backbone.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2},  # Cold backbone
        {'params': custom_head.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}  # Hot classification head
    ])

    # Dynamic learning rate decay to stabilize AdaFace margins over time
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    total_epochs = 50  # Bumped up to give margins room to stretch

    for epoch in range(1, total_epochs + 1):
        print(
            f"\n=== Epoch {epoch}/{total_epochs} [LR Backbone: {optimizer.param_groups[0]['lr']:.7f} | LR Head: {optimizer.param_groups[1]['lr']:.5f}] ===")

        # --- Training Loop ---
        backbone.train()
        custom_head.train()
        running_train_loss = 0.0
        live_samples_logged = 0

        for imgs, labels, filenames, planes in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            features = backbone(imgs)
            logits, norms = custom_head(features, labels)

            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping to prevent s=64 from blowing up the layers
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(custom_head.parameters()), max_norm=5.0)

            optimizer.step()
            running_train_loss += loss.item() * imgs.size(0)

            if live_samples_logged == 0:
                print("\n  [Live Training Proxy Tracking Sample]")
                log_limit = min(5, len(filenames))
                for idx in range(log_limit):
                    print(
                        f"    File: {filenames[idx]} | Plane: {planes[idx]:<8} | Proxy Value (Norm): {norms[idx].item():.4f}")
                print("")
                live_samples_logged = 1

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        scheduler.step()

        # --- Validation Loop ---
        backbone.eval()
        custom_head.eval()
        running_val_loss = 0.0
        correct_preds = 0

        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                features = backbone(imgs)
                logits, _ = custom_head(features, labels)

                loss = criterion(logits, labels)
                running_val_loss += loss.item() * imgs.size(0)

                _, preds = torch.max(logits, 1)
                correct_preds += torch.sum(preds == labels.data)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_preds.double() / len(val_loader.dataset)) * 100

        print(
            f"  Summary -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {epoch_val_acc:.2f}%")

if __name__ == "__main__":
    run_pipeline()