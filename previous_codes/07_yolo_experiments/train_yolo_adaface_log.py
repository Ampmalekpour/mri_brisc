# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/train_yolo_ada_log.py
# Purpose       :
#   YOLO + AdaFace-style head with per-epoch logging (plane-aware tracking dataset).
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


# ==========================================
# 1. THE PLANE-AWARE DATA TRACKER
# ==========================================
class BRISCTrackingDataset(datasets.ImageFolder):
    """
    Custom Dataset that overrides standard ImageFolder so we can
    extract and pass the exact image filename down into the training loop.
    """

    def __getitem__(self, index):
        img, label = super(BRISCTrackingDataset, self).__getitem__(index)
        path, _ = self.samples[index]
        filename = os.path.basename(path)
        return img, label, filename


def parse_mri_plane(filename):
    """Reads the filename string to find the radiological view."""
    if "_ax_" in filename: return "Axial"
    if "_co_" in filename: return "Coronal"
    if "_sa_" in filename: return "Sagittal"
    return "Unknown"


# ==========================================
# 2. THE ADAFACE ADAPTIVE MARGIN HEAD
# ==========================================
class AdaFaceClassificationHead(nn.Module):
    """
    Replaces YOLO's default linear classifier. Uses the embedding norm
    as an indicator of clinical informativeness to scale the margin.
    """

    def __init__(self, embedding_size=512, num_classes=4, m=0.4, h=0.333, s=64.0, t_alpha=0.01):
        super(AdaFaceClassificationHead, self).__init__()
        self.kernel = nn.Parameter(torch.FloatTensor(embedding_size, num_classes))
        nn.init.xavier_uniform_(self.kernel)

        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha

        self.register_buffer('batch_mean', torch.tensor([20.0]))
        self.register_buffer('batch_std', torch.tensor([5.0]))

    def forward(self, embeddings, norms, labels):
        kernel_norm = F.normalize(self.kernel, dim=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if self.training:
            with torch.no_grad():
                mean_now = norms.mean()
                std_now = norms.std()
                self.batch_mean = (1 - self.t_alpha) * self.batch_mean + self.t_alpha * mean_now
                self.batch_std = (1 - self.t_alpha) * self.batch_std + self.t_alpha * std_now

        sigma = 1.0 / self.batch_std
        shifted_norm = (norms - self.batch_mean) * sigma
        image_quality = torch.tanh(shifted_norm)

        g_m = -self.m * image_quality
        g_e = self.h * image_quality - self.h

        theta = torch.acos(cos_theta)
        safe_theta = torch.where(theta + g_m > 0, theta + g_m, theta)
        safe_theta = torch.where(safe_theta < math.pi, safe_theta, theta)

        target_logit = torch.cos(safe_theta) + g_e

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = (one_hot * target_logit) + ((1.0 - one_hot) * cos_theta)
        output *= self.s

        return output


# ==========================================
# 3. TRAINING LOOP WITH LIVE PROXY LOGGING
# ==========================================
def run_adaptive_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_root = r"E:\Datasets\Medical\Brisc\brisc2025\classification_task"

    # Extract the underlying PyTorch backbone out of your YOLOv8-cls weight file
    yolo_base = YOLO('yolov8n-cls.pt')
    backbone = yolo_base.model.model

    # FIX: Append an explicit Global Average Pooling layer to the sliced backbone.
    # This collapses the spatial 7x7 grid into a clean channel-representation.
    feature_extractor = nn.Sequential(
        *list(backbone.children())[:-1],
        nn.AdaptiveAvgPool2d((1, 1))
    ).to(device)

    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')

    num_classes = len(os.listdir(train_dir))

    # FIX: YOLOv8 Nano outputs exactly 256 channels at this terminal stage.
    # We change embedding_size from 512 to 256.
    ada_head = AdaFaceClassificationHead(embedding_size=256, num_classes=num_classes).to(device)
    # Replicates your imgsz=224 transformations
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = BRISCTrackingDataset(train_dir, transform=transform_pipeline)
    val_dataset = BRISCTrackingDataset(val_dir, transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Match your baseline learning rate and optimizer choice
    optimizer = torch.optim.AdamW(list(feature_extractor.parameters()) + list(ada_head.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting BRISC 2025 Ada-YOLO Training Pipeline ---\n")

    for epoch in range(30):
        print(f"=== Epoch {epoch + 1}/30 ===")

        feature_extractor.train()
        ada_head.train()
        running_train_loss = 0.0

        for batch_idx, (images, labels, filenames) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass through YOLO feature backbone
            embeddings = torch.flatten(feature_extractor(images), start_dim=1)
            norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)

            # --- THE LIVE PRINT MECHANISM ---
            # Print individual proxy scores for the first batch of every single epoch
            if batch_idx == 0:
                print("\n  [Live Training Proxy Tracking Sample]")
                for i in range(min(5, len(filenames))):
                    plane = parse_mri_plane(filenames[i])
                    print(f"    File: {filenames[i]} | Plane: {plane:8s} | Proxy Value (Norm): {norms[i].item():.4f}")
                print("")

            normalized_embeddings = embeddings / norms
            logits = ada_head(normalized_embeddings, norms, labels)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        # --- VALIDATION LOOP ---
        feature_extractor.eval()
        ada_head.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, filenames in val_loader:
                images, labels = images.to(device), labels.to(device)

                embeddings = torch.flatten(feature_extractor(images), start_dim=1)
                norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)

                normalized_embeddings = embeddings / norms
                kernel_norm = nn.functional.normalize(ada_head.kernel, dim=0)
                logits = torch.mm(normalized_embeddings, kernel_norm) * ada_head.s

                loss = criterion(logits, labels)
                running_val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(
            f"  Summary -> Train Loss: {running_train_loss / len(train_dataset):.4f} | Val Loss: {running_val_loss / len(val_dataset):.4f} | Val Accuracy: {(val_correct / val_total) * 100:.2f}%\n")


if __name__ == "__main__":
    run_adaptive_training()