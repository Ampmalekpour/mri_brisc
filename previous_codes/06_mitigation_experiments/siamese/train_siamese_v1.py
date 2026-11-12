# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Dual-Branch Siamese Network/train.py
# Purpose       :
#   Dual-branch ResNet50: Stage 1 InfoNCE alignment, Stage 2 SupCon, Stage 3 linear probe (original + zero-out inputs).
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# =====================================================================
# 1. DUAL-BRANCH DATASET LOADER
# =====================================================================
class BRISCDualDataset(Dataset):
    """
    Pairs the original image with its zero-masked counterpart to extract
    the isolated tumor on the fly: Tumor = Original - MaskZero
    """

    def __init__(self, orig_dir, mask_zero_dir, transform=None):
        self.orig_dir = orig_dir
        self.mask_zero_dir = mask_zero_dir
        self.transform = transform

        # Class mapping alignment
        self.classes = sorted([d for d in os.listdir(orig_dir) if os.path.isdir(os.path.join(orig_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        for cls in self.classes:
            orig_cls_dir = os.path.join(orig_dir, cls)
            img_names = [f for f in os.listdir(orig_cls_dir) if f.lower().endswith(valid_extensions)]

            for name in img_names:
                orig_path = os.path.join(orig_dir, cls, name)
                mask_zero_path = os.path.join(mask_zero_dir, cls, name)

                if os.path.exists(mask_zero_path):
                    self.samples.append((orig_path, mask_zero_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        orig_path, mask_zero_path, label = self.samples[idx]

        # Load as RGB images
        img_orig = Image.open(orig_path).convert('RGB')
        img_zero = Image.open(mask_zero_path).convert('RGB')

        if self.transform:
            img_orig = self.transform(img_orig)
            img_zero = self.transform(img_zero)

        # Mathematical derivation of isolated tumor: Original - MaskedBackground
        # Clamp to 0 to prevent negative underflow from minor compression artifacts
        img_tumor = torch.clamp(img_orig - img_zero, min=0.0)

        return img_orig, img_tumor, label


# =====================================================================
# 2. MODEL ARCHITECTURE
# =====================================================================
class DualBranchNet(nn.Module):
    def __init__(self, num_classes=4, feat_dim=128):
        super().__init__()
        # Shared Feature Extractor Backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Outputs [B, 2048, 1, 1]

        # Stage 2: Supervised Contrastive Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )

        # Stage 3: Final Linear Probe Classifier Head
        self.classifier_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward_encoder(self, x):
        features = self.encoder(x)
        return torch.flatten(features, 1)

    def forward_project(self, x):
        feat = self.forward_encoder(x)
        return F.normalize(self.projection_head(feat), dim=1)

    def forward_classify(self, x):
        feat = self.forward_encoder(x)
        return self.classifier_head(feat)


# =====================================================================
# 3. ADVANCED LOSS FUNCTIONS
# =====================================================================
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss for Class Clustering"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Mask out self-contrast targets
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Compute log-likelihood
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        return loss


def info_nce_loss(z_g, z_l, temperature=0.1):
    """Self-Supervised Local-to-Global Alignment Loss"""
    batch_size = z_g.shape[0]
    device = z_g.device

    # Normalize features
    z_g = F.normalize(z_g, dim=1)
    z_l = F.normalize(z_l, dim=1)

    # Cosine similarity between all global configurations and local target crops
    logits = torch.matmul(z_g, z_l.T) / temperature
    labels = torch.arange(batch_size).long().to(device)

    loss = F.cross_entropy(logits, labels)
    return loss


# =====================================================================
# 4. PROGRESSIVE MULTI-STAGE TRAINING ENGINE
# =====================================================================
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target Device: {device}")

    # --- Data Adjustments ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    orig_train_dir = r"D:\Datasets\Medical\brisc2025\classification_task\train"
    mask_zero_dir = r"D:\Datasets\Medical\brisc2025\classification_task\train_mask_zero"

    dataset = BRISCDualDataset(orig_train_dir, mask_zero_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=2)

    model = DualBranchNet(num_classes=4).to(device)
    sup_con_criterion = SupConLoss(temperature=0.07)

    # -----------------------------------------------------------------
    # STAGE 1: Anatomical Grounding (Local-Global Structural Alignment)
    # -----------------------------------------------------------------
    print("\n--- Starting Stage 1: Anatomical Grounding ---")
    model.encoder.requires_grad_(True)
    model.projection_head.requires_grad_(False)
    model.classifier_head.requires_grad_(False)

    optimizer_s1 = torch.optim.AdamW(model.encoder.parameters(), lr=1e-4)

    for epoch in range(5):  # Substituted brief epoch counts for pipeline validation
        model.train()
        total_loss = 0
        for img_orig, img_tumor, _ in dataloader:
            img_orig, img_tumor = img_orig.to(device), img_tumor.to(device)

            optimizer_s1.zero_grad()
            z_g = model.forward_encoder(img_orig)
            z_l = model.forward_encoder(img_tumor)

            loss = info_nce_loss(z_g, z_l)
            loss.backward()
            optimizer_s1.step()
            total_loss += loss.item()
        print(f"Stage 1 | Epoch {epoch + 1} | Alignment Loss: {total_loss / len(dataloader):.4f}")

    # -----------------------------------------------------------------
    # STAGE 2: Semantic Topography (Supervised Contrastive Clustering)
    # -----------------------------------------------------------------
    print("\n--- Starting Stage 2: Semantic Topography ---")
    model.encoder.requires_grad_(True)
    model.projection_head.requires_grad_(True)
    model.classifier_head.requires_grad_(False)

    optimizer_s2 = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.projection_head.parameters()),
        lr=5e-5
    )

    for epoch in range(5):
        model.train()
        total_loss = 0
        for img_orig, _, labels in dataloader:
            img_orig, labels = img_orig.to(device), labels.to(device)

            optimizer_s2.zero_grad()
            proj_features = model.forward_project(img_orig)

            loss = sup_con_criterion(proj_features, labels)
            loss.backward()
            optimizer_s2.step()
            total_loss += loss.item()
        print(f"Stage 2 | Epoch {epoch + 1} | SupCon Clustering Loss: {total_loss / len(dataloader):.4f}")

    # -----------------------------------------------------------------
    # STAGE 3: Linear Probe Calibration (Classification Head Optimization)
    # -----------------------------------------------------------------
    print("\n--- Starting Stage 3: Linear Probe Calibration ---")
    # Freeze encoder completely to guard structured embeddings
    model.encoder.requires_grad_(False)
    model.projection_head.requires_grad_(False)
    model.classifier_head.requires_grad_(True)

    optimizer_s3 = torch.optim.AdamW(model.classifier_head.parameters(), lr=1e-3)
    # Integrating cross-entropy label smoothing to combat boundary confusion
    ce_loss_smoothed = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for img_orig, _, labels in dataloader:
            img_orig, labels = img_orig.to(device), labels.to(device)

            optimizer_s3.zero_grad()
            outputs = model.forward_classify(img_orig)

            loss = ce_loss_smoothed(outputs, labels)
            loss.backward()
            optimizer_s3.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(
            f"Stage 3 | Epoch {epoch + 1} | Classification Loss: {total_loss / len(dataloader):.4f} | Accuracy: {acc:.2f}%")

    # Save finalized network state weights
    torch.save(model.state_dict(), "brisc_dual_branch_model.pth")
    print("\nTraining workflow fully executed. Core weights archived.")


if __name__ == '__main__':
    train_pipeline()