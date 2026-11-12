# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Dual-Branch Siamese Network/train_2.py
# Purpose       :
#   Variant of v1 with multi-view InfoNCE (global / tumor / counterfactual views).
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# =====================================================================
# 1. CAUSAL MULTI-VIEW DATASET
# =====================================================================
class BRISCCausalDataset(Dataset):
    """
    Generates 3 parallel views to force spatial and texture invariance:
    1. Global View (Original Image)
    2. Local View (Isolated Tumor)
    3. Causal View (Tumor intact, Background suppressed by 90%)
    """

    def __init__(self, orig_dir, mask_zero_dir, transform=None):
        self.orig_dir = orig_dir
        self.mask_zero_dir = mask_zero_dir
        self.transform = transform

        self.classes = sorted([d for d in os.listdir(orig_dir) if os.path.isdir(os.path.join(orig_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        for cls in self.classes:
            orig_cls_dir = os.path.join(orig_dir, cls)
            for name in [f for f in os.listdir(orig_cls_dir) if f.lower().endswith(valid_exts)]:
                orig_path = os.path.join(orig_dir, cls, name)
                mask_zero_path = os.path.join(mask_zero_dir, cls, name)
                if os.path.exists(mask_zero_path):
                    self.samples.append((orig_path, mask_zero_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        orig_path, mask_zero_path, label = self.samples[idx]

        img_orig = Image.open(orig_path).convert('RGB')
        img_zero = Image.open(mask_zero_path).convert('RGB')  # Background only

        if self.transform:
            img_orig = self.transform(img_orig)
            img_zero = self.transform(img_zero)

        # 1. Extract pure tumor
        img_tumor = torch.clamp(img_orig - img_zero, min=0.0)

        # 2. Causal Augmentation: Attenuate background by 90%, keep tumor bright
        # This breaks spatial-location dependency (the Pituitary shortcut)
        img_causal = (img_zero * 0.1) + img_tumor

        return img_orig, img_tumor, img_causal, label


# =====================================================================
# 2. ARCHITECTURE WITH PROJECTION HEADS
# =====================================================================
class CausalSiameseNet(nn.Module):
    def __init__(self, num_classes=4, feat_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward_encoder(self, x):
        return torch.flatten(self.encoder(x), 1)

    def forward_project(self, x):
        return F.normalize(self.projection_head(self.forward_encoder(x)), dim=1)

    def forward_classify(self, x):
        return self.classifier_head(self.forward_encoder(x))


# =====================================================================
# 3. MULTI-VIEW LOSS FUNCTIONS
# =====================================================================
def multi_view_info_nce(z_global, z_local, z_causal, temperature=0.1):
    """
    Forces the network to align the Global image with both the pure Local Tumor
    and the Background-Suppressed Causal View.
    """
    labels = torch.arange(z_global.shape[0]).long().to(z_global.device)

    # Global to Local Alignment
    logits_local = torch.matmul(z_global, z_local.T) / temperature
    loss_local = F.cross_entropy(logits_local, labels)

    # Global to Causal Alignment (The Shortcut Breaker)
    logits_causal = torch.matmul(z_global, z_causal.T) / temperature
    loss_causal = F.cross_entropy(logits_causal, labels)

    return (loss_local + loss_causal) / 2.0


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.shape[0]).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()


# =====================================================================
# 4. TRAINING ENGINE WITH MIXED PRECISION (AMP)
# =====================================================================
def train_miccai_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing MICCAI Protocol on: {device}")

    # Aggressive Augmentations for Texture/Shape disentanglement
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Slight texture shifts
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    orig_train_dir = r"D:\Datasets\Medical\brisc2025\classification_task\train"
    mask_zero_dir = r"D:\Datasets\Medical\brisc2025\classification_task\train_mask_zero"

    dataset = BRISCCausalDataset(orig_train_dir, mask_zero_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4)

    model = CausalSiameseNet(num_classes=4).to(device)
    sup_con_criterion = SupConLoss(temperature=0.07)
    scaler = GradScaler()  # For Mixed Precision

    # -----------------------------------------------------------------
    # STAGE 1: Causal Invariance (Breaking spatial shortcuts)
    # -----------------------------------------------------------------
    print("\n[Stage 1] Attacking Spatial Biases via Causal Invariance...")
    model.encoder.requires_grad_(True)
    model.projection_head.requires_grad_(False)
    model.classifier_head.requires_grad_(False)

    opt_s1 = torch.optim.AdamW(model.encoder.parameters(), lr=2e-4, weight_decay=1e-4)
    sched_s1 = CosineAnnealingWarmRestarts(opt_s1, T_0=5, T_mult=2)

    for epoch in range(10):  # Recommended: 20-30 epochs for deep convergence
        model.train()
        epoch_loss = 0
        for img_orig, img_tumor, img_causal, _ in dataloader:
            img_orig, img_tumor, img_causal = img_orig.to(device), img_tumor.to(device), img_causal.to(device)

            opt_s1.zero_grad()
            with autocast():
                z_g = model.forward_project(img_orig)
                z_t = model.forward_project(img_tumor)
                z_c = model.forward_project(img_causal)
                loss = multi_view_info_nce(z_g, z_t, z_c)

            scaler.scale(loss).backward()
            scaler.step(opt_s1)
            scaler.update()
            epoch_loss += loss.item()

        sched_s1.step()
        print(f"Stage 1 | Epoch {epoch + 1} | Multi-View Alignment Loss: {epoch_loss / len(dataloader):.4f}")

    # -----------------------------------------------------------------
    # STAGE 2: Topology Clustering
    # -----------------------------------------------------------------
    print("\n[Stage 2] Topology Clustering via SupCon...")
    model.encoder.requires_grad_(True)
    model.projection_head.requires_grad_(True)

    opt_s2 = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.projection_head.parameters()), lr=5e-5,
                               weight_decay=1e-4)

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for img_orig, _, _, labels in dataloader:
            img_orig, labels = img_orig.to(device), labels.to(device)

            opt_s2.zero_grad()
            with autocast():
                features = model.forward_project(img_orig)
                loss = sup_con_criterion(features, labels)

            scaler.scale(loss).backward()
            scaler.step(opt_s2)
            scaler.update()
            epoch_loss += loss.item()
        print(f"Stage 2 | Epoch {epoch + 1} | Clustering Loss: {epoch_loss / len(dataloader):.4f}")

    # -----------------------------------------------------------------
    # STAGE 3: Linear Probe Calibration
    # -----------------------------------------------------------------
    print("\n[Stage 3] Robust Linear Probe Calibration...")
    model.encoder.requires_grad_(False)
    model.projection_head.requires_grad_(False)
    model.classifier_head.requires_grad_(True)

    opt_s3 = torch.optim.AdamW(model.classifier_head.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for img_orig, _, _, labels in dataloader:
            img_orig, labels = img_orig.to(device), labels.to(device)

            opt_s3.zero_grad()
            with autocast():
                outputs = model.forward_classify(img_orig)
                loss = ce_loss(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(opt_s3)
            scaler.update()
            epoch_loss += loss.item()
        print(f"Stage 3 | Epoch {epoch + 1} | Classification Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "brisc_causal_miccai_model.pth")
    print("MICCAI architecture weights successfully archived.")


if __name__ == '__main__':
    train_miccai_pipeline()