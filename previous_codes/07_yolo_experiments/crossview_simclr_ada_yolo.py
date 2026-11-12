# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/v2.py
# Purpose       :
#   Cross-view adversarial + SimCLR augmentation variant.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import re
import glob
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import YOLOv8 internal architecture components safely
from ultralytics import YOLO


# ==========================================
# 1. GRADIENT REVERSAL LAYER FOR ADA
# ==========================================
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class ViewDiscriminator(nn.Module):
    """
    Discriminates between Axial (ax), Coronal (co), and Sagittal (sa) planes.
    Includes dropout to soften its updates and give the backbone a chance to adapt.
    """

    def __init__(self, input_dim, num_views=3):
        super(ViewDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)  # Prevents the discriminator from instantly overpowering the trunk
        self.fc2 = nn.Linear(256, num_views)

    def forward(self, x, alpha):
        x = GradientReversalFn.apply(x, alpha)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        return self.fc2(x)


# ==========================================
# 2. BRISC DUAL-AUGMENTATION DATASET
# ==========================================
class BRISCSimCLRDataset(Dataset):
    def __init__(self, root_dir, base_transform=None, contrastive_transform=None):
        self.root_dir = root_dir
        self.base_transform = base_transform
        self.contrastive_transform = contrastive_transform

        self.classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.view_to_idx = {'ax': 0, 'co': 1, 'sa': 2}

        self.samples = []
        self._build_dataset()

    def _build_dataset(self):
        # Flexible regex pattern targeting: brisc2025_[split]_[index]_[tumor]_[view]_t1.jpg
        pattern = re.compile(r'brisc2025_(train|test|val)_(\d+)_([a-zA-Z0-9_]+)_([a-z]+)_t1\.jpg')

        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for img_path in glob.glob(os.path.join(cls_dir, '*.jpg')):
                filename = os.path.basename(img_path)
                match = pattern.match(filename)
                if match:
                    _, _, _, view = match.groups()
                    if view in self.view_to_idx:
                        self.samples.append({
                            'path': img_path,
                            'label': self.class_to_idx[cls],
                            'view': self.view_to_idx[view]
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')

        # Generate two distinct augmented views from the same image matrix
        if self.contrastive_transform:
            img1 = self.contrastive_transform(img)
            img2 = self.contrastive_transform(img)
        else:
            img1 = self.base_transform(img)
            img2 = self.base_transform(img)

        return {
            'img1': img1,
            'img2': img2,
            'label': sample['label'],
            'view': sample['view']
        }


# ==========================================
# 3. YOLO-BASED JOINT FRAMEWORK PIPELINE
# ==========================================
class YOLOCrossViewADAModel(nn.Module):
    def __init__(self, num_classes=4, num_views=3):
        super(YOLOCrossViewADAModel, self).__init__()

        # Load pretrained YOLOv8 classification trunk wrapper
        base_yolo = YOLO('yolov8n-cls.pt')
        self.yolo_sequential = base_yolo.model.model

        # YOLOv8 Nano scales explicitly to 256 channels at stage 8
        self.feature_dim = 256

        # Custom heads built on top of the 256-channel embedding space
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.view_discriminator = ViewDiscriminator(self.feature_dim, num_views)

    def extract_backbone_features(self, x):
        """Passes input through YOLO up to the global spatial pooling block."""
        for i, layer in enumerate(self.yolo_sequential):
            if i < 8:
                x = layer(x)
            elif i == 8:
                if hasattr(layer, 'pool'):
                    x = layer.pool(x)
                else:
                    x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                break
        return x

    def forward(self, x, alpha=1.0):
        feat = self.extract_backbone_features(x)
        logits = self.classifier(feat)
        proj = F.normalize(self.projection_head(feat), p=2, dim=1)
        view_logits = self.view_discriminator(feat, alpha)
        return logits, proj, view_logits


# ==========================================
# 4. NT-XENT LOSS (CONTRASTIVE)
# ==========================================
def nt_xent_loss(proj1, proj2, temperature=0.1):
    batch_size = proj1.shape[0]
    out = torch.cat([proj1, proj2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

    mask = list(range(batch_size * 2))
    for i in range(batch_size):
        mask[i] = i + batch_size
        mask[i + batch_size] = i

    positives = sim_matrix[range(batch_size * 2), mask]
    negatives = sim_matrix.sum(dim=1) - torch.exp(torch.tensor(1.0 / temperature, device=proj1.device))
    return -torch.log(positives / negatives).mean()


# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
def run_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing training pipeline using YOLOv8 Backbone on: {device}")

    train_dir = r"E:\Datasets\Medical\Brisc\brisc2025\classification_task\train"
    val_dir = r"E:\Datasets\Medical\Brisc\brisc2025\classification_task\val"

    # Strict spatial verification transform for testing/validation
    base_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dynamic image transformations for contrastive pair creation
    contrastive_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Emulates scanner variance
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = BRISCSimCLRDataset(train_dir, contrastive_transform=contrastive_transforms)
    val_dataset = BRISCSimCLRDataset(val_dir, base_transform=base_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = YOLOCrossViewADAModel(num_classes=4, num_views=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion_cls = nn.CrossEntropyLoss()

    # Regularization balancing terms
    alpha_contrastive = 0.5
    beta_ada = 0.3

    for epoch in range(1, 30):
        model.train()
        epoch_cls_loss = 0.0
        epoch_contra_loss = 0.0
        epoch_ada_loss = 0.0
        epoch_total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            img1, img2 = batch['img1'].to(device), batch['img2'].to(device)
            lbl = batch['label'].to(device)
            view = batch['view'].to(device)

            optimizer.zero_grad()

            # Sigmoid activation ramp-up schedule for GRL alpha alpha
            p = float(batch_idx + epoch * len(train_loader)) / (5 * len(train_loader))
            grl_alpha = 2. / (1. + math.exp(-10 * p)) - 1

            # Execute forward operations
            logits1, proj1, view_logits1 = model(img1, alpha=grl_alpha)
            logits2, proj2, view_logits2 = model(img2, alpha=grl_alpha)

            # 1. Classification Target (calculated on both augmentations)
            loss_cls = (criterion_cls(logits1, lbl) + criterion_cls(logits2, lbl)) / 2

            # 2. Self-Supervised InfoNCE Target
            loss_contrastive = nt_xent_loss(proj1, proj2)

            # 3. Plane Domain Invariance Target
            loss_ada = (criterion_cls(view_logits1, view) + criterion_cls(view_logits2, view)) / 2

            # Total Objective Optimization function
            loss = loss_cls + (alpha_contrastive * loss_contrastive) + (beta_ada * loss_ada)

            loss.backward()
            optimizer.step()

            epoch_cls_loss += loss_cls.item()
            epoch_contra_loss += loss_contrastive.item()
            epoch_ada_loss += loss_ada.item()
            epoch_total_loss += loss.item()

        # --- VALIDATION PASS ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                img, lbl = batch['img1'].to(device), batch['label'].to(device)
                logits, _, _ = model(img)

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == lbl).sum().item()
                val_total += lbl.size(0)

        val_accuracy = (val_correct / val_total) * 100 if val_total > 0 else 0.0

        print(f"\n==== Epoch {epoch} Report ====")
        print(f"  -> Total Combined Loss: {epoch_total_loss / len(train_loader):.4f}")
        print(f"  └─ Breakdown: Cls: {epoch_cls_loss / len(train_loader):.4f} | "
              f"Contra: {epoch_contra_loss / len(train_loader):.4f} | "
              f"ADA: {epoch_ada_loss / len(train_loader):.4f}")
        print(f"  -> Val Tumor Accuracy:  {val_accuracy:.2f}%")


if __name__ == "__main__":
    run_pipeline()