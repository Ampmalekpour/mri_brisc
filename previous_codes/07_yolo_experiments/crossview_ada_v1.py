# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : Yolo/v1.py
# Purpose       :
#   Cross-view adversarial (gradient reversal) model, generic backbone.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import re
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ==========================================
# 1. GRADIENT REVERSAL LAYER FOR ADA IMPLEMENTATION
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
    Discriminates between Axial, Coronal, and Sagittal planes
    to force feature extractor into view-invariance via GRL.
    """

    def __init__(self, input_dim, num_views=3):
        super(ViewDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_views)

    def forward(self, x, alpha):
        x = GradientReversalFn.apply(x, alpha)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


# ==========================================
# 2. BRISC MULTI-VIEW CONTRASTIVE DATASET
# ==========================================
class BRISCCrossViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.view_to_idx = {'ax': 0, 'co': 1, 'sa': 2}

        # Structure pairs based on patient index tracking
        self.samples = []
        self._build_cross_view_pairs()

    def _build_cross_view_pairs(self):
        # Pattern: brisc2025_[split]_[index]_[tumor]_[view]_t1.jpg
        pattern = re.compile(r'brisc2025_(train|test)_(\d+)_([a-z]+)_([a-z]+)_t1\.jpg')
        patient_buckets = {}

        # Scan folder layout
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for img_path in glob.glob(os.path.join(cls_dir, '*.jpg')):
                filename = os.path.basename(img_path)
                match = pattern.match(filename)
                if match:
                    _, pid, _, view = match.groups()
                    if view in self.view_to_idx:
                        if pid not in patient_buckets:
                            patient_buckets[pid] = []
                        patient_buckets[pid].append((img_path, self.class_to_idx[cls], self.view_to_idx[view]))

        # Flatten into pairs for contrastive learning where available; fallback safely if single views exist
        for pid, views in patient_buckets.items():
            if len(views) > 1:
                # Pair consecutive views of the same subject as primary targets
                for i in range(len(views)):
                    for j in range(i + 1, len(views)):
                        self.samples.append((views[i], views[j]))
            else:
                # Fallback self-pairing if a patient has only a unique view slice
                self.samples.append((views[0], views[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v1_info, v2_info = self.samples[idx]

        img1 = Image.open(v1_info[0]).convert('RGB')
        img2 = Image.open(v2_info[0]).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            'img1': img1, 'label1': v1_info[1], 'view1': v1_info[2],
            'img2': img2, 'label2': v2_info[1], 'view2': v2_info[2]
        }


# ==========================================
# 3. JOINT FRAMEWORK SYSTEM ARCHITECTURE
# ==========================================
class CrossViewADAModel(nn.Module):
    def __init__(self, num_classes=4, num_views=3):
        super(CrossViewADAModel, self).__init__()
        # Backbone feature extractor
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_dim = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Task Heads
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.view_discriminator = ViewDiscriminator(self.feature_dim, num_views)

    def forward(self, x, alpha=1.0):
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        logits = self.classifier(feat)
        proj = F.normalize(self.projection_head(feat), p=2, dim=1)
        view_logits = self.view_discriminator(feat, alpha)
        return logits, proj, view_logits


# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================
def nt_xent_loss(proj1, proj2, temperature=0.1):
    """Normalized Temperature-scaled Cross Entropy Loss for cross-view pairs."""
    batch_size = proj1.shape[0]
    out = torch.cat([proj1, proj2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

    mask = list(range(batch_size * 2))
    for i in range(batch_size):
        mask[i] = i + batch_size
        mask[i + batch_size] = i

    positives = sim_matrix[range(batch_size * 2), mask]
    negatives = sim_matrix.sum(dim=1) - torch.exp(torch.tensor(1.0 / temperature))
    return -torch.log(positives / negatives).mean()


# ==========================================
# 5. EXECUTION & EXECUTION PIPELINE
# ==========================================
def run_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on target device: {device}")

    # Paths configured strictly to dataset parameters
    train_dir = r"E:\Datasets\Medical\Brisc\brisc2025\classification_task\train"
    val_dir = r"E:\Datasets\Medical\Brisc\brisc2025\classification_task\val"

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = BRISCCrossViewDataset(train_dir, transform=data_transforms)
    val_dataset = BRISCCrossViewDataset(val_dir, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = CrossViewADAModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion_cls = nn.CrossEntropyLoss()

    # Hyperparameter weighting configurations
    alpha_contrastive = 0.5
    beta_ada = 0.3

    for epoch in range(1, 6):  # Sample tracking loop
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            img1, img2 = batch['img1'].to(device), batch['img2'].to(device)
            lbl1, lbl2 = batch['label1'].to(device), batch['label2'].to(device)
            v1, v2 = batch['view1'].to(device), batch['view2'].to(device)

            optimizer.zero_grad()

            # Dynamic calculation for adaptation factor step scale
            p = float(batch_idx + epoch * len(train_loader)) / (5 * len(train_loader))
            grl_alpha = 2. / (1. + math.exp(-10 * p)) - 1 if 'math' in globals() else 0.5

            # Process forward execution branches
            logits1, proj1, view_logits1 = model(img1, alpha=grl_alpha)
            logits2, proj2, view_logits2 = model(img2, alpha=grl_alpha)

            # Compute loss elements
            loss_cls = (criterion_cls(logits1, lbl1) + criterion_cls(logits2, lbl2)) / 2
            loss_contrastive = nt_xent_loss(proj1, proj2)
            loss_ada = (criterion_cls(view_logits1, v1) + criterion_cls(view_logits2, v2)) / 2

            loss = loss_cls + (alpha_contrastive * loss_contrastive) + (beta_ada * loss_ada)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} complete. Average Loss tracking metrics: {total_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    import math

    run_pipeline()