# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : train_eff_b0_contrastive.py
# Purpose       :
#   SupCon pre-training of EfficientNet-B0 on (original, foreground-only) pairs. Log: miccai_contrastive_train.txt (stops at epoch 1).
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np


# Re-using your Dual-Stream Logger for consistency in reporting
class Logger(object):
    def __init__(self, filename="miccai_contrastive_train.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==============================================================================
# 1. DEFINING THE PAIRWISE DATASET FOR CONTEXT-INVARIANT VIEWS
# ==============================================================================
class BRISC_ContrastiveDataset(Dataset):
    """
    Constructs diagnostic pairs to force context-invariance.
    Returns: (View_orig, View_foreground_only, Label)
    Both views of the same patient are considered positives in SupCon.
    """

    def __init__(self, root_orig, root_fg, transform=None):
        # We ensure identical sorting to perfectly match orig images with generated fg images
        self.orig_ds = datasets.ImageFolder(root_orig)
        self.fg_ds = datasets.ImageFolder(root_fg)

        self.transform = transform

        # Verify dataset alignment
        if len(self.orig_ds) != len(self.fg_ds):
            raise ValueError(f"Dataset mismatch: Orig={len(self.orig_ds)}, FG={len(self.fg_ds)}")
        print(f"Dataset initialized with {len(self.orig_ds)} perfectly aligned pairs.")

    def __getitem__(self, index):
        img_orig, label_orig = self.orig_ds[index]
        img_fg, label_fg = self.fg_ds[index]

        # Double-check alignment integrity
        assert label_orig == label_fg, f"Label mismatch at index {index}"

        # In contrastive learning,View1 and View2 get independent data augmentations.
        if self.transform is not None:
            view1 = self.transform(img_orig)  # Full scan with spatial context
            view2 = self.transform(img_fg)  # Isolated pathology without context
        else:
            to_tensor = transforms.ToTensor()
            view1 = to_tensor(img_orig)
            view2 = to_tensor(img_fg)

        # We return the label so we can use Supervised Contrastive Loss
        return view1, view2, label_orig

    def __len__(self):
        return len(self.orig_ds)


# ==============================================================================
# 2. DEFINING THE SUPERVISED CONTRASTIVE NETWORK (BACKBONE + PROJECTOR)
# ==============================================================================
class SupConEffNet(nn.Module):
    """
    Backbone encoder (EfficientNet-B0) + Projector Head (MLP).
    Projector head forces dimensionality reduction before similarity computation.
    """

    def __init__(self, projector_dim=128):
        super(SupConEffNet, self).__init__()
        # Load backbone, remove final classifier
        orig_effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(orig_effnet.children())[:-1])  # Features output dim: 1280

        feat_dim = 1280

        # The standard projector head used in contrastive literature (SimCLR/SupCon)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projector_dim)  # Final projected embedding
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)  # Backbone representation
        proj_emb = self.projector(feat)  # Latent projected embedding for loss
        # L2 Normalize the projected embedding - mandatory for cosine similarity
        norm_emb = F.normalize(proj_emb, dim=1)
        return norm_emb


# ==============================================================================
# 3. DEFINING THE SUPERVISED CONTRASTIVE LOSS (SUPCON)
# ==============================================================================
class SupConLoss(nn.Module):
    """
    Implements Supervised Contrastive Learning loss.
    Based on arXiv:2004.11362. Pulls embeddings of the same class together,
    pushes different classes apart, including both views of the same image.
    """

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Embeddings shape: [batch_size * n_views, projector_dim]
        Labels shape: [batch_size]
        """
        device = embeddings.device
        batch_size = labels.shape[0]
        n_views = embeddings.shape[0] // batch_size  # Will be 2 in our case (orig vs fg)

        # Correctly replicate labels for all views
        labels_replic = labels.contiguous().view(-1, 1)
        # Create mask for same-class pairs: [bs * n_views, bs * n_views]
        label_mask = torch.eq(labels_replic.repeat(n_views, 1), labels_replic.repeat(1, n_views).T).float().to(device)

        # Calculate logits (scaled cosine similarity between ALL embeddings)
        # Result dim: [bs * n_views, bs * n_views]
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        logits = similarity_matrix / self.temperature

        # Mask out self-similarity (the diagonal), we don't want to compare an image to itself
        logits_mask = torch.scatter(
            torch.ones_like(label_mask),
            1,
            torch.arange(batch_size * n_views).view(-1, 1).to(device),
            0
        )
        # label_mask combines same-class identification AND removes self-similarity
        label_mask = label_mask * logits_mask

        # Compute logarithm of the sum of exponents
        # We subtract max(logits) for numerical stability (LogSumExp trick)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        exp_logits = torch.exp(logits - logits_max) * logits_mask
        sum_exp_logits = exp_logits.sum(1, keepdim=True)

        # Compute supervised contrastive loss
        log_prob = logits - logits_max - torch.log(sum_exp_logits + 1e-10)  # 1e-10 prevents NaN

        # Compute mean of log-likelihood over positive (same class) pairs
        mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-10)

        # Final temperature-scaled loss
        loss = - (self.temperature / self.temperature) * mean_log_prob_pos
        loss = loss.view(n_views, batch_size).mean()

        return loss


# ==============================================================================
# 4. MAIN TRAINING ROUTINE
# ==============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for v1, v2, labels in tqdm(dataloader, desc="Training Epoch", unit="batch"):
        v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)

        # For contrastive loss, we concatenate views: [2 * batch_size, resolution^2 * 3]
        inputs = torch.cat([v1, v2], dim=0)

        optimizer.zero_grad()

        # Generate normalized projected embeddings
        embeddings = model(inputs)

        # Compute SupCon loss based on replicated labels
        loss = criterion(embeddings, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Setup paths from your PyTorch audit script organization
    BASE_DATA_DIR = r"D:\Datasets\Medical\brisc2025\classification_task"
    ROOT_ORIG = os.path.join(BASE_DATA_DIR, "train")
    ROOT_FG = os.path.join(BASE_DATA_DIR, "train_foreground_only")

    # Audit log initialization
    sys.stdout = Logger()

    # HYPERPARAMETERS tailered for BRISC/EffNetB0 SupCon
    BATCH_SIZE = 64  # High BS is critical for contrastive performance.
    TEMPERATURE = 0.07
    LR = 3e-4  # Moderate LR for contrastive pre-training
    EPOCHS = 50  # Contrastive requires long training; 50 is a solid minimal test.
    RESOLUTION = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MICCAI REPRESENTATION DE-BIASING PIPELINE (SupCon) ===")
    print(f"Backbone Architecture: EfficientNet-B0")
    print(f"Goal: Minimize contextual shortcuts (Radial Depth, Dural border)")
    print(f"Execution Core Hardware: {device}\n")

    if not os.path.exists(ROOT_ORIG) or not os.path.exists(ROOT_FG):
        print(f"[CRITICAL ERROR] Required aligned datasets 'train' and 'train_foreground_only' not found.")
        print("Please generate the folders using your previous MICCAI audit scripts first.")
        return

    # Data Transforms: Standard Augmentations are CRITICAL for contrastive success.
    # ColorJitter prevents the model from relying on subtle intensity/contrast biases.
    # HorizontalFlip prevents relying on left-right hemisphere location.
    data_transforms = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloader (Batch shuffling is mandatory for contrastive)
    contrastive_dataset = BRISC_ContrastiveDataset(ROOT_ORIG, ROOT_FG, data_transforms)
    dataloader = DataLoader(contrastive_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
                            drop_last=True)

    # Initialize model, criterion, and optimizer
    model = SupConEffNet(projector_dim=128).to(device)
    criterion = SupConLoss(temperature=TEMPERATURE)

    # Contrastive sensitive to optimizer; Adam standard; AdamW better for w_decay.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine annealing is standard practice for SupCon/SimCLR pre-training schedules
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 50.0)

    best_loss = float('inf')

    print(f"Starting Context-Invariant Contrastive Pre-training for {EPOCHS} epochs...")
    print(f"Metric Focus: Morphology of Glioma/Meningioma/Pituitary vs. Spatial Priors.\n")

    # Training Loop
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")

        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Mean SupCon Loss = {avg_loss:.4f} | Learning Rate = {current_lr:.6f}")

        # Save based on best loss convergence
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "brisc_effnetB0_spatially_debiased_encoder.pth")
            print(f"      ---> Learned parameters converged. Saving Encoder.")

    print("\nContrastive Pre-training Complete.")
    print(f"Encoder state successfully saved to: {os.path.abspath('brisc_effnetB0_spatially_debiased_encoder.pth')}")
    print("\nNext Steps for the MICCAI Experiment:")
    print("1. Freeze this learned encoder (backbone).")
    print("2. Train a linear classifier (Linear Head) on top using standard classification loss.")
    print("3. Re-run the Diagnostic Audit scorecard on this debiased model to verify morphology focus.")


if __name__ == '__main__':
    main()