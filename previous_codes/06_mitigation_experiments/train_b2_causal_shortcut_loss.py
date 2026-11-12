# ============================================================================
# PREVIOUS CODE (archived, logic unchanged)
# Original file : EfficientNets/train_b2_pair_causal.py
# Purpose       :
#   EfficientNet-B2 trained on (clean, zero-out) pairs: CE(clean) + alpha*KL(uniform || softmax(zero-out)).
#   CAVEAT: the penalty teaches the model to recognise black blobs; evaluate on UNSEEN interventions.
# Paths below (D:\Datasets\..., relative weight folders) are the original author's
# machine paths and were deliberately left as-is; see ../README.md.
# ============================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# ---------------------------------------------------------
# 1. CUSTOM PAIRED DATALOADER
# ---------------------------------------------------------
class PairedCausalDataset(Dataset):
    """
    Loads the clean image and its corresponding zero-out mask simultaneously.
    """

    def __init__(self, clean_dir, zero_dir, transform=None):
        self.clean_ds = datasets.ImageFolder(clean_dir)
        self.zero_dir = zero_dir
        self.transform = transform
        self.classes = self.clean_ds.classes

    def __len__(self):
        return len(self.clean_ds)

    def __getitem__(self, idx):
        # Get clean image and label
        img_path, label = self.clean_ds.samples[idx]
        clean_img = Image.open(img_path).convert('RGB')

        # Reconstruct path to find the matching zero-out image
        rel_path = os.path.relpath(img_path, self.clean_ds.root)
        zero_path = os.path.join(self.zero_dir, rel_path)

        try:
            zero_img = Image.open(zero_path).convert('RGB')
        except FileNotFoundError:
            # Fallback to clean image if mask generation failed for this specific file
            zero_img = clean_img.copy()

        if self.transform:
            clean_tensor = self.transform(clean_img)
            zero_tensor = self.transform(zero_img)

        return clean_tensor, zero_tensor, label


# ---------------------------------------------------------
# 2. CAUSAL PENALTY LOSS FUNCTION
# ---------------------------------------------------------
class CausalShortcutLoss(nn.Module):
    """
    Forces the network to be highly uncertain when the pathology is erased.
    """

    def __init__(self, alpha=1.0):
        super(CausalShortcutLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, outputs_clean, outputs_zero, labels):
        # 1. Learn the actual pathology from the clean image
        loss_main = self.ce_loss(outputs_clean, labels)

        # 2. Punish the network if it is confident on the erased image
        num_classes = outputs_zero.size(1)
        uniform_targets = torch.ones_like(outputs_zero) / num_classes
        log_probs_zero = torch.log_softmax(outputs_zero, dim=1)

        # KL Divergence pushes the prediction towards a flat 25% for all 4 classes
        penalty = nn.KLDivLoss(reduction='batchmean')(log_probs_zero, uniform_targets)

        return loss_main + (self.alpha * penalty)


# ---------------------------------------------------------
# 3. MASTER EXECUTION LOOP
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INITIALIZING CAUSAL DEFENSE TRAINING ===")
    print(f"Execution Core Hardware: {device}\n")

    data_dir = r"D:\Datasets\Medical\brisc2025\classification_task"

    # Define directories
    train_clean_dir = os.path.join(data_dir, "train")
    train_zero_dir = os.path.join(data_dir, "train_mask_zero")  # Ensure you generate this!

    val_clean_dir = os.path.join(data_dir, "val")
    val_zero_dir = os.path.join(data_dir, "val_mask_zero")

    batch_size = 16  # Lowered slightly to account for dual-forward pass memory
    num_epochs = 10
    learning_rate = 0.001

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Initialize Paired Datasets
    image_datasets = {
        'train': PairedCausalDataset(train_clean_dir, train_zero_dir, transform=data_transforms['train']),
        'val': PairedCausalDataset(val_clean_dir, val_zero_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Model Initialization (EfficientNetB2)
    weights = models.EfficientNet_B2_Weights.DEFAULT
    model = models.efficientnet_b2(weights=weights)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # Use our custom causal loss
    criterion = CausalShortcutLoss(alpha=1.0)  # alpha controls how strictly we punish shortcuts
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    save_path = 'brisc_effnet_b2_causal_v2.pth'

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Loader now unpacks 3 items
            for inputs_clean, inputs_zero, labels in dataloaders[phase]:
                inputs_clean = inputs_clean.to(device)
                inputs_zero = inputs_zero.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass both variants
                    outputs_clean = model(inputs_clean)
                    outputs_zero = model(inputs_zero)

                    _, preds = torch.max(outputs_clean, 1)

                    # Calculate custom dual-loss
                    loss = criterion(outputs_clean, outputs_zero, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Tracking metrics based on clean performance
                running_loss += loss.item() * inputs_clean.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f">>> Saved new causally-defended model: {save_path}")

    print(f'\nTraining complete. Best Validation Accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()