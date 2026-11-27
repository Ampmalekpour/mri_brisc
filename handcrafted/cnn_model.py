"""Matched full-image and mask-guided CNN components."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from data import grayscale, read_image, read_mask, resolve


class MRIViews(Dataset):
    def __init__(self, rows, target, size, mask_guided):
        self.rows = list(rows.itertuples(index=False))
        self.target = target
        self.size = (size, size)
        self.mask_guided = mask_guided

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        gray = grayscale(read_image(resolve(row.path)))
        image = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if self.mask_guided:
            mask = read_mask(resolve(row.mask_path), gray.shape)
            if not mask.any():
                raise ValueError(f"Mask changed since indexing for {row.sample_id}. Rerun s01_index.py.")
            mask = cv2.resize(mask.astype(np.float32), self.size, interpolation=cv2.INTER_AREA)
        else:
            mask = np.zeros(self.size, dtype=np.float32)
        return torch.from_numpy(np.stack((image, mask)).astype(np.float32)), int(self.target[index])


class MRIClassifier(nn.Module):
    """Same network and two-channel input for both variants; channel 2 is zero for the full image."""

    def __init__(self, classes):
        super().__init__()
        layers = []
        source = 2
        for width in (24, 48, 96, 128):
            layers.extend(
                [
                    nn.Conv2d(source, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            source = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, classes)
        )

    def forward(self, image):
        return self.head(self.features(image))
