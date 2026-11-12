"""
Generates a tiny FAKE BRISC-like dataset (ellipse 'brains' with class-dependent tumor blobs) so the
whole pipeline can be smoke-tested without real data:

    python tools/make_synthetic_brisc.py D:\\tmp\\fake_brisc
    set BRISC_ROOT=D:\\tmp\\fake_brisc      (then run the steps; PRETRAINED=False recommended)

The tumor *location* is class-dependent on purpose (pituitary low-centre, meningioma at the rim,
glioma interior) and its *texture* is class-dependent only weakly, so shortcut analyses have
something to find. Numbers from this data mean nothing scientifically.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "fake_brisc")
N = {"train": 40, "val": 12, "test": 20}          # images per class per split
PLANES = {"ax": "t1", "co": "t1", "sa": "t1"}
rng = np.random.default_rng(0)
cls_tag = {"glioma": "gl", "meningioma": "me", "no_tumor": "nt", "pituitary": "pi"}


def make(cls, split, i):
    H = W = 256
    img = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[:H, :W]
    cx, cy, a, b = 128 + rng.integers(-6, 6), 128 + rng.integers(-6, 6), 90 + rng.integers(-8, 8), 105 + rng.integers(-8, 8)
    brain = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 < 1
    img[brain] = 90 + rng.normal(0, 8, brain.sum())
    img = cv2.GaussianBlur(img, (0, 0), 2)
    mask = np.zeros((H, W), np.uint8)
    if cls != "no_tumor":
        if cls == "pituitary":   px, py, r = cx + rng.normal(0, 5), cy + 0.35 * b + rng.normal(0, 5), 12 + rng.integers(0, 5)
        elif cls == "meningioma": ang = rng.uniform(0, 2 * np.pi); px, py, r = cx + 0.82 * a * np.cos(ang), cy + 0.82 * b * np.sin(ang), 14 + rng.integers(0, 8)
        else:                    ang = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.1, 0.55); px, py, r = cx + rr * a * np.cos(ang), cy + rr * b * np.sin(ang), 16 + rng.integers(0, 12)
        blob = ((xx - px) ** 2 + (yy - py) ** 2) < r ** 2
        tex = {"glioma": 190, "meningioma": 175, "pituitary": 165}[cls]
        img[blob] = tex + rng.normal(0, 12, blob.sum())
        mask[blob & brain] = 255
    img = np.clip(cv2.GaussianBlur(img, (0, 0), 1.0), 0, 255).astype(np.uint8)
    return img, mask


for split, n in N.items():
    for cls in cls_tag:
        (root / "classification_task" / split / cls).mkdir(parents=True, exist_ok=True)
        (root / "segmentation_task" / split / "images").mkdir(parents=True, exist_ok=True)
        (root / "segmentation_task" / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            plane = list(PLANES)[i % 3]
            name = f"brisc2025_{split}_{cls_tag[cls]}{i:04d}_{cls_tag[cls]}_{plane}_t1"
            img, mask = make(cls, split, i)
            cv2.imwrite(str(root / "classification_task" / split / cls / f"{name}.jpg"), img)
            if cls != "no_tumor":
                cv2.imwrite(str(root / "segmentation_task" / split / "images" / f"{name}.jpg"), img)
                cv2.imwrite(str(root / "segmentation_task" / split / "masks" / f"{name}.png"), mask)
print("done ->", root)
