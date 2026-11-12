"""
Shared helpers: image/mask IO, brain segmentation, the image *variants* (interventions), the
dataset class and the model factory. Every other script is a thin layer on top of this file.
"""
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import config as C

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------------------------ misc
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_plane(name: str) -> str:
    """Axial / Coronal / Sagittal from BRISC file names such as brisc2025_train_00001_gl_ax_t1.jpg."""
    toks = [t for t in re.split(r"[_.\-\s]+", Path(name).stem.lower()) if t]
    if any(t in ("co", "cor", "coronal") for t in toks):
        return "Coronal"
    if any(t in ("sa", "sag", "sagittal") for t in toks):
        return "Sagittal"
    if any(t in ("ax", "tra", "axial") for t in toks):
        return "Axial"
    return "Unknown"


def load_index(splits=None, tumor_only=False) -> pd.DataFrame:
    if not C.INDEX_CSV.exists():
        raise FileNotFoundError(f"{C.INDEX_CSV} missing - run s00_build_index.py first.")
    df = pd.read_csv(C.INDEX_CSV)
    if splits is not None:
        df = df[df["split"].isin(splits if isinstance(splits, (list, tuple)) else [splits])]
    if tumor_only:
        df = df[df["class"].isin(C.TUMOR_CLASSES)]
    return df.reset_index(drop=True)


# ------------------------------------------------------------------------------ image IO
def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path, shape) -> np.ndarray:
    """Binary tumor mask resized (nearest) to `shape`=(H,W). Empty mask if path is missing."""
    if not isinstance(path, str) or not path or not Path(path).exists():
        return np.zeros(shape[:2], dtype=bool)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape[:2], dtype=bool)
    if m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def brain_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Foreground (head/brain) mask: threshold + morphology + largest connected contour, filled."""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(g, C.BRAIN_THRESHOLD, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(g.shape, np.uint8)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out > 0


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or not mask.any():
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask.astype(np.uint8), ker) > 0


def centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else (np.nan, np.nan)


# ------------------------------------------------------------------------------ interventions
def translate(arr: np.ndarray, dx: float, dy: float, interp=cv2.INTER_LINEAR) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def paste_tumor(base, donor_img, donor_mask, dx=0.0, dy=0.0, feather=2.0, ring=3):
    """Paste the donor tumor (pixels under its mask, plus a thin ring) onto `base`, shifted by (dx,dy).
    A Gaussian-feathered alpha hides the seam."""
    dm = dilate(donor_mask, ring)
    sh_img = translate(donor_img, dx, dy)
    sh_m = translate(dm.astype(np.uint8) * 255, dx, dy, cv2.INTER_NEAREST)
    alpha = cv2.GaussianBlur(sh_m.astype(np.float32) / 255.0, (0, 0), feather)[..., None]
    out = alpha * sh_img.astype(np.float32) + (1 - alpha) * base.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def inpaint(img, mask, k=0):
    m = dilate(mask, k).astype(np.uint8) * 255
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA) if m.any() else img.copy()


def random_brain_shift(mask, brain, rng, margin=12):
    """Random translation (dx,dy) that moves the mask centroid to a random pixel inside the
    (eroded) brain. Used for the placebo intervention and for location-randomising augmentation."""
    cx, cy = centroid(mask)
    if np.isnan(cx):
        return 0.0, 0.0
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    inner = cv2.erode(brain.astype(np.uint8), ker) > 0
    ys, xs = np.nonzero(inner)
    if len(xs) == 0:
        return 0.0, 0.0
    j = rng.integers(len(xs))
    return float(xs[j] - cx), float(ys[j] - cy)


def apply_variant(img, mask, variant: str, rng=None, k=0):
    """
    Return the intervened image. `variant` is one of

      clean        untouched
      zero         tumor (dilated by k px) set to black         -> leaves a black silhouette
      gaussian     tumor replaced by noise matched to the brain intensity statistics
      inpaint      tumor removed and filled in by Telea inpainting (no silhouette, small blur)
      placebo      the SAME black silhouette but moved to a random brain location (tumor stays!)
      relocate     tumor moved to a random brain location, old site inpainted (training augmentation)
      foreground   only the tumor is kept, everything else black
      crop         tight bounding box around the tumor (+25 % margin), resized to full frame
      mask         the binary mask itself as an image (shape/location only, no tissue)

    Images without a mask (no_tumor) are returned unchanged for every variant except `mask`/`crop`
    (all black / centre crop) which are only meaningful for tumor-only experiments.
    """
    rng = rng or np.random.default_rng()
    H, W = img.shape[:2]
    if variant == "clean":
        return img
    if variant == "mask":
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    if not mask.any():
        return img
    if variant == "zero":
        out = img.copy(); out[dilate(mask, k)] = 0; return out
    if variant == "gaussian":
        br = brain_mask(img); vals = img[br] if br.any() else img.reshape(-1, 3)
        mu, sd = vals.mean(0), vals.std(0) + 1e-6
        noise = np.clip(rng.normal(mu, sd, img.shape), 0, 255).astype(np.uint8)
        out = img.copy(); m = dilate(mask, k); out[m] = noise[m]; return out
    if variant == "inpaint":
        return inpaint(img, mask, k)
    if variant == "placebo":
        dx, dy = random_brain_shift(mask, brain_mask(img), rng)
        shifted = translate(dilate(mask, k).astype(np.uint8), dx, dy, cv2.INTER_NEAREST) > 0
        out = img.copy(); out[shifted] = 0; return out
    if variant == "relocate":
        dx, dy = random_brain_shift(mask, brain_mask(img), rng)
        return paste_tumor(inpaint(img, dilate(mask, 3)), img, mask, dx, dy)
    if variant == "foreground":
        out = np.zeros_like(img); m = dilate(mask, k); out[m] = img[m]; return out
    if variant == "crop":
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        my, mx = int(0.25 * (y1 - y0)) + 2, int(0.25 * (x1 - x0)) + 2
        y0, y1, x0, x1 = max(0, y0 - my), min(H, y1 + my), max(0, x0 - mx), min(W, x1 + mx)
        return cv2.resize(img[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"unknown variant {variant}")


# ------------------------------------------------------------------------------ dataset
def to_tensor(img_rgb: np.ndarray, size: int) -> torch.Tensor:
    x = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class BriscDataset(Dataset):
    """
    Reads images (+ masks) from index rows and applies `variant` on the fly, so no extra image
    folders have to be generated on disk.

    train=True adds light augmentation (flip, small rotation) and, with probability `p_relocate`,
    the location-randomising mitigation (tumor moved to a random brain position).
    """

    def __init__(self, df, classes, variant="clean", k=0, size=C.IMAGE_SIZE, train=False,
                 p_relocate=0.0, seed=0):
        self.df = df.reset_index(drop=True)
        self.cls2idx = {c: i for i, c in enumerate(classes)}
        self.variant, self.k, self.size, self.train, self.p_relocate = variant, k, size, train, p_relocate
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # deterministic noise per (image, seed) for evaluation, free-running rng for training
        rng = self.rng if self.train else np.random.default_rng(self.seed * 1_000_003 + i)
        img = read_rgb(r["path"])
        mask = read_mask(r.get("mask_path", ""), img.shape)
        variant = self.variant
        if self.train and self.p_relocate > 0 and mask.any() and rng.random() < self.p_relocate:
            variant = "relocate"
        img = apply_variant(img, mask, variant, rng, self.k)
        if self.train:
            if rng.random() < 0.5:
                img = img[:, ::-1]
            ang = rng.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), ang, 1.0)
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (img.shape[1], img.shape[0]))
        return to_tensor(np.ascontiguousarray(img), self.size), self.cls2idx[r["class"]], i


# ------------------------------------------------------------------------------ models
def make_model(arch: str, num_classes: int, pretrained=None, img_size=C.IMAGE_SIZE):
    import timm
    pretrained = C.PRETRAINED if pretrained is None else pretrained
    kw = dict(pretrained=pretrained, num_classes=num_classes)
    if arch.startswith(("vit", "deit")):
        kw["img_size"] = img_size
    return timm.create_model(arch, **kw)


def load_run(run_dir, device=None):
    """Load a trained run (model + metadata)."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    model = make_model(meta["arch"], len(meta["classes"]), pretrained=False, img_size=meta["size"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.to(device or get_device()).eval()
    return model, meta


@torch.no_grad()
def predict(model, dataset, device, batch_size=C.BATCH_SIZE):
    """Softmax probabilities for a dataset (returns array [N, K], in dataset order)."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=C.NUM_WORKERS)
    out = []
    model.eval()
    for x, _, _ in loader:
        out.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(dataset.cls2idx)))


def bootstrap_ci(values, n=C.N_BOOTSTRAP, seed=0, alpha=0.05):
    """Mean and percentile bootstrap CI of a 1-D array (e.g. per-image correctness)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
