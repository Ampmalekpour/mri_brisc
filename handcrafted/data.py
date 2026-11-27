"""BRISC image I/O and measurements shared by the stages."""

import json
import re
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config as C


PLANES = ("Axial", "Coronal", "Sagittal")


def plane_from_filename(path):
    tokens = set(re.split(r"[_.\-\s]+", Path(path).stem.lower()))
    for label, names in zip(
        PLANES, ({"ax", "tra", "axial"}, {"co", "cor", "coronal"}, {"sa", "sag", "sagittal"})
    ):
        if names & tokens:
            return label
    return "Unknown"


def read_image(path):
    # Ignore EXIF orientation so grayscale, RGB and annotation coordinates agree.
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def grayscale(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def read_mask(path, shape):
    if not path:
        return np.zeros(shape, dtype=bool)
    with Image.open(path) as source:
        raw = np.asarray(source.convert("L"))
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    # PNG masks may be binary 0/1 or stored as 0/255 (potentially with JPEG noise).
    return raw > (0 if raw.max() <= 1 else 127)


def bounds(mask):
    y, x = np.nonzero(mask)
    return slice(y.min(), y.max() + 1), slice(x.min(), x.max() + 1)


def distance_to_zeros(support):
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Squared pixel distances are integers. Recover them before comparing ring widths.
    return np.sqrt(np.rint(np.square(distance, dtype=np.float64)))


def distance_inside(mask):
    return distance_to_zeros(np.pad(mask, 1))[1:-1, 1:-1]


def foreground(gray):
    """Largest filled foreground component, a 2-D head outline proxy."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    raw = (gray > C.BRAIN_THRESHOLD).astype("uint8")
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(cleaned)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return out.astype(bool)


def output_path(*parts):
    path = C.OUTPUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(table, *parts):
    path = output_path(*parts)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def manifest_digest():
    return sha256(output_path("index", "manifest.csv").read_bytes()).hexdigest()


def save_json(contents, *parts):
    path = output_path(*parts)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def feature_settings(stage=None):
    settings = {"head_threshold": C.BRAIN_THRESHOLD}
    if stage in (None, 3, 4):
        settings["texture_bins"] = C.TEXTURE_BINS
    if stage in (None, 3):
        settings["texture_sensitivity_bins"] = list(C.TEXTURE_SENSITIVITY_BINS)
    if stage in (None, 4):
        settings["ring_edges"] = list(C.RING_EDGES)
        settings["patch_sampling_seed"] = C.SEED
    return settings


def load_manifest():
    path = output_path("index", "manifest.csv")
    if not path.exists():
        raise FileNotFoundError(f"Run s01_index.py first. Missing {path}")
    return pd.read_csv(path, keep_default_na=False)


def resolve(relative_path):
    return C.DATA_ROOT / relative_path


def sample_arrays(row):
    image = read_image(resolve(row.path))
    gray = grayscale(image)
    mask_path = resolve(row.mask_path) if row.mask_usable else None
    mask = read_mask(mask_path, gray.shape)
    if row.mask_usable and not mask.any():
        raise ValueError(f"Mask changed since indexing for {row.sample_id}. Rerun s01_index.py.")
    return image, gray, mask
