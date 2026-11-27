"""Small image and feature helpers used by the extraction stages."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from data import bounds, distance_inside, foreground


EPS = 1e-8


@dataclass
class Regions:
    gray: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.head = foreground(self.gray)
        self.scale = min(self.gray.shape) / 256.0
        self.inside = distance_inside(self.mask) if self.mask.any() else None
        self.outside = ndi.distance_transform_edt(~self.mask) if self.mask.any() else None

    def ring(self, low, high, restrict=False):
        ring = (self.outside > low * self.scale) & (self.outside <= high * self.scale)
        return ring & self.head if restrict else ring


def prefix(name, values):
    return {f"{name}_{key}": value for key, value in values.items()}


def vector(name, values):
    return {f"{name}_{i:03d}": float(value) for i, value in enumerate(np.ravel(values))}


def histogram(values, bins=32, low=0, high=256):
    # Counts and density have distinct semantics. Return frequencies summing to one.
    counts = np.histogram(values, bins=bins, range=(low, high))[0].astype(float)
    return counts / counts.sum() if counts.sum() else np.full(bins, np.nan)


def entropy(frequencies):
    positive = frequencies[frequencies > 0]
    return float(-np.sum(positive * np.log2(positive))) if positive.size else np.nan


def stats(values, detailed=False):
    values = np.asarray(values, dtype=float)
    if not values.size:
        values = np.array([np.nan])
    mean = values.mean()
    sd = values.std()
    median = np.median(values)
    result = {
        "mean": mean,
        "std": sd,
        "median": median,
        "min": values.min(),
        "max": values.max(),
        "mad": np.median(np.abs(values - median)),
        "p10": np.percentile(values, 10),
        "p90": np.percentile(values, 90),
        "skewness": np.mean(((values - mean) / sd) ** 3) if sd > EPS else 0.0 * sd,
    }
    if detailed:
        percentiles = (1, 5, 25, 75, 95, 99)
        result.update(dict(zip((f"p{p:02d}" for p in percentiles), np.percentile(values, percentiles))))
        result.update(
            range=result["max"] - result["min"],
            iqr=result["p75"] - result["p25"],
            variance=sd**2,
            rms=np.sqrt(np.mean(values**2)),
            kurtosis=(np.mean(((values - mean) / sd) ** 4) - 3) if sd > EPS else 0.0 * sd,
        )
    return result


def ratio(a, b):
    return a / b if abs(b) > EPS else np.nan


def region_intensity(gray, support, detailed=False):
    values = gray[support]
    result = stats(values, detailed)
    result["entropy"] = entropy(histogram(values))
    return result


def area_resize(image, size):
    return cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def centered_view(image, mask, size=32, occupancy=0.8):
    """Fit the lesion box isotropically to the central canvas, using the same transform for image and mask."""
    ys, xs = bounds(mask)
    height, width = ys.stop - ys.start, xs.stop - xs.start
    scale = occupancy * size / max(height, width)
    x = (size - 1) / 2 - scale * (xs.start + xs.stop - 1) / 2
    y = (size - 1) / 2 - scale * (ys.start + ys.stop - 1) / 2
    matrix = np.array([[scale, 0, x], [0, scale, y]], dtype=float)
    values = cv2.warpAffine(image.astype(np.float32), matrix, (size, size), flags=cv2.INTER_LINEAR)
    support = cv2.warpAffine(mask.astype(np.uint8), matrix, (size, size), flags=cv2.INTER_NEAREST).astype(
        bool
    )
    if not support.any():
        yy, xx = np.argwhere(mask)[0]
        cy, cx = np.clip(np.rint((scale * yy + y, scale * xx + x)).astype(int), 0, size - 1)
        support[cy, cx] = True
    return values, support


def mask_weighted_resample(gray, mask, size=32):
    """Resize a tight crop. Only real ROI pixels contribute to each intensity cell."""
    box = bounds(mask)
    support = area_resize(mask[box], size)
    summed = area_resize(np.where(mask[box], gray[box], 0), size)
    result = np.divide(summed, support, out=np.full_like(summed, np.nan), where=support > EPS)
    return result, support
