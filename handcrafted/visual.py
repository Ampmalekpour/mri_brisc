"""F23–F24. Classical descriptors and compact whole-image representations."""

import cv2
import numpy as np
from scipy.fft import dctn
from skimage.feature import hog

from data import bounds
from feature_utils import (
    area_resize,
    centered_view,
    histogram,
    mask_weighted_resample,
    prefix,
    region_intensity,
    vector,
)

PATCH_DIM = 16


def hog_vector(image):
    return hog(
        image.astype(float),
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def tumor_patch_descriptors(gray, mask):
    """Local 8×8 DCT descriptors from complete ROI patches in a normalized tumor crop."""
    image, support = mask_weighted_resample(gray, mask, size=64)
    patches = []
    for y in range(0, 57, 4):
        for x in range(0, 57, 4):
            roi = np.s_[y : y + 8, x : x + 8]
            if (support[roi] >= 0.999).all():
                patch = image[roi] / 255
                patches.append(dctn(patch, norm="ortho")[:4, :4].ravel())
    return np.asarray(patches, dtype=np.float32).reshape(-1, PATCH_DIM)


def whole_image(regions):
    gray, head = regions.gray, regions.head
    image = gray.astype(float)
    support = np.ones(gray.shape, dtype=bool)
    out = region_intensity(gray, support, detailed=True)
    out.update(vector("whole_histogram", histogram(gray.ravel(), 32)))
    for size in (8, 16):
        out.update(vector(f"whole_mri_{size}", area_resize(gray, size) / 255))
    head_view = gray[bounds(head)] if head.any() else gray
    out.update(vector("head_crop_8", area_resize(head_view, 8) / 255))
    out.update(vector("whole_low_dct", dctn(area_resize(gray, 32) / 255, norm="ortho")[:8, :8]))
    laplacian = cv2.Laplacian(image, -1)
    residual = image - cv2.medianBlur(gray, 3)
    out.update(
        laplacian_variance=laplacian.var(),
        median_filter_residual_mad=np.median(abs(residual)),
        saturated_fraction=(gray == 255).mean(),
        black_fraction=(gray == 0).mean(),
        unique_levels=np.unique(gray).size,
        **prefix("head", region_intensity(gray, head)),
    )
    for axis in (0, 1):
        values = np.moveaxis(image, axis, 0)
        near, far = values[:-1].ravel(), values[1:].ravel()
        out[f"pixel_correlation_axis{axis}"] = (
            np.corrcoef(near, far)[0, 1] if near.std() > 0 and far.std() > 0 else np.nan
        )
    return out


def extract(regions):
    gray, mask = regions.gray, regions.mask
    fine = {"F23": vector("full_hog", hog_vector(area_resize(gray, 32) / 255)), "F24": whole_image(regions)}
    if not mask.any():
        return fine, np.empty((0, PATCH_DIM), dtype=np.float32)

    context, centered_mask = centered_view(gray, mask, size=32, occupancy=0.65)
    tumor, _ = mask_weighted_resample(gray, mask)
    tumor = np.where(np.isfinite(tumor), tumor, gray[mask].mean()) / 255
    fine["F23"].update(vector("context_hog", hog_vector(context / 255)))
    fine["F23"].update(vector("mask_hog", hog_vector(centered_mask)))
    fine["F23"].update(vector("tumor_hog", hog_vector(tumor)))
    fine["F23"].update(vector("tumor_low_dct", dctn(tumor, norm="ortho")[:8, :8]))
    for size in (8, 16):
        image, _ = mask_weighted_resample(gray, mask, size)
        fine["F23"].update(vector(f"tumor_mri_{size}", image / 255))
    return fine, tumor_patch_descriptors(gray, mask)
