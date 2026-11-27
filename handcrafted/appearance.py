"""F12–F20. Lesion brightness, texture, architecture and boundary appearance."""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, measure

import config as C
from feature_utils import EPS, entropy, histogram, prefix, ratio, region_intensity, stats, vector
from texture import lbp_features, multiscale, radiomics


def normalized_intensity(regions):
    gray, mask = regions.gray, regions.mask
    distant = regions.head & (regions.outside > 14 * regions.scale)
    local = regions.ring(3, 14, restrict=True)
    tumor = gray[mask].astype(float)
    out = {}
    for name, support in (("distant", distant), ("local", local)):
        reference = gray[support].astype(float)
        if len(reference) < 20:
            reference = np.array([np.nan])
        center = np.median(reference)
        mad = np.median(abs(reference - center))
        normalized = (tumor - center) / (1.4826 * mad) if mad > EPS else np.full(tumor.size, np.nan)
        out.update(prefix(f"{name}_robust_z", stats(normalized)))
        out[f"{name}_median_ratio"] = ratio(np.median(tumor), center)
        out[f"{name}_mean_ratio"] = ratio(tumor.mean(), reference.mean())
        out[f"{name}_median_difference"] = np.median(tumor) - center
    return out


def compartments(regions):
    gray, mask = regions.gray, regions.mask
    tumor = gray[mask]
    p10, p25, p75, p90 = np.percentile(tumor, (10, 25, 75, 90))
    out = dict(
        p90_minus_p10=p90 - p10,
        median_minus_p25=np.median(tumor) - p25,
        p75_minus_median=p75 - np.median(tumor),
    )
    bright = mask & (gray > p75)
    dark = mask & (gray < p25)
    for name, part in (("dark", dark), ("bright", bright)):
        sizes = np.bincount(measure.label(part, connectivity=2).ravel())[1:]
        out.update(
            prefix(
                name,
                dict(
                    fraction=part.sum() / mask.sum(),
                    count=len(sizes),
                    largest_fraction=sizes.max(initial=0) / mask.sum(),
                    fragmentation=ratio(len(sizes), part.sum()),
                    mean_depth=regions.inside[part].mean() / regions.inside.max() if part.any() else np.nan,
                ),
            )
        )
    out["bright_minus_dark_mean"] = (
        (gray[bright].mean() - gray[dark].mean()) if bright.any() and dark.any() else np.nan
    )
    for size in (5, 9, 15):
        valid = regions.inside > np.sqrt(2) * (size // 2)
        image = gray.astype(float)
        local_mean = ndi.uniform_filter(image, size)
        variance = np.maximum(0, ndi.uniform_filter(image**2, size) - local_mean**2)
        out.update(prefix(f"local_std_{size}", stats(np.sqrt(variance[valid]))))
    for name, sign in (("bright", 1), ("dark", -1)):
        response = -sign * ndi.gaussian_laplace(gray.astype(float), 1.5)
        valid = (regions.inside > 9) & (response > 1)
        peaks = valid & (response == ndi.maximum_filter(response, 5))
        out[f"{name}_blob_density"] = peaks.sum() / mask.sum()
        out[f"{name}_blob_strength"] = response[peaks].mean() if peaks.any() else np.nan
    return out


def radial(regions):
    gray, mask = regions.gray, regions.mask
    depth = regions.inside / regions.inside.max()
    gray_values = gray[mask]
    bright_threshold, dark_threshold = np.percentile(gray_values, (75, 25))
    means, dispersions = [], []
    out = {}
    for index in range(4):
        band = mask & (depth > index / 4) & (depth <= (index + 1) / 4)
        measures = region_intensity(gray, band)
        out.update(prefix(f"band_{index}_rim_to_core", measures))
        means.append(measures["mean"])
        dispersions.append(measures["std"])
    rim = mask & (depth <= 0.25)
    core = depth > 0.75
    rim_bright = rim & (gray > bright_threshold)
    bright_parts = np.bincount(measure.label(rim_bright, connectivity=2).ravel())[1:]
    out.update(
        rim_minus_core=means[0] - means[-1],
        rim_to_core=ratio(means[0], means[-1]),
        intensity_slope=np.polyfit(np.arange(4), means, 1)[0] if np.isfinite(means).all() else np.nan,
        dispersion_slope=np.polyfit(np.arange(4), dispersions, 1)[0]
        if np.isfinite(dispersions).all()
        else np.nan,
        bright_rim_coverage=ratio(rim_bright.sum(), rim.sum()),
        dark_core_fraction=ratio((core & (gray < dark_threshold)).sum(), core.sum()),
        bright_rim_continuity=ratio(bright_parts.max(initial=0), rim_bright.sum()),
        bright_rim_depth_fraction=depth[rim_bright].mean() if rim_bright.any() else np.nan,
    )
    return out


def boundary(regions):
    gray, mask = regions.gray, regions.mask
    inner_edge = mask & ~ndi.binary_erosion(mask)
    y, x = np.nonzero(inner_edge)
    signed = regions.inside - regions.outside
    dy, dx = np.gradient(ndi.gaussian_filter(signed, 1))
    length = np.hypot(dx[y, x], dy[y, x])
    ny, nx = dy[y, x] / (length + EPS), dx[y, x] / (length + EPS)
    spacing = max(regions.scale, 0.75)
    offsets = np.arange(-3, 4) * spacing
    coordinates = (y[:, None] + ny[:, None] * offsets, x[:, None] + nx[:, None] * offsets)
    samples = ndi.map_coordinates(gray.astype(float), coordinates, order=1, mode="constant", cval=np.nan)
    nearest_support = ndi.map_coordinates(mask.astype(np.uint8), coordinates, order=0, mode="constant")
    usable = (
        np.isfinite(samples).all(axis=1)
        & (length > EPS)
        & (nearest_support[:, 0] == 0)
        & (nearest_support[:, -1] == 1)
    )
    samples = samples[usable]
    transitions = np.diff(samples, axis=1) / spacing
    slopes = abs(transitions).max(axis=1)
    contrast = samples[:, -2:].mean(axis=1) - samples[:, :2].mean(axis=1)
    transition_width = (abs(transitions[slopes > EPS]) >= slopes[slopes > EPS, None] * 0.5).sum(
        axis=1
    ) * spacing
    gx = cv2.Sobel(gray.astype(float), -1, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(gray.astype(float), -1, 0, 1, ksize=3) / 8
    gradient = np.hypot(gx[y, x], gy[y, x])
    alignment = abs(gx[y, x] * nx + gy[y, x] * ny) / (gradient + EPS)
    canny = feature.canny(gray.astype(float) / 255, sigma=1)
    edge_distance = ndi.distance_transform_edt(~canny)[y, x] if canny.any() else np.full(len(y), np.nan)
    return {
        **prefix("gradient", stats(gradient)),
        **prefix("normal_contrast", stats(contrast)),
        **prefix("normal_peak_slope", stats(slopes)),
        "mean_half_peak_width_px": transition_width.mean() if len(transition_width) else np.nan,
        "strong_gradient_fraction": (gradient >= 12).mean(),
        "weak_gradient_fraction": (gradient < 3).mean(),
        "mean_normal_alignment": alignment.mean(),
        "mean_canny_distance_px": edge_distance.mean(),
        "valid_profile_fraction": usable.mean(),
        "boundary_intensity_entropy": entropy(histogram(gray[inner_edge])),
        "within_profile_texture_change": (samples[:, -3:].std(1) - samples[:, :3].std(1)).mean()
        if len(samples)
        else np.nan,
    }


def extract(regions, sensitivity=True):
    gray, mask = regions.gray, regions.mask
    if not mask.any():
        return {}
    values = gray[mask]
    groups = {
        "F12": region_intensity(gray, mask, detailed=True),
        "F13": normalized_intensity(regions),
        "F14": vector("gray_frequency", histogram(values, 32)),
        "F15": radiomics(gray, mask, C.TEXTURE_BINS),
        "F16": lbp_features(gray, mask),
        "F17": multiscale(gray, mask),
        "F18": compartments(regions),
        "F19": radial(regions),
        "F20": boundary(regions),
    }
    for levels in C.TEXTURE_SENSITIVITY_BINS if sensitivity else ():
        groups[f"F15_bins{levels}"] = radiomics(gray, mask, levels)
    return groups
