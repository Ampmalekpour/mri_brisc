"""Masked 2-D texture matrices with fixed-bin grayscale discretization."""

import cv2
import numpy as np
import pywt
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import local_binary_pattern

from data import bounds, distance_inside
from feature_utils import EPS, entropy, prefix, stats, vector


DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def discretize(gray, levels):
    return np.minimum(gray.astype(np.int32) * levels // 256, levels - 1)


def shifted_views(array, dy, dx):
    height, width = array.shape
    if abs(dy) >= height or abs(dx) >= width:
        return array[:0, :0], array[:0, :0]
    a = array[max(0, -dy) : min(height, height - dy), max(0, -dx) : min(width, width - dx)]
    b = array[max(0, dy) : min(height, height + dy), max(0, dx) : min(width, width + dx)]
    return a, b


def glcm_statistics(matrix):
    p = matrix / matrix.sum()
    i, j = np.indices(p.shape)
    diff = abs(i - j)
    mean = (i * p).sum()
    variance = (((i - mean) ** 2) * p).sum()
    cluster = i + j - 2 * mean
    return dict(
        contrast=((diff**2) * p).sum(),
        dissimilarity=(diff * p).sum(),
        homogeneity=(p / (1 + diff**2)).sum(),
        energy=np.sqrt((p**2).sum()),
        correlation=(((i - mean) * (j - mean) * p).sum() / variance if variance > EPS else 1),
        joint_entropy=entropy(p),
        cluster_shade=((cluster**3) * p).sum(),
        cluster_prominence=((cluster**4) * p).sum(),
        max_probability=p.max(),
        difference_entropy=entropy(np.bincount(diff.ravel(), p.ravel())),
    )


def glcm_features(quantized, mask, levels, distances=(1, 2, 4, 8), variation=True):
    out = {}
    for distance in distances:
        directions = []
        for vy, vx in DIRECTIONS:
            offsets = vy * distance, vx * distance
            first, second = shifted_views(quantized, *offsets)
            m1, m2 = shifted_views(mask, *offsets)
            valid = m1 & m2
            if valid.any():
                code = first[valid] * levels + second[valid]
                matrix = np.bincount(code, minlength=levels**2).reshape(levels, levels)
                directions.append(glcm_statistics(matrix + matrix.T))
        for name in (
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "joint_entropy",
            "cluster_shade",
            "cluster_prominence",
            "max_probability",
            "difference_entropy",
        ):
            values = [row[name] for row in directions]
            out[f"glcm_d{distance}_{name}_mean"] = np.mean(values) if values else np.nan
            if variation:
                out[f"glcm_d{distance}_{name}_sd"] = np.std(values) if values else np.nan
    return out


def matrix_features(gray_levels, lengths, voxel_count, kind):
    # Tabulate occurrences at occupied (gray level, run/zone/dependence length) entries.
    coordinates, count = np.unique(np.column_stack([gray_levels + 1, lengths]), axis=0, return_counts=True)
    gray, length = coordinates.T.astype(float)
    p = count / count.sum()
    gray_totals = np.bincount(coordinates[:, 0], count)
    length_totals = np.bincount(coordinates[:, 1], count)
    return {
        f"small_{kind}_emphasis": (p / length**2).sum(),
        f"large_{kind}_emphasis": (p * length**2).sum(),
        "low_gray_emphasis": (p / gray**2).sum(),
        "high_gray_emphasis": (p * gray**2).sum(),
        f"small_{kind}_low_gray": (p / ((gray * length) ** 2)).sum(),
        f"large_{kind}_high_gray": (p * ((gray * length) ** 2)).sum(),
        "gray_nonuniformity_normalized": ((gray_totals / count.sum()) ** 2).sum(),
        f"{kind}_nonuniformity_normalized": ((length_totals / count.sum()) ** 2).sum(),
        "gray_variance": (p * (gray - (p * gray).sum()) ** 2).sum(),
        f"{kind}_variance": (p * (length - (p * length).sum()) ** 2).sum(),
        "entropy": entropy(p),
        "entries_per_pixel": count.sum() / voxel_count,
    }


def run_lengths(quantized, mask):
    y, x = np.nonzero(mask)
    gray = quantized[mask]
    for line, position in ((y, x), (x, y), (y - x, y), (y + x, y)):
        order = np.lexsort((position, line))
        line_values, coords, levels = line[order], position[order], gray[order]
        breaks = (np.diff(line_values) != 0) | (np.diff(coords) != 1) | (np.diff(levels) != 0)
        start = np.r_[0, np.flatnonzero(breaks) + 1]
        yield levels[start], np.diff(np.r_[start, len(levels)])


def zone_matrices(quantized, mask):
    zones = measure.label(np.where(mask, quantized, -1), background=-1, connectivity=2)
    ids = zones[mask]
    voxel_count = len(ids)
    areas = np.bincount(ids)[1:]
    zone_gray = np.zeros(len(areas) + 1, dtype=int)
    zone_gray[ids] = quantized[mask]
    # Distance is computed within the whole ROI with 4-connected erosion layers.
    depth = ndi.distance_transform_cdt(np.pad(mask, 1), metric="taxicab")[1:-1, 1:-1]
    distances = ndi.minimum(depth[mask], labels=ids, index=np.arange(1, len(areas) + 1)).astype(int)
    return {
        **prefix("glszm", matrix_features(zone_gray[1:], areas, voxel_count, "zone")),
        **prefix("gldzm", matrix_features(zone_gray[1:], distances, voxel_count, "distance")),
    }


def neighborhood_matrices(quantized, mask, levels):
    image = np.pad(quantized, 1)
    support = np.pad(mask, 1)
    height, width = mask.shape
    n = np.zeros_like(quantized)
    total = np.zeros_like(quantized)
    dependents = np.ones_like(quantized)
    for y in range(3):
        for x in range(3):
            if (y, x) == (1, 1):
                continue
            adjacent = support[y : y + height, x : x + width]
            neighbor = image[y : y + height, x : x + width]
            n += adjacent
            total += adjacent * neighbor
            dependents += adjacent & (abs(neighbor - quantized) <= 1)

    dependence = matrix_features(quantized[mask], dependents[mask], mask.sum(), "dependence")
    valid = mask & (n > 0)
    tone = dict.fromkeys(("coarseness", "contrast", "busyness", "complexity", "strength"), np.nan)
    if valid.any():
        gray = quantized[valid]
        differences = abs(gray - total[valid] / n[valid])
        counts = np.bincount(gray, minlength=levels)
        present = counts > 0
        p = counts[present] / valid.sum()
        s = np.bincount(gray, weights=differences, minlength=levels)[present]
        i = np.flatnonzero(present) + 1
        diff = abs(i[:, None] - i)
        pair_probability = p[:, None] * p
        weighted = p * s
        busyness_denominator = np.abs((i * p)[:, None] - i * p).sum()
        contrast = (pair_probability * diff**2).sum() / max(1, len(i) * (len(i) - 1))
        tone = dict(
            coarseness=1 / max(weighted.sum(), EPS),
            contrast=contrast * s.sum() / valid.sum(),
            busyness=weighted.sum() / max(busyness_denominator, EPS),
            complexity=(diff * (weighted[:, None] + weighted) / (p[:, None] + p)).sum() / valid.sum(),
            strength=((p[:, None] + p) * diff**2).sum() / max(s.sum(), EPS),
        )
    return {**prefix("gldm", dependence), **prefix("ngtdm", tone)}


def radiomics(gray, mask, levels, short=False):
    if not mask.any():
        return {}
    box = bounds(mask)
    support = mask[box]
    quantized = discretize(gray[box], levels)
    distances = (1, 2) if short else (1, 2, 4, 8)
    features = glcm_features(quantized, support, levels, distances, variation=not short)
    direction_features = [
        matrix_features(g, length, support.sum(), "run") for g, length in run_lengths(quantized, support)
    ]
    if short:
        for key in ("small_run_emphasis", "large_run_emphasis", "entropy"):
            features["glrlm_" + key] = np.mean([row[key] for row in direction_features])
        zones = zone_matrices(quantized, support)
        features.update(
            {
                key: zones[key]
                for key in ("glszm_small_zone_emphasis", "glszm_large_zone_emphasis", "glszm_entropy")
            }
        )
        return features
    for key in direction_features[0]:
        values = [row[key] for row in direction_features]
        features[f"glrlm_{key}_mean"] = np.mean(values)
        features[f"glrlm_{key}_sd"] = np.std(values)
    features.update(zone_matrices(quantized, support))
    features.update(neighborhood_matrices(quantized, support, levels))
    return features


def lbp_features(gray, support, radii=(1, 2, 3)):
    if not support.any():
        return {}
    box = bounds(support)
    gray, support = gray[box].astype(np.uint8), support[box]
    depth = distance_inside(support)
    out = {}
    for radius in radii:
        points = 8 * radius
        valid = depth > np.sqrt(2) * np.ceil(radius)
        codes = local_binary_pattern(gray, points, radius, method="uniform")[valid].astype(int)
        counts = np.bincount(codes, minlength=points + 2)
        out.update(
            vector(f"lbp_r{radius}", counts / counts.sum() if counts.sum() else np.full(len(counts), np.nan))
        )
    return out


def filter_response(values, valid):
    sampled = values[valid]
    measures = stats(sampled)
    return {
        "mean": measures["mean"],
        "std": measures["std"],
        "mad": measures["mad"],
        "energy": np.mean(sampled**2) if sampled.size else np.nan,
        "abs_p90": np.percentile(abs(sampled), 90) if sampled.size else np.nan,
    }


def multiscale(gray, mask):
    ys, xs = bounds(mask)
    gray, mask = gray[ys, xs].astype(np.float32) / 255, mask[ys, xs]
    depth = distance_inside(mask)
    out = {}
    for sigma in (1.0, 2.0, 3.0):
        radius = int(np.ceil(3 * sigma))
        response = ndi.gaussian_laplace(gray, sigma, radius=radius) * sigma**2
        out.update(prefix(f"log_s{int(sigma)}", filter_response(response, depth > np.sqrt(2) * radius)))
    for wavelength, sigma in ((3, 1.0), (6, 1.7), (10, 2.4)):
        radius = int(np.ceil(2 * sigma))
        valid = depth > np.sqrt(2) * radius
        for direction in range(4):
            theta = direction * np.pi / 4
            kernels = [
                cv2.getGaborKernel((2 * radius + 1,) * 2, sigma, theta, wavelength, 0.7, phase)
                for phase in (0, np.pi / 2)
            ]
            responses = [cv2.filter2D(gray, -1, kernel - kernel.mean()) for kernel in kernels]
            out.update(
                prefix(f"gabor_w{wavelength}_dir{direction}", filter_response(np.hypot(*responses), valid))
            )
    # Haar wavelet coefficients only from complete 2×2 or 4×4 blocks inside the mask.
    approximation, valid = gray, mask
    for level in (1, 2):
        h, w = (np.asarray(approximation.shape) // 2) * 2
        if h and w:
            source = approximation[:h, :w]
            approximation, details = pywt.dwt2(source, "haar")
            valid = valid[:h, :w].reshape(h // 2, 2, w // 2, 2).all(axis=(1, 3))
        else:
            details = [np.empty((0, 0))] * 3
            valid = np.empty((0, 0), dtype=bool)
        for band, response in zip(("horizontal", "vertical", "diagonal"), details):
            out.update(prefix(f"haar_l{level}_{band}", filter_response(response, valid)))
    return out
