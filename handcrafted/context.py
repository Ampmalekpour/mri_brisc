"""F01–F03, F21–F22 and F25. Acquisition controls and tissue context."""

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin
from scipy import ndimage as ndi

import config as C
from data import PLANES, bounds, distance_inside, distance_to_zeros, resolve
from feature_utils import entropy, histogram, prefix, ratio, region_intensity
from texture import lbp_features, radiomics


def acquisition(row, rgb):
    path = resolve(row.path)
    with Image.open(path) as image:
        tables = getattr(image, "quantization", {})
        try:
            sampling = JpegImagePlugin.get_sampling(image) if image.format == "JPEG" else np.nan
        except (AttributeError, ValueError):
            sampling = np.nan
        file_size = path.stat().st_size
        height, width = rgb.shape[:2]
        red, green, blue = rgb.transpose(2, 0, 1).astype(np.int16)
        table = np.asarray(tables.get(0, []), dtype=float)
        return dict(
            width=width,
            height=height,
            aspect=width / height,
            bytes=file_size,
            bits_per_pixel=8 * file_size / (width * height),
            is_jpeg=image.format == "JPEG",
            jpeg_subsampling=sampling if sampling != -1 else np.nan,
            jpeg_table_count=len(tables),
            jpeg_luma_mean=table.mean() if len(table) else np.nan,
            jpeg_luma_dc=table[0] if len(table) else np.nan,
            jpeg_luma_ac_mean=table[1:].mean() if len(table) > 1 else np.nan,
            native_channels=len(image.getbands()),
            colored_pixel_fraction=((red != green) | (red != blue)).mean(),
            mean_channel_difference=(abs(red - green) + abs(red - blue) + abs(green - blue)).mean() / 3,
        )


def background(regions):
    gray, head = regions.gray, regions.head
    h, w = gray.shape
    y, x = np.nonzero(head)
    result = dict(black_fraction=(gray <= 3).mean(), foreground_fraction=head.mean())
    if x.size:
        result.update(
            pad_top=y.min() / h,
            pad_bottom=(h - 1 - y.max()) / h,
            pad_left=x.min() / w,
            pad_right=(w - 1 - x.max()) / w,
            head_width=np.ptp(x) / w,
            head_height=np.ptp(y) / h,
            head_aspect=(np.ptp(x) + 1) / (np.ptp(y) + 1),
            head_cx=x.mean() / w,
            head_cy=y.mean() / h,
            head_touches_frame=float(x.min() == 0 or y.min() == 0 or x.max() == w - 1 or y.max() == h - 1),
        )
    outside = ~head
    result.update(prefix("outside_head", region_intensity(gray, outside)))
    # Keep all texture samples outside the head, without manufacturing a black image.
    gray_float = gray.astype(float)
    iy, ix = np.ogrid[-2:3, -2:3]
    disk = (iy**2 + ix**2 <= 4).astype(np.uint8)
    support = cv2.erode(outside.astype(np.uint8), disk, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(
        bool
    )
    laplacian = cv2.Laplacian(gray_float, -1)
    residual = gray_float - cv2.medianBlur(gray, 3)
    result["outside_laplacian_std"] = laplacian[support].std() if support.any() else np.nan
    result["outside_residual_mad"] = np.median(abs(residual[support])) if support.any() else np.nan
    markings = outside & (gray >= 80)
    for name, part in (("upper", markings[: h // 2]), ("lower", markings[h // 2 :])):
        result[f"{name}_outside_bright_pixel_fraction"] = part.sum() / gray.size
    for axis in (0, 1):
        values = np.moveaxis(gray_float, axis, 0)
        roi = np.moveaxis(outside, axis, 0)
        diff = abs(np.diff(values, axis=0))
        valid = roi[1:] & roi[:-1]
        edge = (np.arange(1, len(values)) % 8 == 0)[:, None]
        on_block, off_block = diff[valid & edge], diff[valid & ~edge]
        result[f"outside_jpeg_block_excess_axis{axis}"] = (
            on_block.mean() - off_block.mean() if on_block.size and off_block.size else np.nan
        )
    return result


def peritumoral(regions):
    gray, mask = regions.gray, regions.mask
    image = gray.astype(float)
    local = ndi.uniform_filter(image, 5)
    variance = np.maximum(0, ndi.uniform_filter(image**2, 5) - local**2)
    sobel = np.hypot(ndi.sobel(image, axis=0), ndi.sobel(image, axis=1)) / 8
    tumor_mean = gray[mask].mean()
    out = {}
    yy, xx = np.indices(gray.shape)
    y, x = np.nonzero(mask)
    quadrants = 2 * (yy >= y.mean()) + (xx >= x.mean())
    edges = C.RING_EDGES
    for restrict in (False, True):
        tag = "head" if restrict else "unrestricted"
        means = []
        for low, high in zip(edges, edges[1:]):
            raw = regions.ring(low, high)
            support = raw & regions.head if restrict else raw
            values = gray[support]
            measured = region_intensity(gray, support)
            means.append(measured["mean"])
            valid_gradient = distance_inside(support) > np.sqrt(2)
            valid_variance = distance_inside(support) > 2 * np.sqrt(2)
            measured.update(
                tumor_minus_ring=tumor_mean - measured["mean"],
                tumor_to_ring=ratio(tumor_mean, measured["mean"]),
                head_coverage=ratio((raw & regions.head).sum(), raw.sum()),
                gradient_energy=np.mean(sobel[valid_gradient] ** 2) if valid_gradient.any() else np.nan,
                strong_edges=(sobel[valid_gradient] > 12).mean() if valid_gradient.any() else np.nan,
                local_std=np.sqrt(variance[valid_variance]).mean() if valid_variance.any() else np.nan,
            )
            for quadrant in range(4):
                part = values[quadrants[support] == quadrant]
                measured[f"quadrant_{quadrant}_brightness"] = part.mean() if part.size else np.nan
            measured.update(radiomics(gray, support, C.TEXTURE_BINS, short=True))
            measured.update(lbp_features(gray, support, radii=(1,)))
            out.update(prefix(f"{tag}_ring_{low:02}_{high:02}", measured))
        midpoints = (np.asarray(edges[1:]) + edges[:-1]) / 2
        valid = np.isfinite(means)
        out[f"{tag}_peripheral_intensity_slope"] = (
            np.polyfit(midpoints[valid], np.asarray(means)[valid], 1)[0] if valid.sum() > 1 else np.nan
        )
    return out


def anatomical_relationships(regions):
    mask, head, gray = regions.mask, regions.head, regions.gray
    if not head.any():
        return {}
    hy, hx = bounds(head)
    center_x = (hx.start + hx.stop - 1) / 2
    inside = distance_inside(head)
    signed = inside - distance_to_zeros(~head)
    contour = mask & ~ndi.binary_erosion(mask)
    near = abs(signed[contour])
    y, x = np.nonzero(mask)
    reflected = np.rint(2 * center_x - x).astype(int)
    valid = np.clip(reflected, 0, gray.shape[1] - 1)
    usable = (valid == reflected) & head[y, valid] & (regions.outside[y, valid] > 7 * regions.scale)
    opposite = gray[y[usable], valid[usable]].astype(float)
    selected = gray[y[usable], x[usable]].astype(float)
    h, w = hy.stop - hy.start, hx.stop - hx.start
    ring = regions.ring(3, 14)
    ry, rx = np.nonzero(ring)
    angle = (np.arctan2(ry - y.mean(), rx - x.mean()) + 2 * np.pi) % (2 * np.pi)
    out = dict(
        outside_head_fraction=(~head[mask]).mean(),
        crosses_head_box_midline=float(x.min() <= center_x <= x.max()),
        midline_strip_fraction=(abs(x - center_x) <= 0.025 * w).mean(),
        peripheral_head_fraction=(inside[mask] <= 0.03 * min(h, w)).mean(),
        contour_near_head_03px=(near <= 3 * regions.scale).mean(),
        contour_near_head_07px=(near <= 7 * regions.scale).mean(),
        mirrored_pairs_fraction=usable.mean(),
        left_right_head_area_asymmetry=abs(head[:, : int(center_x)].sum() - head[:, int(center_x) :].sum())
        / head.sum(),
        mirrored_mean_difference=(selected - opposite).mean() if opposite.size else np.nan,
        mirrored_std_difference=selected.std() - opposite.std() if opposite.size else np.nan,
        mirrored_entropy_difference=entropy(histogram(selected)) - entropy(histogram(opposite)),
    )
    for sector in range(8):
        selected = (angle >= sector * np.pi / 4) & (angle < (sector + 1) * np.pi / 4)
        out[f"sector{sector}_head_fraction"] = (
            head[ry[selected], rx[selected]].mean() if selected.any() else np.nan
        )
    return out


def distant_tissue(regions, seed):
    gray, head = regions.gray, regions.head
    if not head.any():
        return {}
    hy, hx = bounds(head)
    # Pool a fixed count of complete patches from a regular head grid. Mask and patch
    # locations are not included among the model inputs.
    x_edges = np.linspace(hx.start, hx.stop, 9, dtype=int)
    y_edges = np.linspace(hy.start, hy.stop, 9, dtype=int)
    safe = head & (regions.outside > C.RING_EDGES[-1] * regions.scale)
    patches = []
    for y0, y1 in zip(y_edges, y_edges[1:]):
        for x0, x1 in zip(x_edges, x_edges[1:]):
            pick = np.s_[y0:y1, x0:x1]
            if y1 - y0 >= 4 and x1 - x0 >= 4 and safe[pick].all():
                patches.append(gray[pick])
    if len(patches) < 8:
        return {}
    selected = np.random.default_rng(seed).choice(len(patches), 8, replace=False)
    records = []
    for index in selected:
        patch = cv2.resize(patches[index], (16, 16), interpolation=cv2.INTER_AREA)
        roi = np.ones(patch.shape, dtype=bool)
        values = region_intensity(patch, roi)
        values.update(radiomics(patch, roi, C.TEXTURE_BINS, short=True))
        values.update(lbp_features(patch, roi, radii=(1,)))
        records.append(values)
    return {
        f"{key}_{summary}": operation([row[key] for row in records])
        for key in records[0]
        for summary, operation in (("median", np.median), ("sd", np.std))
    }


def image_controls(row, rgb, regions):
    return {
        "F01": acquisition(row, rgb),
        "F02": background(regions),
        "F03": {name.lower(): float(row.plane == name) if row.plane in PLANES else np.nan for name in PLANES},
    }


def extract(regions):
    if not regions.mask.any():
        return {}
    return {
        "F21": peritumoral(regions),
        "F22": anatomical_relationships(regions),
        "F25": distant_tissue(regions, C.SEED),
    }
