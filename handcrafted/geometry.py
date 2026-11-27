"""F04–F11. Lesion position, size, shape, topology and occupancy."""

import cv2
import mahotas
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from skimage import measure, morphology

from data import bounds, distance_inside, distance_to_zeros
from feature_utils import EPS, area_resize, centered_view, prefix, ratio, stats, vector


def image_location(mask, x, y):
    h, w = mask.shape
    cx, cy = x.mean(), y.mean()
    return dict(
        centroid_x=cx,
        centroid_y=cy,
        cx_norm=cx / w,
        cy_norm=cy / h,
        left=x.min() / w,
        right=(w - 1 - x.max()) / w,
        top=y.min() / h,
        bottom=(h - 1 - y.max()) / h,
        rightmost=x.max() / w,
        bottommost=y.max() / h,
        center_distance=np.hypot(cx / w - 0.5, cy / h - 0.5),
        **vector(
            "quadrant_tumor_fraction", np.bincount((y >= h / 2) * 2 + (x >= w / 2), minlength=4) / len(x)
        ),
    )


def anatomical_location(regions, x, y):
    head = regions.head
    if not head.any():
        return {}
    hy, hx = bounds(head)
    width, height = hx.stop - hx.start, hy.stop - hy.start
    cx, cy = (x.mean() - hx.start) / width, (y.mean() - hy.start) / height
    radius = np.sqrt(head.sum() / np.pi)
    depth = distance_inside(head)[y, x] / radius
    overlap = head[y, x]
    boundary_distance = depth.copy()
    if not overlap.all():
        outside = distance_to_zeros(~head)
        boundary_distance[~overlap] = outside[y[~overlap], x[~overlap]] / radius
    angle = np.arctan2(cy - 0.5, cx - 0.5)
    return dict(
        head_relative_x=cx,
        head_relative_y=cy,
        relative_center_distance=np.hypot(cx - 0.5, cy - 0.5),
        head_box_midline_distance=abs(cx - 0.5),
        angle_sin=np.sin(angle),
        angle_cos=np.cos(angle),
        overlap_fraction=overlap.mean(),
        min_depth=depth.min(),
        mean_depth=depth.mean(),
        max_depth=depth.max(),
        median_depth=np.median(depth),
        min_boundary_distance=boundary_distance.min(),
        mean_boundary_distance=boundary_distance.mean(),
    )


def feret(hull):
    if len(hull) < 2:
        return 0.0, 0.0
    farthest = pdist(hull).max()
    if len(hull) == 2:
        return farthest, 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    normals = np.column_stack((-edges[:, 1], edges[:, 0])).astype(float)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    projections = hull @ normals.T
    return farthest, np.ptp(projections, axis=0).min()


def size_features(regions, region, hull):
    mask, head = regions.mask, regions.head
    minimum_y, minimum_x, maximum_y, maximum_x = region.bbox
    h, w = maximum_y - minimum_y, maximum_x - minimum_x
    maximum, minimum = feret(hull)
    head_h, head_w = (np.nan, np.nan)
    if head.any():
        hy, hx = bounds(head)
        head_h, head_w = hy.stop - hy.start, hx.stop - hx.start
    return dict(
        area_px=region.area,
        area_image_fraction=region.area / mask.size,
        area_head_fraction=ratio(region.area, head.sum()),
        width_px=w,
        height_px=h,
        box_area_px=w * h,
        equivalent_diameter_px=region.equivalent_diameter_area,
        max_feret_px=maximum,
        min_feret_px=minimum,
        major_axis_px=region.axis_major_length,
        minor_axis_px=region.axis_minor_length,
        max_inscribed_radius_px=regions.inside.max(),
        mean_inscribed_radius_px=regions.inside[mask].mean(),
        width_to_head=ratio(w, head_w),
        height_to_head=ratio(h, head_h),
        width_image_fraction=w / mask.shape[1],
        height_image_fraction=h / mask.shape[0],
    )


def basic_shape(region, hull, contour_perimeter, maximum, minimum):
    perimeter = region.perimeter_crofton
    hull_perimeter = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    return dict(
        circularity=ratio(4 * np.pi * region.area, perimeter**2),
        compactness=ratio(perimeter**2, 4 * np.pi * region.area),
        solidity=region.solidity,
        extent=region.extent,
        eccentricity=region.eccentricity,
        elongation=ratio(region.axis_major_length, region.axis_minor_length),
        convexity=ratio(hull_perimeter, contour_perimeter),
        feret_ratio=ratio(minimum, maximum),
        box_aspect_ratio=ratio(region.image.shape[1], region.image.shape[0]),
        orientation_sin=np.sin(2 * region.orientation),
        orientation_cos=np.cos(2 * region.orientation),
    )


def normalized_contour(contour, count=128):
    points = contour.reshape(-1, 2).astype(float)
    if len(points) < 3:
        return None
    closed = np.vstack([points, points[0]])
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    t = np.linspace(0, cumulative[-1], count, endpoint=False)
    samples = np.column_stack([np.interp(t, cumulative, closed[:, dim]) for dim in (0, 1)])
    samples -= samples.mean(axis=0)
    return samples / np.sqrt(np.mean(np.sum(samples**2, axis=1)))


def advanced_shape(normalized, contour):
    hu = cv2.HuMoments(cv2.moments(normalized.astype(np.uint8), binaryImage=True)).ravel()
    centered = normalized.astype(np.uint8)
    center = ndi.center_of_mass(centered)
    zernike = mahotas.features.zernike_moments(centered, centered.shape[0] * 0.48, degree=8, cm=center)
    coefficients = np.full(16, np.nan)
    if contour is not None:
        spectrum = np.abs(np.fft.fft(contour[:, 0] + 1j * contour[:, 1]))
        coefficients = (spectrum[1:17] + spectrum[-1:-17:-1]) / len(contour)
    return {
        **vector("hu_signed_log", -np.sign(hu) * np.log10(np.maximum(np.abs(hu), EPS))),
        **vector("zernike", zernike),
        **vector("fourier", coefficients),
    }


def boundary_geometry(contour, perimeter, area):
    if contour is None:
        return {}
    smooth = ndi.gaussian_filter1d(contour, 2, axis=0, mode="wrap")
    tangent = np.roll(smooth, -2, axis=0) - np.roll(smooth, 2, axis=0)
    angle = np.arctan2(tangent[:, 1], tangent[:, 0])
    turn = (np.roll(angle, -1) - angle + np.pi) % (2 * np.pi) - np.pi
    radial = np.linalg.norm(smooth, axis=1)
    radial_spectrum = np.abs(np.fft.rfft(radial))
    smooth_closed = np.vstack([smooth, smooth[0]])
    smooth_length = np.linalg.norm(np.diff(smooth_closed, axis=0), axis=1).sum()
    center_length = np.linalg.norm(np.diff(np.vstack([contour, contour[0]]), axis=0), axis=1).sum()
    return {
        **prefix("absolute_turn", stats(abs(turn))),
        **prefix("radius_normalized", stats(radial)),
        "strong_turn_fraction": (abs(turn) > 0.25).mean(),
        "perimeter_to_root_area": perimeter / np.sqrt(area),
        "rough_to_smooth_length": ratio(center_length, smooth_length),
        "radial_peaks": ((radial > np.roll(radial, 1)) & (radial > np.roll(radial, -1))).sum(),
        "radial_harmonics_3_to_8": radial_spectrum[3:9].sum() / max(radial_spectrum[0], EPS),
    }


def topology(mask):
    components = np.bincount(measure.label(mask, connectivity=2).ravel())[1:]
    holes = ndi.binary_fill_holes(mask) & ~mask
    hole_count = measure.label(holes, connectivity=1).max()
    skeleton = morphology.skeletonize(mask)
    neighbours = (
        ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    )
    return dict(
        components=len(components),
        holes=hole_count,
        euler_number=len(components) - hole_count,
        largest_component_fraction=components.max() / mask.sum(),
        hole_area_fraction=holes.sum() / mask.sum(),
        skeleton_fraction=skeleton.sum() / mask.sum(),
        skeleton_length_scaled=skeleton.sum() / np.sqrt(mask.sum()),
        skeleton_endpoints=(skeleton & (neighbours == 1)).sum(),
        skeleton_branch_pixels=(skeleton & (neighbours >= 3)).sum(),
    )


def occupancy(mask):
    features = {}
    for size in (3, 5, 8, 16):
        grid = area_resize(mask, size)
        features.update(vector(f"image_cell_occupancy_{size}", grid))
        features.update(vector(f"image_tumor_fraction_{size}", grid / grid.sum()))
    _, centered = centered_view(mask, mask, size=64)
    features.update(vector("centered_mask_16", area_resize(centered, 16)))
    distance = distance_inside(centered)
    features.update(vector("centered_distance_8", area_resize(distance / distance.max(), 8)))
    return features


def extract(regions):
    mask = regions.mask
    if not mask.any():
        return {}
    y, x = np.nonzero(mask)
    region = measure.regionprops(mask.astype(np.uint8))[0]
    ys, xs = bounds(mask)
    contours, _ = cv2.findContours(mask[ys, xs].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.concatenate(contours)).reshape(-1, 2)
    major_contour = max(contours, key=cv2.contourArea)
    contour_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    contour = normalized_contour(major_contour)
    sizes = size_features(regions, region, hull)
    _, centered = centered_view(mask, mask, size=64)
    return {
        "F04": image_location(mask, x, y),
        "F05": anatomical_location(regions, x, y),
        "F06": sizes,
        "F07": basic_shape(region, hull, contour_perimeter, sizes["max_feret_px"], sizes["min_feret_px"]),
        "F08": advanced_shape(centered, contour),
        "F09": boundary_geometry(contour, contour_perimeter, region.area),
        "F10": topology(mask[ys, xs]),
        "F11": occupancy(mask),
    }
