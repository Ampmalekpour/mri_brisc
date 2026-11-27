"""Run one decomposition stage, print summaries and save auditable feature tables."""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion

import config as C
from data import (
    feature_settings,
    load_manifest,
    manifest_digest,
    output_path,
    sample_arrays,
    save_csv,
    save_json,
)
from feature_utils import Regions


GROUPS = {
    "F01": ("Acquisition metadata", ("width", "height", "bits_per_pixel")),
    "F02": ("Image background", ("black_fraction", "foreground_fraction", "pad_top")),
    "F03": ("Imaging plane", ("axial", "coronal", "sagittal")),
    "F04": ("Image-relative position", ("cx_norm", "cy_norm", "center_distance")),
    "F05": ("Head-relative position", ("head_relative_x", "head_relative_y", "mean_depth")),
    "F06": ("Lesion size", ("area_px", "area_image_fraction", "max_feret_px")),
    "F07": ("Basic shape", ("circularity", "solidity", "eccentricity")),
    "F08": ("Advanced shape", ("hu_signed_log_000", "zernike_002", "fourier_001")),
    "F09": ("Boundary geometry", ("perimeter_to_root_area", "rough_to_smooth_length", "radial_peaks")),
    "F10": ("Topology", ("components", "holes", "skeleton_endpoints")),
    "F11": ("Mask spatial representation", ("image_cell_occupancy_3_004", "image_tumor_fraction_3_004")),
    "F12": ("Raw lesion intensities", ("median", "std", "entropy")),
    "F13": ("Reference-normalized intensities", ("distant_robust_z_median", "local_median_ratio")),
    "F14": ("ROI histogram", ()),
    "F15": ("Radiomic texture", ("glcm_d1_contrast_mean", "glszm_entropy", "ngtdm_coarseness")),
    "F16": ("Local binary patterns", ()),
    "F17": ("Multiscale texture", ("log_s1_energy", "haar_l1_diagonal_std", "gabor_w6_dir0_energy")),
    "F18": ("Internal compartments", ("bright_fraction", "dark_fraction", "bright_minus_dark_mean")),
    "F19": ("Radial architecture", ("rim_minus_core", "intensity_slope", "bright_rim_coverage")),
    "F20": ("Boundary appearance", ("gradient_median", "normal_contrast_median", "mean_canny_distance_px")),
    "F21": ("Peritumoral rings", ("head_ring_00_03_mean", "unrestricted_ring_03_07_mean")),
    "F22": (
        "Anatomical relationships",
        ("outside_head_fraction", "peripheral_head_fraction", "mirrored_mean_difference"),
    ),
    "F23": ("Classical visual descriptors", ()),
    "F24": ("Whole MRI", ("median", "entropy", "head_median")),
    "F25": ("Distant tissue", ("mean_median", "entropy_median", "glrlm_entropy_median")),
}


def save_group(records, group, manifest):
    table = pd.DataFrame.from_records(records).set_index("sample_id").reindex(manifest.sample_id)
    table = table.sort_index(axis=1).replace([np.inf, -np.inf], np.nan)
    save_csv(table.reset_index(), "features", f"{group}.csv")
    title, desired = GROUPS[group[:3]]
    if group.startswith("F15_bins"):
        title += f" ({group.removeprefix('F15_bins')} gray levels)"
    if not len(table.columns):
        print(f"  {group} | {title} | 0 columns. No usable regions in this stage.")
        return
    coverage = table.notna().sum()
    catalog = pd.DataFrame(
        dict(
            feature=table.columns,
            present=coverage,
            missing=len(table) - coverage,
            minimum=table.min(),
            median=table.median(),
            maximum=table.max(),
        )
    )
    save_csv(catalog, "features", "summaries", f"{group}.csv")
    features = table.loc[:, coverage > 0]
    defined = features.notna().any(axis=1).sum()
    print(f"  {group} | {title} | {table.shape[1]:,} columns, {defined:,} images with measurements")
    if not features.empty:
        example_columns = [name for name in desired if name in features]
        first = features[example_columns] if example_columns else features.iloc[:, :3]
        medians = first.groupby(manifest.set_index("sample_id").label).median()
        save_csv(medians.reset_index(), "features", "summaries", f"{group}_example_by_class.csv")
        examples = ", ".join(f"{column}={features[column].median():.4g}" for column in first)
        print(f"             Example medians: {examples}")


def save_qc(row, gray, regions, stage):
    array = np.repeat(gray[..., None], 3, axis=2).astype(float)
    if regions.mask.any():
        mask = regions.mask
        if stage == 4:
            colors = [(250, 130, 40), (245, 205, 60), (65, 205, 220), (120, 100, 250)]
            for color, low, high in zip(colors, C.RING_EDGES, C.RING_EDGES[1:]):
                ring = regions.ring(low, high)
                array[ring] = 0.55 * array[ring] + 0.45 * np.array(color)
            array[mask] = 0.55 * array[mask] + 0.45 * np.array([255, 75, 75])
        else:
            relative_depth = regions.inside / regions.inside.max()
            colors = [(255, 80, 80), (255, 155, 55), (225, 210, 60), (90, 200, 100)]
            for index, color in enumerate(colors):
                band = mask & (relative_depth > index / 4) & (relative_depth <= (index + 1) / 4)
                array[band] = 0.55 * array[band] + 0.45 * np.array(color)
    array[regions.head & ~binary_erosion(regions.head)] = [50, 220, 245]
    side_by_side = np.concatenate([np.repeat(gray[..., None], 3, axis=2), array.astype(np.uint8)], axis=1)
    safe_id = row.sample_id.replace("/", "__").replace("\\", "__")
    Image.fromarray(side_by_side).save(output_path("qc", f"step{stage}", f"{safe_id}.png"))


def run_stage(title, group_ids, extract_row, stage, save_patches=False):
    cv2.setNumThreads(1)
    manifest = load_manifest()
    records = {group: [] for group in group_ids}
    descriptor_records = {}
    qc_count = {}
    print(f"{title} | {len(manifest):,} indexed images", flush=True)
    for number, row in enumerate(manifest.itertuples(index=False), 1):
        rgb, gray, mask = sample_arrays(row)
        regions = Regions(gray, mask)
        groups, patches = extract_row(row, rgb, regions)
        for group in records:
            values = groups.get(group, {})
            records[group].append(dict(sample_id=row.sample_id, **values))
        if save_patches and patches is not None and len(patches):
            descriptor_records[row.sample_id] = patches
        key = row.label
        suitable = row.mask_usable or row.label == "no_tumor"
        if row.split == C.TRAIN_SPLIT and suitable and qc_count.get(key, 0) < C.QC_IMAGES_PER_CLASS:
            save_qc(row, gray, regions, stage)
            qc_count[key] = qc_count.get(key, 0) + 1
        if number % 100 == 0 or number == len(manifest):
            print(f"  Processed {number:,}/{len(manifest):,}", flush=True)
    for group, rows in records.items():
        save_group(rows, group, manifest)
    if save_patches:
        path = output_path("features", "F23_patches.npz")
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **descriptor_records)
        temporary.replace(path)
        print(f"  F23 local descriptors available for {len(descriptor_records):,} images.")
    save_json(
        dict(manifest_sha256=manifest_digest(), groups=list(group_ids), settings=feature_settings(stage)),
        "features",
        f"step{stage}.json",
    )
    print(f"Saved feature tables and summaries in {output_path('features', 'example').parent}")
