"""Optional Step 8. Measure descriptor sensitivity to small mask and image changes."""

from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

import appearance
import config as C
import context
import geometry
from data import load_manifest, sample_arrays, save_csv
from feature_utils import Regions


def perturbations(gray, mask):
    height, width = gray.shape
    steps = max(1, round(min(height, width) / 256))
    yield "mask_erosion", gray, ndi.binary_erosion(mask, iterations=steps)
    yield "mask_dilation", gray, ndi.binary_dilation(mask, iterations=steps)
    yield "mask_smoothing", gray, ndi.gaussian_filter(mask.astype(float), steps) >= 0.5
    with BytesIO() as buffer:
        Image.fromarray(gray).save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        with Image.open(buffer) as image:
            compressed = np.asarray(image).copy()
    yield "jpeg_quality_70", compressed, mask
    changed = np.clip(gray.astype(float) * 0.9 + 8, 0, 255).astype(np.uint8)
    yield "intensity_gain_and_offset", changed, mask

    reduced = (max(1, int(width * 0.65)), max(1, int(height * 0.65)))
    small = cv2.resize(gray, reduced, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), reduced, interpolation=cv2.INTER_NEAREST)
    restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    restored_mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    yield "resolution_roundtrip", restored, restored_mask


def measurements(gray, mask, selected):
    regions = Regions(gray, mask)
    groups = {}
    if selected & {f"F{i:02}" for i in range(4, 12)}:
        groups.update(geometry.extract(regions))
    if selected & {f"F{i:02}" for i in range(12, 21)}:
        groups.update(appearance.extract(regions, sensitivity=False))
    if selected & {"F21", "F22", "F25"}:
        groups.update(context.extract(regions))
    return {name: values for name, values in groups.items() if name in selected}


def main():
    chosen = set(C.ROBUSTNESS_GROUPS)
    supported = {f"F{i:02}" for i in range(4, 23)} | {"F25"}
    if not chosen or chosen - supported or C.ROBUSTNESS_IMAGES_PER_CLASS < 1:
        raise ValueError("Use groups F04–F22 or F25 and a positive sample count.")
    manifest = load_manifest()
    train = manifest[manifest.split.eq(C.TRAIN_SPLIT) & manifest.mask_usable]
    if train.empty:
        raise ValueError("No usable training masks for the robustness study.")
    rng = np.random.default_rng(C.SEED)
    selected = pd.concat(
        [
            group.sample(min(len(group), C.ROBUSTNESS_IMAGES_PER_CLASS), random_state=rng)
            for _, group in train.groupby("label")
        ]
    )
    print(f"Step 8 | Sensitivity checks on {len(selected)} training images", flush=True)
    rows, unmeasurable = [], []
    for row in selected.itertuples(index=False):
        _, gray, mask = sample_arrays(row)
        baseline = measurements(gray, mask, chosen)
        for variant, altered_gray, altered_mask in perturbations(gray, mask):
            values = measurements(altered_gray, altered_mask, chosen)
            if not altered_mask.any():
                unmeasurable.append(dict(sample_id=row.sample_id, perturbation=variant, reason="empty_mask"))
            for group, features in baseline.items():
                changed = values.get(group, {})
                for feature, original in features.items():
                    perturbed = changed.get(feature, np.nan)
                    rows.append(
                        dict(
                            sample_id=row.sample_id,
                            label=row.label,
                            perturbation=variant,
                            group=group,
                            feature=feature,
                            baseline=original,
                            perturbed=perturbed,
                            absolute_change=abs(perturbed - original),
                        )
                    )
    raw = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    baselines = raw[["sample_id", "group", "feature", "baseline"]].drop_duplicates()
    spread = (
        baselines.dropna(subset=["baseline"])
        .groupby(["group", "feature"])
        .baseline.agg(
            iqr=lambda values: values.quantile(0.75) - values.quantile(0.25),
            median_magnitude=lambda values: values.abs().median(),
        )
    )
    scale = spread[["iqr", "median_magnitude"]].mul([1, 0.05]).max(axis=1)
    spread["normalization"] = scale.where(scale > 1e-8)
    raw = raw.merge(spread[["normalization"]].reset_index(), on=["group", "feature"], how="left")
    raw["scaled_change"] = raw.absolute_change / raw.normalization
    summary = (
        raw.groupby(["group", "feature", "perturbation"])
        .agg(
            pairs=("sample_id", "size"),
            valid_pairs=("scaled_change", "count"),
            median_absolute_change=("absolute_change", "median"),
            median_scaled_change=("scaled_change", "median"),
            p90_scaled_change=("scaled_change", lambda values: values.quantile(0.9)),
        )
        .reset_index()
    )
    summary = summary.sort_values("median_scaled_change", ascending=False)
    save_csv(raw, "robustness", "per_image_changes.csv")
    save_csv(
        pd.DataFrame(unmeasurable, columns=["sample_id", "perturbation", "reason"]),
        "robustness",
        "unmeasurable.csv",
    )
    path = save_csv(summary, "robustness", "feature_sensitivity.csv")
    print(
        summary[["group", "feature", "perturbation", "median_scaled_change"]].head(10).to_string(index=False)
    )
    print(f"Saved per-image changes and feature sensitivity in {path.parent}")


if __name__ == "__main__":
    main()
