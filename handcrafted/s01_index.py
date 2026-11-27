"""Step 1. Index the original splits, match masks and group likely duplicate images."""

from collections import Counter, defaultdict
from hashlib import sha256

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config as C
from data import grayscale, plane_from_filename, read_image, read_mask, save_csv, save_json


def image_files(directory):
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in C.IMAGE_EXTENSIONS
    )


def key(path):
    stem = path.stem.casefold()
    return stem.removesuffix("_mask").removesuffix("-mask")


def normalized_class(name):
    return {"notumor": "no_tumor", "no tumor": "no_tumor", "no-tumor": "no_tumor"}.get(
        name.casefold(), name.casefold()
    )


def mask_index():
    by_name = defaultdict(list)
    root = C.DATA_ROOT / "segmentation_task"
    for path in image_files(root):
        folders = path.relative_to(root).parts[:-1]
        if any("mask" in folder.casefold() for folder in folders):
            by_name[key(path)].append((folders[0], path))
    return by_name


def match_mask(image_path, split, label, shape, masks):
    if label not in C.TUMOR_CLASSES:
        return "", "not_applicable", False
    matches = masks.get(key(image_path), [])
    same_split = [path for folder, path in matches if folder.casefold() == split.casefold()]
    candidates = same_split or [path for _, path in matches]
    if len(candidates) > 1:
        # Some distributions copy an identical mask into multiple split folders.
        if len({sha256(path.read_bytes()).digest() for path in candidates}) != 1:
            return "", "ambiguous", False
    if not candidates:
        return "", "missing", False
    path = candidates[0]
    relative = path.relative_to(C.DATA_ROOT).as_posix()
    try:
        with Image.open(path) as image:
            width, height = image.size
        h, w = shape
        if abs(width / height - w / h) / (w / h) > 0.015:
            return relative, "aspect_mismatch", False
        if not read_mask(path, shape).any():
            return relative, "empty", False
    except (OSError, ValueError):
        return relative, "unreadable", False
    status = "matched" if (height, width) == shape else "resized"
    return relative, status if same_split else f"{status}_cross_split", True


def scan():
    masks = mask_index()
    rows, hashes = [], []
    for split in (C.TRAIN_SPLIT, C.VAL_SPLIT, C.TEST_SPLIT):
        folder = C.DATA_ROOT / "classification_task" / split
        print(f"Reading split {split!r} from {folder}", flush=True)
        for path in image_files(folder):
            parts = path.relative_to(folder).parts
            if len(parts) < 2:
                raise ValueError(f"Expected a class folder under {folder}: {path}")
            label = normalized_class(parts[0])
            if label not in C.ALL_CLASSES:
                raise ValueError(f"Unknown BRISC class {parts[0]!r} in {path}")
            gray = grayscale(read_image(path))
            relative = path.relative_to(C.DATA_ROOT).as_posix()
            mask_path, status, usable = match_mask(path, split, label, gray.shape, masks)
            thumb = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
            hashes.append((thumb[:, 1:] > thumb[:, :-1]).ravel())
            rows.append(
                dict(
                    sample_id=relative,
                    path=relative,
                    split=split,
                    label=label,
                    plane=plane_from_filename(path),
                    mask_path=mask_path,
                    mask_status=status,
                    mask_usable=usable,
                    height=gray.shape[0],
                    width=gray.shape[1],
                    sha256=sha256(path.read_bytes()).hexdigest(),
                    mask_sha256=sha256((C.DATA_ROOT / mask_path).read_bytes()).hexdigest() if usable else "",
                )
            )
    if not rows:
        raise FileNotFoundError(
            f"Set DATA_ROOT to the folder containing classification_task and segmentation_task. Current value: {C.DATA_ROOT}"
        )
    rows = pd.DataFrame(rows)
    missing = {C.TRAIN_SPLIT, C.TEST_SPLIT} - set(rows.split)
    if missing:
        raise ValueError(f"Required split folders have no images: {sorted(missing)}")
    return rows, np.asarray(hashes, dtype=np.float32)


def make_groups(rows, hashes):
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        a, b = root(left), root(right)
        parent[max(a, b)] = min(a, b)

    rows["patient_id"] = ""
    if C.PATIENT_CSV:
        patients = pd.read_csv(C.PATIENT_CSV, dtype=str, keep_default_na=False)
        ids = patients["sample_id"].str.strip().str.replace("\\", "/", regex=False)
        if ids.duplicated().any():
            raise ValueError("PATIENT_CSV contains duplicate sample_id values.")
        mapping = dict(zip(ids, patients["patient_id"].str.strip()))
        if not set(mapping) & set(rows.sample_id):
            raise ValueError("PATIENT_CSV has no sample IDs matching the manifest.")
        unknown = len(set(mapping) - set(rows.sample_id))
        print(f"Patient table contains {unknown} sample IDs outside the indexed splits.")
        rows["patient_id"] = rows.sample_id.map(mapping).fillna("")

    for column in ("sha256", "patient_id"):
        for _, indices in rows.loc[rows[column].ne("")].groupby(column, sort=False).groups.items():
            for index in indices[1:]:
                join(int(indices[0]), int(index))

    # Block the all-pairs search to limit memory. +/-1 dot product gives the Hamming distance.
    signed = 2 * hashes - 1
    pair_counts = Counter()
    splits = rows.split.to_numpy()
    names = rows.sample_id.to_numpy()
    labels = rows.label.to_numpy()
    pairs = []
    for first in range(0, len(rows), 256):
        distances = ((signed.shape[1] - signed[first : first + 256] @ signed.T) * 0.5).astype(np.int16)
        for offset, left in enumerate(range(first, min(first + 256, len(rows)))):
            candidates = np.flatnonzero(distances[offset, left + 1 :] <= C.DHASH_MAX_DISTANCE)
            for right in candidates + left + 1:
                join(left, int(right))
                pair_counts[tuple(sorted((splits[left], splits[right])))] += 1
                if splits[left] != splits[right] or labels[left] != labels[right]:
                    pairs.append(
                        (
                            names[left],
                            names[right],
                            splits[left],
                            splits[right],
                            int(distances[offset, right]),
                            labels[left] == labels[right],
                        )
                    )
    rows["group"] = [f"g{root(i):06}" for i in range(len(rows))]
    overlap = rows.loc[rows.split.eq(C.TRAIN_SPLIT), "group"]
    rows["train_overlap"] = rows.group.isin(overlap) & rows.split.ne(C.TRAIN_SPLIT)
    return pairs, pair_counts


def main():
    print(f"BRISC dataset root: {C.DATA_ROOT}", flush=True)
    if len({C.TRAIN_SPLIT, C.VAL_SPLIT, C.TEST_SPLIT}) < 3:
        raise ValueError("TRAIN_SPLIT, VAL_SPLIT and TEST_SPLIT must be distinct.")
    if not 0 <= C.DHASH_MAX_DISTANCE <= 256:
        raise ValueError("DHASH_MAX_DISTANCE must be between 0 and 256.")
    rows, hashes = scan()
    pairs, counts = make_groups(rows, hashes)
    path = save_csv(rows, "index", "manifest.csv")
    save_csv(
        pd.DataFrame(
            pairs, columns=["sample_a", "sample_b", "split_a", "split_b", "hamming_distance", "same_label"]
        ),
        "index",
        "cross_split_or_label_duplicate_pairs.csv",
    )
    save_csv(
        pd.DataFrame(
            [(a, b, n) for (a, b), n in sorted(counts.items())], columns=["split_a", "split_b", "pairs"]
        ),
        "index",
        "duplicate_counts.csv",
    )
    summary = rows.groupby(["split", "label", "mask_status", "train_overlap"]).size().rename("images")
    save_csv(summary.reset_index(), "index", "coverage.csv")
    groups = rows.groupby("group").agg(
        images=("sample_id", "size"),
        splits=("split", "nunique"),
        labels=("label", "nunique"),
        patients=("patient_id", lambda ids: ids[ids.ne("")].nunique()),
    )
    save_csv(groups.reset_index(), "index", "groups.csv")
    for column in ("plane", "width", "height"):
        save_csv(pd.crosstab(rows.label, rows[column]).reset_index(), "index", f"class_by_{column}.csv")
    test = rows.loc[rows.split.eq(C.TEST_SPLIT)]
    tumors = rows.label.isin(C.TUMOR_CLASSES)
    save_json(
        dict(
            train_split=C.TRAIN_SPLIT,
            val_split=C.VAL_SPLIT,
            test_split=C.TEST_SPLIT,
            dhash_max_distance=C.DHASH_MAX_DISTANCE,
            patient_ids_provided=int(rows.patient_id.ne("").sum()),
        ),
        "index",
        "settings.json",
    )
    print(pd.crosstab(rows.split, rows.label).to_string())
    print(f"Usable masks: {rows.loc[tumors, 'mask_usable'].sum():,} / {tumors.sum():,} tumor images.")
    print(f"Independent duplicate/patient groups indexed: {rows.group.nunique():,}.")
    print(
        f"Official test: {len(test):,} images. Leakage-reduced test: {(~test.train_overlap).sum():,} images."
    )
    print(f"Saved index and audit tables in {path.parent}")


if __name__ == "__main__":
    main()
