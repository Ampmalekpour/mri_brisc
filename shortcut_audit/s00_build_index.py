"""
STEP 0 - Build the master index and run data-hygiene checks.

Creates   outputs/index.csv  (one row per classification image: path, class, split, plane, mask_path,
                              image size, dHash)
Reports   outputs/results/hygiene_*.csv / hygiene.txt
   * how many tumor images have a matching segmentation mask
   * exact + near-duplicate images ACROSS splits (possible leakage; BRISC gives no patient id)
   * class x plane and class x image-size tables with chi-square tests (acquisition confounds)
Run:  python s00_build_index.py
"""
import numpy as np
import pandas as pd
import cv2
from scipy.stats import chi2_contingency
from tqdm import tqdm

import config as C
from common import parse_plane


def find_masks():
    """stem -> mask path. Masks are picked from folders whose path contains 'mask'."""
    table = {}
    for p in C.SEGMENTATION_DIR.rglob("*"):
        if p.suffix.lower() in C.IMAGE_EXTS and "mask" in str(p.relative_to(C.SEGMENTATION_DIR)).lower():
            table.setdefault(p.stem, str(p))
    return table


def dhash(path, size=16):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()      # 256 bits


def main():
    masks = find_masks()
    rows = []
    for split in (C.TRAIN_SPLIT, C.VAL_SPLIT, C.TEST_SPLIT):
        sdir = C.CLASSIFICATION_DIR / split
        if not sdir.exists():
            print(f"[info] split '{split}' not found - skipped")
            continue
        for cdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in C.IMAGE_EXTS:
                    rows.append(dict(path=str(f), split=split, **{"class": cdir.name},
                                     plane=parse_plane(f.name),
                                     mask_path=masks.get(f.stem, "")))
    if not rows:
        raise SystemExit(f"No images found under {C.CLASSIFICATION_DIR}. Check config.DATA_ROOT.")
    df = pd.DataFrame(rows)

    bits, sizes = [], []
    for p in tqdm(df["path"], desc="hashing"):
        bits.append(dhash(p))
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        sizes.append(f"{im.shape[1]}x{im.shape[0]}")
    df["img_size"] = sizes
    bits = np.array(bits, dtype=np.uint8)
    df.to_csv(C.INDEX_CSV, index=False)

    log = []
    P = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))
    P("=== INDEX ===")
    P(df.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())
    tum = df[df["class"].isin(C.TUMOR_CLASSES)]
    P(f"\nTumor images with a mask: {(tum.mask_path != '').sum()} / {len(tum)}")
    if (tum.mask_path == "").mean() > 0.05:
        P("[WARNING] >5% of tumor images have no mask. Check SEGMENTATION_DIR layout / mask folder names.")

    # ---- duplicates across splits (bit-vector Hamming distance via matrix products)
    P("\n=== CROSS-SPLIT NEAR-DUPLICATES (dHash 16x16, Hamming <= %d / 256) ===" % C.NEAR_DUP_HAMMING)
    rows_dup = []
    splits = df["split"].unique()
    B = bits.astype(np.float32)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            ia, ib = np.where(df.split == a)[0], np.where(df.split == b)[0]
            ham = B[ia].shape[1] - (B[ia] @ B[ib].T + (1 - B[ia]) @ (1 - B[ib]).T)
            hit = np.argwhere(ham <= C.NEAR_DUP_HAMMING)
            P(f"{a} vs {b}: {len(hit)} near-duplicate pairs "
              f"({len(set(hit[:, 1]))} of {len(ib)} {b} images affected)")
            for x, y in hit[:2000]:
                rows_dup.append(dict(split_a=a, split_b=b, path_a=df.path[ia[x]], path_b=df.path[ib[y]],
                                     hamming=int(ham[x, y]), same_class=df["class"][ia[x]] == df["class"][ib[y]]))
    pd.DataFrame(rows_dup).to_csv(C.RESULTS_DIR / "hygiene_near_duplicates.csv", index=False)
    P("If these counts are not ~0, test accuracy is inflated by leakage; remove those pairs from the test set.")

    # ---- acquisition confounds
    P("\n=== CONFOUND CHECKS (tumor classes + no_tumor, all splits) ===")
    for col in ("plane", "img_size"):
        ct = pd.crosstab(df["class"], df[col])
        chi2, p, _, _ = chi2_contingency(ct)
        P(f"\nclass x {col}   chi2={chi2:.1f}  p={p:.2e}\n{ct.to_string()}")
        ct.to_csv(C.RESULTS_DIR / f"hygiene_class_x_{col}.csv")
    P("\nA significant class x plane or class x image-size association means these cues alone can "
      "predict the class -> report them as baselines (see s02).")
    (C.RESULTS_DIR / "hygiene.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
