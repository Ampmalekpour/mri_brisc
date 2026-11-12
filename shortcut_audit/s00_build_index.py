"""
STEP 0 - Build the master index and run data-hygiene checks.

Creates   outputs/index.csv  (one row per classification image: path, class, split, plane, mask_path,
                              image size, dHash)
Reports   outputs/results/hygiene_*.csv / hygiene.txt
   * relabels leaky images: test images with a near-duplicate in train -> split "test_leaky"; a val folder that is a
     copy of test -> "val_copy_of_test" (config.REMOVE_LEAKY); train gets a near-duplicate "group" id
   * how many tumor images have a matching segmentation mask
   * near-duplicate images within and ACROSS splits (BRISC has no patient id)
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

    bits, sizes, ws, hs = [], [], [], []
    for p in tqdm(df["path"], desc="hashing"):
        bits.append(dhash(p))
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        sizes.append(f"{im.shape[1]}x{im.shape[0]}"); ws.append(im.shape[1]); hs.append(im.shape[0])
    df["img_size"] = sizes; df["img_w"] = ws; df["img_h"] = hs
    bits = np.array(bits, dtype=np.uint8)
    log = []
    P = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))
    P("=== INDEX ===")
    P(df.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())
    tum = df[df["class"].isin(C.TUMOR_CLASSES)]
    P(f"\nTumor images with a mask: {(tum.mask_path != '').sum()} / {len(tum)}")
    if (tum.mask_path == "").mean() > 0.05:
        P("[WARNING] >5% of tumor images have no mask. Check SEGMENTATION_DIR layout / mask folder names.")

    # ---- duplicates (bit-vector Hamming distance via matrix products)
    P("\n=== NEAR-DUPLICATES (dHash 16x16, Hamming <= %d / 256) ===" % C.NEAR_DUP_HAMMING)
    B = bits.astype(np.float32)
    ham = lambda ia, ib: B.shape[1] - (B[ia] @ B[ib].T + (1 - B[ia]) @ (1 - B[ib]).T)
    sp = df["split"].values
    where = {s_: np.where(sp == s_)[0] for s_ in np.unique(sp)}
    rows_dup, train_dup = [], np.zeros(len(df), bool)
    names = list(where)
    for i, a_ in enumerate(names):
        for b_ in names[i + 1:]:
            H = ham(where[a_], where[b_])
            hit = np.argwhere(H <= C.NEAR_DUP_HAMMING)
            P(f"{a_} vs {b_}: {len(hit)} near-duplicate pairs ({len(set(hit[:, 1]))} of {len(where[b_])} {b_} "
              f"images / {len(set(hit[:, 0]))} of {len(where[a_])} {a_} images)")
            for x, y in hit[:3000]:
                rows_dup.append(dict(split_a=a_, split_b=b_, path_a=df.path[where[a_][x]], path_b=df.path[where[b_][y]],
                                     hamming=int(H[x, y]),
                                     same_class=df["class"][where[a_][x]] == df["class"][where[b_][y]]))
            if C.TRAIN_SPLIT in (a_, b_):             # which non-train images have a twin in train
                other, mine = (b_, hit[:, 1]) if a_ == C.TRAIN_SPLIT else (a_, hit[:, 0])
                train_dup[where[other][np.unique(mine)]] = True
    pd.DataFrame(rows_dup).to_csv(C.RESULTS_DIR / "hygiene_near_duplicates.csv", index=False)

    # train-internal near-duplicate groups (same patient / neighbouring slices) -> used for grouped hold-out
    group = -np.ones(len(df), int)
    ti = where.get(C.TRAIN_SPLIT, np.array([], int))
    if len(ti):
        H = ham(ti, ti); np.fill_diagonal(H, 999)
        parent = list(range(len(ti)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for x, y in np.argwhere(H <= C.NEAR_DUP_HAMMING):
            parent[find(x)] = find(y)
        group[ti] = [find(i) for i in range(len(ti))]
        P(f"train: {len(set(group[ti]))} near-duplicate groups for {len(ti)} images")
    df["group"] = group
    df["train_dup"] = train_dup

    # ---- act on leakage (config.REMOVE_LEAKY)
    if C.REMOVE_LEAKY:
        va, te = where.get(C.VAL_SPLIT, []), where.get(C.TEST_SPLIT, [])
        if len(va) and len(te):
            Hvt = ham(va, te)
            frac = float((Hvt.min(1) <= C.NEAR_DUP_HAMMING).mean())
            if frac > 0.9:
                P(f"\n[ACTION] {frac:.0%} of '{C.VAL_SPLIT}' images are copies of '{C.TEST_SPLIT}' images -> '{C.VAL_SPLIT}' is NOT an independent "
                  f"validation set. It is relabelled 'val_copy_of_test' and a grouped 10 % hold-out of train is used for model selection instead.")
                df.loc[df.split == C.VAL_SPLIT, "split"] = "val_copy_of_test"
        n_leaky = int(((df.split == C.TEST_SPLIT) & df.train_dup).sum())
        P(f"[ACTION] {n_leaky} '{C.TEST_SPLIT}' images have a near-duplicate in train -> relabelled 'test_leaky' "
          f"(excluded from every evaluation; {int((df.split == C.TEST_SPLIT).sum())} clean test images remain).")
        df.loc[(df.split == C.TEST_SPLIT) & df.train_dup, "split"] = "test_leaky"
    df.to_csv(C.INDEX_CSV, index=False)
    P("\nFinal split sizes:\n" + df.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())

    # ---- acquisition confounds
    P("\n=== CONFOUND CHECKS (all images) ===")
    for col in ("plane", "img_size"):
        ct = pd.crosstab(df["class"], df[col])
        chi2, p, _, _ = chi2_contingency(ct)
        cramers = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))) if min(ct.shape) > 1 else float('nan')
        ct.to_csv(C.RESULTS_DIR / f"hygiene_class_x_{col}.csv")
        P(f"\nclass x {col}: chi2={chi2:.0f}  p={p:.1e}  Cramer's V={cramers:.3f}  ({ct.shape[1]} distinct values)")
        if col == "plane":
            P(ct.to_string())
    top = df.groupby("class")["img_size"].agg(lambda s: f"{s.value_counts().index[0]} ({s.value_counts().iloc[0] / len(s):.0%} of class)")
    P("\nmost common original image size per class:\n" + top.to_string())
    P("\nA high Cramer's V for image size means the FILE SIZE ALONE reveals the class (curation artefact) - "
      "s02 quantifies this as a probe, and CNNs may exploit resolution / interpolation traces after resizing.")
    (C.RESULTS_DIR / "hygiene.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
