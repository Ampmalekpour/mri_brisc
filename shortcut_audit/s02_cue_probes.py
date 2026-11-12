"""
STEP 2 - What does each *cue alone* tell us about the class?  (no CNN involved)

Random forests are trained on TUMOR images only (glioma / meningioma / pituitary), fitted on the
TRAIN split and scored on the TEST split, repeated over several seeds. Each probe sees only one cue:

   plane            acquisition plane only (one-hot)
   location         radial_depth, dural_dist_norm, min_edge_dist, dy_norm, dx_abs
   location+plane   the above + plane
   shape            area_ratio, solidity, circularity, eccentricity
   mask_only        32x32 down-sampled binary tumor mask (silhouette: location + size + shape)
   all_hand         location + plane + shape
   image_file_size  original width / height / aspect of the file (curation artefact, not biology)
   intensity_noise  global grey-level statistics and noise level (scanner / protocol cue)

Interpretation: a probe far above chance (33 %) is an information ceiling for that cue. If
'location' alone approaches CNN accuracy, location is a *potential* shortcut; whether the CNNs use
it is tested in s05/s06. The CNN-based probes (tumor crop only, inpainted only) are run through
s03_train.py --variant crop / inpaint and compared in s07.
Output: outputs/results/cue_probes.csv
Run:  python s02_cue_probes.py
"""
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score

import config as C
from common import read_mask, read_rgb

LOC = ["radial_depth", "dural_dist_norm", "min_edge_dist", "dy_norm", "dx_abs"]
SHAPE = ["area_ratio", "solidity", "circularity", "eccentricity"]
INTENSITY = ["int_mean", "int_std", "int_p5", "int_p50", "int_p95", "bg_mean", "noise_lap"]
GEOM = ["img_w", "img_h", "aspect"]   # original file size: pure acquisition/curation cue


def mask_vec(row, size=32):
    img = read_rgb(row["path"])
    m = read_mask(row["mask_path"], img.shape).astype(np.uint8) * 255
    return (cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA) / 255.0).flatten()


def main():
    idx = pd.read_csv(C.INDEX_CSV)
    feats = pd.read_csv(C.FEATURES_CSV)
    df = idx.merge(feats, on="path")
    df["aspect"] = df["img_w"] / df["img_h"]
    df["bg_mean"] = df["bg_mean"].fillna(0.0)
    df = df[df["class"].isin(C.TUMOR_CLASSES)].dropna(subset=LOC + SHAPE + INTENSITY).reset_index(drop=True)
    for p in ("Axial", "Coronal", "Sagittal"):
        df[f"plane_{p}"] = (df["plane"] == p).astype(float)
    PLANE = [c for c in df.columns if c.startswith("plane_")]
    tr, te = df[df.split == C.TRAIN_SPLIT], df[df.split == C.TEST_SPLIT]
    if len(te) == 0:
        raise SystemExit(f"no '{C.TEST_SPLIT}' split found; adjust config.TEST_SPLIT")

    print("building silhouette vectors ...")
    M_tr = np.stack([mask_vec(r) for _, r in tr.iterrows()])
    M_te = np.stack([mask_vec(r) for _, r in te.iterrows()])

    sets = {"plane": PLANE, "location": LOC, "location+plane": LOC + PLANE, "shape": SHAPE,
            "all_hand": LOC + PLANE + SHAPE,
            "image_file_size": GEOM, "intensity_noise": INTENSITY, "acquisition(size+intensity)": GEOM + INTENSITY}
    chance = te["class"].value_counts(normalize=True).max()
    rows = []
    for seed in C.SEEDS:
        for name, cols in sets.items():
            rf = RandomForestClassifier(300, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1)
            rf.fit(tr[cols], tr["class"]); pr = rf.predict(te[cols])
            rows.append(dict(probe=name, seed=seed, acc=accuracy_score(te["class"], pr),
                             bal_acc=balanced_accuracy_score(te["class"], pr)))
        rf = RandomForestClassifier(300, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(M_tr, tr["class"]); pr = rf.predict(M_te)
        rows.append(dict(probe="mask_only", seed=seed, acc=accuracy_score(te["class"], pr),
                         bal_acc=balanced_accuracy_score(te["class"], pr)))
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "cue_probes_raw.csv", index=False)
    summ = res.groupby("probe")[["acc", "bal_acc"]].agg(["mean", "std"]).round(4)
    summ.to_csv(C.RESULTS_DIR / "cue_probes.csv")
    print(f"\nTEST split, tumor classes only (majority-class rate = {chance:.3f}, chance = 0.333)")
    print(summ.to_string())

    rf = RandomForestClassifier(300, max_depth=12, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(tr[LOC + PLANE + SHAPE], tr["class"])
    imp = pd.Series(rf.feature_importances_, LOC + PLANE + SHAPE).sort_values(ascending=False)
    imp.to_csv(C.RESULTS_DIR / "cue_probes_importance.csv", header=["importance"])
    print("\nfeature importance (all_hand):\n", imp.round(3).to_string())


if __name__ == "__main__":
    main()
