"""
STEP 4 - NATURAL EXPERIMENT: do CNNs fail when the tumor sits in an atypical place?   (no image editing)

1. A location+plane random forest is fitted on TRAIN tumors. For each TEST tumor it gives
   typicality = P(true class | location, plane): high = 'sits where its class usually sits'.
2. Test tumors are split into typicality terciles *within each class*  (low / mid / high).
3. For each trained model the accuracy in each tercile is reported with a bootstrap CI.
   Shortcut reliance predicts: accuracy(low) << accuracy(high), while a model that reads tissue
   shows no gradient. Also reports the Spearman correlation between typicality and correctness.

Pass --runs <names> (folders in outputs/runs); default = every clean model.
Outputs: results/typicality_accuracy.csv, results/typicality_probe.csv
Run: python s04_location_typicality.py
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

import config as C
from common import bootstrap_ci
from s02_cue_probes import LOC


def typicality_table():
    df = pd.read_csv(C.INDEX_CSV).merge(pd.read_csv(C.FEATURES_CSV), on="path")
    df = df[df["class"].isin(C.TUMOR_CLASSES)].dropna(subset=LOC).reset_index(drop=True)
    for p in ("Axial", "Coronal", "Sagittal"):
        df[f"plane_{p}"] = (df["plane"] == p).astype(float)
    cols = LOC + [c for c in df.columns if c.startswith("plane_")]
    tr, te = df[df.split == C.TRAIN_SPLIT], df[df.split == C.TEST_SPLIT].copy()
    rf = RandomForestClassifier(500, max_depth=10, min_samples_leaf=3, class_weight="balanced",
                                random_state=0, n_jobs=-1).fit(tr[cols], tr["class"])
    P = rf.predict_proba(te[cols])
    te["typicality"] = [P[i, list(rf.classes_).index(c)] for i, c in enumerate(te["class"])]
    # tercile within class (rank first so ties don't break qcut)
    te["tercile"] = te.groupby("class")["typicality"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
    return te[["path", "class", "typicality", "tercile"]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    typ = typicality_table()
    typ.to_csv(C.RESULTS_DIR / "typicality_probe.csv", index=False)
    runs = a.runs or sorted(p.name for p in C.RUNS_DIR.iterdir() if (p / "preds_test.csv").exists())
    rows = []
    for r in runs:
        pr = pd.read_csv(C.RUNS_DIR / r / "preds_test.csv")
        m = pr.merge(typ, on=["path", "class"])
        if m.empty:
            continue
        m["correct"] = (m["pred"] == m["class"]).astype(float)
        rho, p = spearmanr(m["typicality"], m["correct"])
        for terc in ["low", "mid", "high", "ALL"]:
            sub = m if terc == "ALL" else m[m.tercile == terc]
            mean, lo, hi = bootstrap_ci(sub["correct"])
            rows.append(dict(run=r, tercile=terc, n=len(sub), acc=mean, ci_lo=lo, ci_hi=hi,
                             spearman_rho=rho, spearman_p=p))
        for cls in C.TUMOR_CLASSES:
            for terc in ["low", "high"]:
                sub = m[(m["class"] == cls) & (m.tercile == terc)]
                mean, lo, hi = bootstrap_ci(sub["correct"])
                rows.append(dict(run=r, tercile=f"{cls}:{terc}", n=len(sub), acc=mean, ci_lo=lo, ci_hi=hi,
                                 spearman_rho=rho, spearman_p=p))
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "typicality_accuracy.csv", index=False)
    print(res[res.tercile.isin(["low", "mid", "high", "ALL"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
