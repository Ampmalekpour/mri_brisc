"""
STEP 5 - Controlled interventions on the TEST tumors (what happens to a trained model when we edit the image?)

Every number is accuracy on test tumor images with a bootstrap 95 % CI, per class, plus the share of
predictions that flip to 'no_tumor'.  Variants (see common.apply_variant):

 A. ERASURE SWEEP     zero | gaussian | inpaint, mask dilated by k = config.DILATION_SWEEP pixels.
                      Dilating checks for residual pathology (edema / mass effect / mask edge): if accuracy
                      keeps falling as k grows, the 'tumor-free' image was NOT tumor-free at k=0.
 B. PLACEBO           the same black silhouette moved to a random brain location, tumor left in place.
                      zero(k=0) vs placebo separates 'model needs tumor tissue' from 'model is just
                      disturbed by a black blob'.
 C. TRANSPLANT        tumor pixels pasted onto a no_tumor scan of the same plane.
      same-location   at the donor's own position  -> does the model call it the donor class?
      cue-conflict    moved to where a DIFFERENT class's tumors usually are
                      -> reports share predicting the appearance class A, the location class B, or no_tumor.
 D. FOREGROUND        tumor only on black background.

Interpretation guide: a model that diagnoses from tissue keeps predicting A in C(cue-conflict) and falls
to 'no_tumor' in A(inpaint, large k). A location-shortcut model follows B in C and stays high on inpaint.

Outputs: runs/<run>/interventions_long.csv, results/interventions_summary.csv, results/transplant_summary.csv
Run: python s05_interventions.py --run efficientnet_b0_clean_s0     (or no args: every clean run)
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C
from common import (BriscDataset, apply_variant, bootstrap_ci, brain_mask, centroid, get_device,
                    load_run, paste_tumor, predict, read_mask, read_rgb, to_tensor)


# ------------------------------------------------------------------ transplant dataset
class TransplantDataset(Dataset):
    """Items are (recipient no_tumor image, donor tumor image + mask, shift). Label = donor class."""

    def __init__(self, items, classes, size):
        self.items, self.size = items, size
        self.cls2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        rec = read_rgb(it["recipient"]); don = read_rgb(it["donor"])
        dm = read_mask(it["donor_mask"], don.shape)
        out = paste_tumor(rec, don, dm, it["dx"], it["dy"])
        return to_tensor(out, self.size), self.cls2idx[it["donor_class"]], i


def build_transplants(test_df, train_feats, rng, n_per_class=60, mode="same"):
    """mode 'same': keep donor location.  mode 'swap': move to the centroid of a tumor of another class."""
    nt = test_df[test_df["class"] == "no_tumor"]
    items = []
    for cls in C.TUMOR_CLASSES:
        donors = test_df[test_df["class"] == cls]
        donors = donors.sample(min(n_per_class, len(donors)), random_state=int(rng.integers(1e6)))
        for _, d in donors.iterrows():
            rec_pool = nt[nt["plane"] == d["plane"]]
            if rec_pool.empty or not d["mask_path"]:
                continue
            don_img = read_rgb(d["path"]); dm = read_mask(d["mask_path"], don_img.shape)
            cx, cy = centroid(dm)
            for _try in range(12):
                rec = rec_pool.iloc[rng.integers(len(rec_pool))]
                rec_img = read_rgb(rec["path"]); br = brain_mask(rec_img)
                if mode == "same":
                    dx = dy = 0.0; loc_cls = cls
                else:
                    others = train_feats[(train_feats["plane"] == d["plane"]) & (train_feats["class"] != cls)
                                         & train_feats["class"].isin(C.TUMOR_CLASSES)].dropna(subset=["t_cx"])
                    if others.empty:
                        break
                    o = others.iloc[rng.integers(len(others))]
                    dx, dy, loc_cls = o["t_cx"] - cx, o["t_cy"] - cy, o["class"]
                tx, ty = int(round(cx + dx)), int(round(cy + dy))
                if 0 <= ty < br.shape[0] and 0 <= tx < br.shape[1] and br[ty, tx]:
                    items.append(dict(recipient=rec["path"], donor=d["path"], donor_mask=d["mask_path"],
                                      dx=dx, dy=dy, donor_class=cls, location_class=loc_cls))
                    break
    return items


# ------------------------------------------------------------------ main logic
def evaluate_run(run, max_per_class, rng):
    dev = get_device()
    model, meta = load_run(C.RUNS_DIR / run, dev)
    classes, size = meta["classes"], meta["size"]
    idx = pd.read_csv(C.INDEX_CSV)
    te = idx[idx.split == C.TEST_SPLIT]
    tum = te[te["class"].isin(C.TUMOR_CLASSES) & (te.mask_path.fillna("") != "")]
    tum = pd.concat([tum[tum["class"] == c].head(max_per_class) for c in C.TUMOR_CLASSES]).reset_index(drop=True)

    variants = [("clean", 0)]
    variants += [(v, k) for v in ("zero", "gaussian", "inpaint") for k in C.DILATION_SWEEP]
    variants += [("placebo", 0), ("placebo", 10), ("foreground", 0)]
    long_rows = []
    for v, k in variants:
        ds = BriscDataset(tum, classes, v, k, size, False, 0.0, seed=0)
        P = predict(model, ds, dev)
        for i, r in tum.iterrows():
            row = dict(run=run, variant=v, k=k, path=r["path"], cls=r["class"], pred=classes[P[i].argmax()],
                       p_true=P[i, classes.index(r["class"])])
            row["p_no_tumor"] = P[i, classes.index("no_tumor")] if "no_tumor" in classes else np.nan
            long_rows.append(row)
        print(f"  {run}: {v:10s} k={k:<3d} acc={np.mean([x['pred']==x['cls'] for x in long_rows[-len(tum):]]):.3f}")
    long = pd.DataFrame(long_rows)
    long.to_csv(C.RUNS_DIR / run / "interventions_long.csv", index=False)

    summ = []
    for (v, k), g in long.groupby(["variant", "k"]):
        for cls in ["ALL"] + C.TUMOR_CLASSES:
            s = g if cls == "ALL" else g[g.cls == cls]
            mean, lo, hi = bootstrap_ci((s.pred == s.cls).astype(float))
            summ.append(dict(run=run, variant=v, k=k, cls=cls, n=len(s), acc=mean, ci_lo=lo, ci_hi=hi,
                             frac_pred_no_tumor=float((s.pred == "no_tumor").mean()), mean_p_true=s.p_true.mean()))

    # ---- transplants (needs a 4-class model)
    trans = []
    if "no_tumor" in classes:
        feats = idx[idx.split == C.TRAIN_SPLIT].merge(pd.read_csv(C.FEATURES_CSV), on="path")
        n_pc = max(10, max_per_class)
        for mode in ("same", "swap"):
            items = build_transplants(te, feats, rng, n_per_class=n_pc, mode=mode)
            if not items:
                continue
            ds = TransplantDataset(items, classes, size)
            P = predict(model, ds, dev); pred = np.array([classes[j] for j in P.argmax(1)])
            for A in C.TUMOR_CLASSES:
                sel = np.array([it["donor_class"] == A for it in items])
                if not sel.any():
                    continue
                row = dict(run=run, mode=mode, donor_class=A, n=int(sel.sum()),
                           pred_donor=float((pred[sel] == A).mean()),
                           pred_no_tumor=float((pred[sel] == "no_tumor").mean()))
                if mode == "swap":
                    locs = np.array([it["location_class"] for it in items])[sel]
                    row["pred_location_class"] = float((pred[sel] == locs).mean())
                trans.append(row)
        # control: untouched no_tumor recipients
        nt = te[te["class"] == "no_tumor"].head(max_per_class * 3).reset_index(drop=True)
        P = predict(model, BriscDataset(nt, classes, "clean", 0, size), dev)
        trans.append(dict(run=run, mode="control_no_tumor_clean", donor_class="-", n=len(nt),
                          pred_no_tumor=float((P.argmax(1) == classes.index("no_tumor")).mean())))
    return summ, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*")
    ap.add_argument("--max_per_class", type=int, default=200)
    a = ap.parse_args()
    runs = a.run or sorted(p.name for p in C.RUNS_DIR.iterdir() if "_clean_" in p.name)
    runs = [r for r in runs if (C.RUNS_DIR / r / "model.pt").exists()]
    rng = np.random.default_rng(0)
    S, T = [], []
    for r in runs:
        s, t = evaluate_run(r, a.max_per_class, rng); S += s; T += t
    pd.DataFrame(S).to_csv(C.RESULTS_DIR / "interventions_summary.csv", index=False)
    pd.DataFrame(T).to_csv(C.RESULTS_DIR / "transplant_summary.csv", index=False)
    print("\nsaved interventions_summary.csv / transplant_summary.csv in", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
