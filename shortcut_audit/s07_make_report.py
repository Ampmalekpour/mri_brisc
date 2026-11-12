"""
STEP 7 - Collect every result table into one readable report  (outputs/results/REPORT.md + figures).

Aggregates over seeds (mean +- std across runs of the same arch/variant/tag). Nothing is recomputed here;
re-run it any time after s02-s06 to refresh the report.
Run: python s07_make_report.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

R = C.RESULTS_DIR


def md(df, floatfmt=3):
    if df is None or len(df) == 0:
        return "_not available (step not run)_\n"
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def family(run):
    """efficientnet_b0_clean_relocate_s1 -> (efficientnet_b0, clean, relocate)"""
    m = json.loads((C.RUNS_DIR / run / "config.json").read_text())
    return m["arch"], m["variant"], m["tag"] or "-"


def main():
    out = ["# Shortcut-learning audit - results\n"]
    # 1. models
    rows = []
    for p in sorted(C.RUNS_DIR.glob("*/config.json")):
        m = json.loads(p.read_text())
        rows.append(dict(arch=m["arch"], variant=m["variant"], tag=m["tag"] or "-", seed=m["seed"],
                         classes=len(m["classes"]), test_acc=m["test_acc"]))
    runs = pd.DataFrame(rows)
    if len(runs):
        g = runs.groupby(["arch", "variant", "tag", "classes"])["test_acc"].agg(["mean", "std", "count"]).reset_index()
        out += ["## 1. Trained models (TEST accuracy, mean/std over seeds)\n", md(g),
                "`variant` = what the network saw: clean image, inpainted (tumor removed), tumor crop, or silhouette mask. "
                "`tag=relocate` = trained with location-randomising augmentation (the mitigation).\n"]
    # 2. cue probes
    cp = R / "cue_probes.csv"
    if cp.exists():
        raw = pd.read_csv(R / "cue_probes_raw.csv").groupby("probe")[["acc", "bal_acc"]].agg(["mean", "std"])
        raw.columns = ["_".join(c) for c in raw.columns]
        out += ["## 2. Non-CNN cue probes (tumor classes, chance = 0.33)\n", md(raw.reset_index()),
                "Reading: how much each cue alone tells about the class. Compare with the CNN rows above.\n"]
    # 3. typicality
    ta = R / "typicality_accuracy.csv"
    if ta.exists():
        t = pd.read_csv(ta)
        t = t[t.tercile.isin(["low", "mid", "high", "ALL"])]
        t["family"] = t["run"].map(lambda r: "/".join(family(r)))
        piv = t.groupby(["family", "tercile"])["acc"].mean().unstack()[["low", "mid", "high", "ALL"]].reset_index()
        piv["gap_high_minus_low"] = piv["high"] - piv["low"]
        rho = t.groupby("family")["spearman_rho"].mean().reset_index()
        out += ["## 3. Natural experiment - accuracy vs location typicality (low/mid/high terciles within class)\n",
                md(piv.merge(rho, on="family")),
                "A large `gap_high_minus_low` and positive rho = performance depends on the tumor sitting where its class usually sits.\n"]
    # 4. interventions
    ia = R / "interventions_summary.csv"
    if ia.exists():
        s = pd.read_csv(ia)
        s = s[s.cls == "ALL"].copy()
        s["family"] = s["run"].map(lambda r: "/".join(family(r)))
        s["intervention"] = s["variant"] + "(k=" + s["k"].astype(str) + ")"
        piv = s.groupby(["family", "intervention"])["acc"].mean().unstack()
        order = [c for c in ["clean(k=0)", "zero(k=0)", "zero(k=20)", "gaussian(k=0)", "gaussian(k=20)", "inpaint(k=0)",
                             "inpaint(k=20)", "placebo(k=0)", "placebo(k=10)", "foreground(k=0)"] if c in piv.columns]
        out += ["## 4. Erasure / placebo / foreground interventions (accuracy on test tumors, mean over seeds)\n",
                md(piv[order].reset_index()),
                "`placebo` keeps the tumor but adds the black silhouette elsewhere: if zero(k=0) falls but placebo does not, the model needs tumor tissue; "
                "if inpaint stays high the class is recoverable WITHOUT the tumor (context / residual signal). Compare k=0 vs k=20 for residual pathology.\n"]
        nt = s.groupby(["family", "intervention"])["frac_pred_no_tumor"].mean().unstack()
        keep = [c for c in ["zero(k=0)", "inpaint(k=0)", "inpaint(k=20)"] if c in nt.columns]
        out += ["Share of predictions that become `no_tumor`:\n", md(nt[keep].reset_index())]
        fig, ax = plt.subplots(figsize=(9, 4))
        piv[order].T.plot(kind="bar", ax=ax); ax.set_ylabel("accuracy on test tumors"); ax.set_ylim(0, 1.02)
        plt.xticks(rotation=40, ha="right"); plt.tight_layout(); plt.savefig(R / "fig_interventions.png", dpi=140); plt.close()
        out.append("![interventions](fig_interventions.png)\n")
    # 5. transplant
    tp = R / "transplant_summary.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        t["family"] = t["run"].map(lambda r: "/".join(family(r)))
        g = t.groupby(["family", "mode", "donor_class"])[["pred_donor", "pred_location_class", "pred_no_tumor"]].mean().reset_index()
        out += ["## 5. Tumor transplant onto no_tumor scans\n", md(g),
                "`same`: tumor pasted at its own location. `swap` (cue conflict): tumor of class A pasted where class B usually sits; "
                "`pred_donor` = follows tissue, `pred_location_class` = follows location. `control` row: untouched no_tumor accuracy.\n"]
    # 6. XAI
    xs = R / "xai_summary.csv"
    if xs.exists():
        x = pd.read_csv(xs)
        out += ["## 6. Explainability metrics (mean over test tumors)\n", md(x, 3),
                "`enrichment` > 1 = attribution concentrates on the tumor; `deletion_auc_attr` should be below `_random` and above `_oracle` "
                "if the map is faithful; `sanity_rho_*` near 0 = map changes when the model is destroyed (passes the sanity check).\n"]
    # 7. hygiene
    h = R / "hygiene.txt"
    if h.exists():
        out += ["## 0. Data hygiene (from s00)\n", "```\n" + h.read_text(encoding="utf-8") + "\n```\n"]
    (R / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote", R / "REPORT.md")


if __name__ == "__main__":
    main()
