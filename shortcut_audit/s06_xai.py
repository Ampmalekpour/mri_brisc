"""
STEP 6 - Explainability as QUANTIFIED evidence (not a picture gallery).

Two attribution methods per model: Grad-CAM (final feature map) and SmoothGrad (input gradient).
For N test tumors per class (default 50) the following are computed against the tumor mask:

  mass_in_mask      share of attribution inside the tumor
  enrichment        mass_in_mask / mask area share  (1 = no preference, >1 = focuses on tumor)
  mass_ring         share in a context ring around the tumor (dilated 20px, minus tumor)
  mass_background   share outside the brain (pure shortcut/artefact attention)
  pointing_game     1 if the attribution maximum falls inside the (5px-dilated) tumor
  deletion_auc      P(predicted class) while removing the most-attributed 14x14 cells  (lower = more faithful)
  insertion_auc     P(predicted class) while adding them to a blank image            (higher = more faithful)
                    each computed for the method's order, a RANDOM order and an ORACLE (tumor-first) order
  sanity            Spearman correlation between the map of the trained model and the map of
                    (a) the same model with a re-initialised classifier head, (b) a fully random model.
                    A trustworthy method must change (correlation near 0) when the model is destroyed.

READ THIS AS: attention on the tumor says the model *looks* there - it does not say that the model uses
tumor *appearance* rather than tumor *location*. That distinction comes from s04/s05. XAI is only
supporting evidence; deletion/insertion tell you whether the map is faithful at all.

Outputs: results/xai_per_image.csv, results/xai_summary.csv, results/xai_examples_<run>.png
Run: python s06_xai.py --run efficientnet_b0_clean_s0 --n_per_class 50
"""
import argparse
import copy

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import config as C
from common import (bootstrap_ci, brain_mask, dilate, get_device, load_run, make_model, read_mask,
                    read_rgb, to_tensor, IMNET_MEAN, IMNET_STD)

GRID = 14
_trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy 2 renamed trapz


# ------------------------------------------------------------------ attribution methods
def pick_layer(model, arch):
    if arch.startswith("efficientnet"):
        return model.bn2
    if arch.startswith("resnet"):
        return model.layer4
    if arch.startswith(("vit", "deit")):
        return model.blocks[-1].norm1
    raise ValueError(f"no Grad-CAM layer registered for {arch} - add one in pick_layer()")


def gradcam(model, layer, x, target):
    store = {}
    def fwd(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))
    h = layer.register_forward_hook(fwd)
    model.zero_grad()
    logits = model(x)
    logits[0, target].backward()
    h.remove()
    A, G = store["a"].detach(), store["g"].detach()
    if A.ndim == 3:                                   # ViT tokens [1, 1+N, C] -> [1, C, g, g]
        A, G = A[:, 1:], G[:, 1:]
        g = int(round(A.shape[1] ** 0.5))
        A, G = (t.transpose(1, 2).reshape(1, -1, g, g) for t in (A, G))
    w = G.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return cam.cpu().numpy()


def smoothgrad(model, x, target, n=8, sigma=0.15):
    acc = torch.zeros_like(x[0, 0])
    rng = (x.max() - x.min()).item()
    for _ in range(n):
        xn = (x + torch.randn_like(x) * sigma * rng).requires_grad_(True)
        model.zero_grad()
        model(xn)[0, target].backward()
        acc += xn.grad.abs().mean(1)[0]
    m = acc.cpu().numpy() / n
    return cv2.GaussianBlur(m, (0, 0), 3)


# ------------------------------------------------------------------ metrics
def norm_map(m):
    m = np.maximum(m, 0).astype(np.float64)
    return m / (m.sum() + 1e-12)


@torch.no_grad()
def curve_auc(model, x, target, order_cells, mode, steps=15):
    """Deletion (replace by dataset mean) / insertion (start blank) curve over 14x14 cells."""
    H = x.shape[-1]; cs = H // GRID
    ks = np.unique(np.linspace(0, GRID * GRID, steps).astype(int))
    batch = []
    for k in ks:
        keep = torch.zeros(GRID * GRID); keep[order_cells[:k]] = 1
        m = F.interpolate(keep.view(1, 1, GRID, GRID), size=(GRID * cs, GRID * cs), mode="nearest")
        m = F.pad(m, (0, H - GRID * cs, 0, H - GRID * cs)).to(x.device)
        batch.append(x * (1 - m) if mode == "del" else x * m)
    p = torch.softmax(model(torch.cat(batch)), 1)[:, target].cpu().numpy()
    return float(_trapz(p, ks / (GRID * GRID)))


def cell_scores(m):
    t = torch.from_numpy(m).float()[None, None]
    return F.adaptive_avg_pool2d(t, GRID).flatten().numpy()


def image_metrics(model, x, amap, mask, brain, target, rng):
    a = norm_map(amap)
    area = mask.mean()
    ring = dilate(mask, 20) & ~mask
    out = dict(mass_in_mask=float((a * mask).sum()), mass_ring=float((a * ring).sum()),
               mass_background=float((a * ~brain).sum()))
    out["enrichment"] = out["mass_in_mask"] / (area + 1e-12)
    yy, xx = np.unravel_index(np.argmax(amap), amap.shape)
    out["pointing_game"] = float(dilate(mask, 5)[yy, xx])
    orders = {"attr": np.argsort(-cell_scores(a)), "random": rng.permutation(GRID * GRID),
              "oracle": np.argsort(-cell_scores(mask.astype(np.float64)))}
    for name, o in orders.items():
        out[f"deletion_auc_{name}"] = curve_auc(model, x, target, o, "del")
        out[f"insertion_auc_{name}"] = curve_auc(model, x, target, o, "ins")
    return out


# ------------------------------------------------------------------ driver
def run_one(run, n_per_class, rng):
    dev = get_device()
    model, meta = load_run(C.RUNS_DIR / run, dev)
    classes, size, arch = meta["classes"], meta["size"], meta["arch"]
    layer = pick_layer(model, arch)
    head_rand = copy.deepcopy(model); head_rand.reset_classifier(len(classes))
    full_rand = make_model(arch, len(classes), pretrained=False, img_size=size).to(dev).eval()
    layer_h, layer_f = pick_layer(head_rand, arch), pick_layer(full_rand, arch)

    idx = pd.read_csv(C.INDEX_CSV)
    te = idx[(idx.split == C.TEST_SPLIT) & idx["class"].isin(C.TUMOR_CLASSES) & (idx.mask_path.fillna("") != "")]
    te = pd.concat([te[te["class"] == c].sample(min(n_per_class, (te["class"] == c).sum()), random_state=0)
                    for c in C.TUMOR_CLASSES])
    rows, examples = [], []
    for _, r in te.iterrows():
        img = read_rgb(r["path"]); mask0 = read_mask(r["mask_path"], img.shape)
        mask = cv2.resize(mask0.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
        small = cv2.resize(img, (size, size)); brain = brain_mask(small)
        x = to_tensor(img, size)[None].to(dev)
        with torch.no_grad():
            target = int(model(x).argmax(1))
        maps = {"gradcam": gradcam(model, layer, x, target), "smoothgrad": smoothgrad(model, x, target)}
        sanity = {"gradcam_headrand": gradcam(head_rand, layer_h, x, target),
                  "gradcam_fullrand": gradcam(full_rand, layer_f, x, target)}
        for meth, m in maps.items():
            row = dict(run=run, method=meth, path=r["path"], cls=r["class"], pred=classes[target],
                       correct=classes[target] == r["class"], **image_metrics(model, x, m, mask, brain, target, rng))
            if meth == "gradcam":
                row["sanity_rho_headrand"] = spearmanr(m.ravel(), sanity["gradcam_headrand"].ravel())[0]
                row["sanity_rho_fullrand"] = spearmanr(m.ravel(), sanity["gradcam_fullrand"].ravel())[0]
            rows.append(row)
        if len(examples) < 6:
            examples.append((small, mask, maps))
    # figure
    fig, ax = plt.subplots(len(examples), 3, figsize=(7.5, 2.5 * len(examples)))
    ax = np.atleast_2d(ax)
    for i, (im, mk, mp) in enumerate(examples):
        ax[i, 0].imshow(im); ax[i, 0].contour(mk, colors="lime", linewidths=1)
        for j, (k, m) in enumerate(mp.items()):
            ax[i, j + 1].imshow(im); ax[i, j + 1].imshow(m / (m.max() + 1e-9), cmap="jet", alpha=0.45)
            ax[i, j + 1].contour(mk, colors="white", linewidths=0.8)
            if i == 0: ax[i, j + 1].set_title(k)
        ax[i, 0].set_title("input + tumor mask" if i == 0 else "")
    for a_ in ax.ravel(): a_.axis("off")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / f"xai_examples_{run}.png", dpi=120); plt.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*"); ap.add_argument("--n_per_class", type=int, default=50)
    a = ap.parse_args()
    runs = a.run or sorted(p.name for p in C.RUNS_DIR.iterdir() if (p / "model.pt").exists() and "_clean_" in p.name)
    rng = np.random.default_rng(0)
    df = pd.concat([run_one(r, a.n_per_class, rng) for r in runs], ignore_index=True)
    df.to_csv(C.RESULTS_DIR / "xai_per_image.csv", index=False)
    metrics = [c for c in df.columns if c.startswith(("mass", "enrich", "pointing", "deletion", "insertion", "sanity"))]
    summ = df.groupby(["run", "method"])[metrics].mean().round(4)
    summ.to_csv(C.RESULTS_DIR / "xai_summary.csv")
    print(summ.T.to_string())


if __name__ == "__main__":
    main()
