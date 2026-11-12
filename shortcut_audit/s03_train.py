"""
STEP 3 - Train classifiers (any architecture, any input variant, several seeds).

Every model is trained on the TRAIN split; the best epoch is chosen on the VAL split (if there is
none, 10 % of train is held out); the TEST split is only touched at the very end.

 --variant   what the network is shown (all built on the fly from image + mask, see common.apply_variant)
     clean       normal images                               -> the standard 'baseline' model
     inpaint     tumor erased + inpainted                     -> ROAR-style: how much class info is left WITHOUT the tumor
     crop        tight crop around the tumor                  -> how much is left WITHOUT the context   (tumor classes only)
     mask        binary silhouette only                       -> location + shape, no tissue            (tumor classes only)
 --p_relocate  probability that a training tumor is moved to a random brain position (the ONE mitigation
               we study: location-randomising augmentation). 0 = off.
 --tumor_only  3-class problem (glioma / meningioma / pituitary). Needed for crop and mask.

Outputs in outputs/runs/<name>/ : model.pt, config.json, history.csv, preds_test.csv (clean test probs)

Examples (PyCharm: Run > Edit configurations > Parameters)
    python s03_train.py --arch efficientnet_b0 --seed 0
    python s03_train.py --arch efficientnet_b0 --variant inpaint --seed 0
    python s03_train.py --arch efficientnet_b0 --p_relocate 0.5 --tag relocate --seed 0
    python s03_train.py --all            # architectures x seeds for the clean baseline
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config as C
from common import BriscDataset, get_device, load_index, make_model, predict, set_seed


def run_name(a):
    parts = [a.arch, a.variant] + ([a.tag] if a.tag else []) + [f"s{a.seed}"]
    return "_".join(parts)


def train_one(a):
    set_seed(a.seed)
    dev = get_device()
    classes = C.TUMOR_CLASSES if (a.tumor_only or a.variant in ("crop", "mask")) else C.CLASS_NAMES
    tumor_only = classes == C.TUMOR_CLASSES
    df_tr = load_index(C.TRAIN_SPLIT, tumor_only)
    df_te = load_index(C.TEST_SPLIT, tumor_only)
    df_va = load_index(C.VAL_SPLIT, tumor_only)
    if len(df_va) == 0:
        df_tr, df_va = train_test_split(df_tr, test_size=0.1, stratify=df_tr["class"], random_state=a.seed)
        print("[info] no val split found -> held out 10 % of train for model selection")
    if len(df_te) == 0:
        raise SystemExit(f"no test split '{C.TEST_SPLIT}' in the index")

    name = run_name(a)
    out = C.RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists() and not a.force:
        print(f"[skip] {name} already trained (use --force to retrain)")
        return out

    mk = lambda df, train: BriscDataset(df, classes, a.variant, 0, C.IMAGE_SIZE, train,
                                        a.p_relocate if train else 0.0, a.seed)
    ds_tr, ds_va, ds_te = mk(df_tr, True), mk(df_va, False), mk(df_te, False)
    loader = DataLoader(ds_tr, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=C.NUM_WORKERS, drop_last=len(ds_tr) > C.BATCH_SIZE)

    model = make_model(a.arch, len(classes), img_size=C.IMAGE_SIZE).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    y_va = np.array([ds_va.cls2idx[c] for c in df_va["class"]])

    best, hist = -1.0, []
    print(f"=== {name} | device={dev} | {len(ds_tr)} train / {len(ds_va)} val / {len(ds_te)} test ===")
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y, _ in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        va_acc = float((predict(model, ds_va, dev).argmax(1) == y_va).mean())
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(ds_tr), val_acc=va_acc, sec=time.time() - t0))
        print(f"  epoch {ep + 1:02d}/{a.epochs} loss {tot / len(ds_tr):.4f} val_acc {va_acc:.4f}")
        if va_acc > best:
            best = va_acc; torch.save(model.state_dict(), out / "model.pt")
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    probs = predict(model, ds_te, dev)
    pred = df_te[["path", "class", "split"]].copy()
    for i, c in enumerate(classes):
        pred[f"p_{c}"] = probs[:, i]
    pred["pred"] = [classes[i] for i in probs.argmax(1)]
    pred.to_csv(out / "preds_test.csv", index=False)
    json.dump(dict(arch=a.arch, variant=a.variant, p_relocate=a.p_relocate, tag=a.tag, seed=a.seed,
                   classes=classes, size=C.IMAGE_SIZE, best_val_acc=best, epochs=a.epochs,
                   test_acc=float((pred["pred"] == pred["class"]).mean())), open(out / "config.json", "w"), indent=2)
    print(f"  -> test accuracy {(pred['pred'] == pred['class']).mean():.4f}  saved to {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=C.ARCHS[0])
    ap.add_argument("--variant", default="clean", choices=["clean", "inpaint", "crop", "mask"])
    ap.add_argument("--p_relocate", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tumor_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all", action="store_true", help="clean baseline for every arch in config.ARCHS x config.SEEDS")
    a = ap.parse_args()
    if a.all:
        for arch in C.ARCHS:
            for s in C.SEEDS:
                a.arch, a.seed = arch, s
                train_one(a)
    else:
        train_one(a)


if __name__ == "__main__":
    main()
