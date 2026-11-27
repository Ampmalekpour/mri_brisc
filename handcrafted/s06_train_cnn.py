"""Step 6. Train matched full-image and mask-guided CNNs on the same cohorts."""

import random

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

try:
    import torch
except ImportError as error:
    raise SystemExit(
        "Install requirements-cnn.txt in your PyCharm interpreter before running step 6."
    ) from error

from torch import nn
from torch.utils.data import DataLoader

import config as C
from cnn_model import MRIClassifier, MRIViews
from data import load_manifest, manifest_digest, output_path, save_csv, save_json
from evaluation import cohort_rows, encode, evaluate, grouped_folds, prediction_frame


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(rows, targets, guided, shuffle, seed, device):
    dataset = MRIViews(rows, targets, C.CNN_SIZE, guided)
    return DataLoader(
        dataset,
        batch_size=C.CNN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )


def train_epoch(model, batches, optimizer, loss, device):
    model.train()
    accumulated, examples = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        value = loss(model(x), y)
        value.backward()
        optimizer.step()
        accumulated += value.item() * len(y)
        examples += len(y)
    return accumulated / examples


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    return np.vstack([model(x.to(device)).softmax(dim=1).cpu().numpy() for x, _ in batches])


def fresh_model(classes, device, seed):
    set_seed(seed)
    model = MRIClassifier(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    return model, optimizer, nn.CrossEntropyLoss()


def train_variant(train, test, y_train, y_test, classes, fit_idx, val_idx, experiment, guided, seed, device):
    stem = f"{experiment}_seed{seed}"
    print(
        f"\n{stem} | {len(fit_idx):,} training and {len(val_idx):,} validation images for epoch selection",
        flush=True,
    )
    fit_loader = loader(train.iloc[fit_idx], y_train[fit_idx], guided, True, seed, device)
    val_loader = loader(train.iloc[val_idx], y_train[val_idx], guided, False, seed, device)
    model, optimizer, loss = fresh_model(classes, device, seed)
    history = []
    best_epoch, best_score = 1, -1.0
    for epoch in range(1, C.CNN_EPOCHS + 1):
        training_loss = train_epoch(model, fit_loader, optimizer, loss, device)
        validation = predict(model, val_loader, device).argmax(axis=1)
        score = balanced_accuracy_score(y_train[val_idx], validation)
        history.append(
            dict(phase="epoch_selection", epoch=epoch, train_loss=training_loss, val_balanced_accuracy=score)
        )
        print(
            f"  Epoch {epoch:>2}/{C.CNN_EPOCHS} | train loss {training_loss:.3f} | val balanced acc {score:.3f}",
            flush=True,
        )
        if score > best_score:
            best_epoch, best_score = epoch, score

    print(f"  Refit on all {len(train):,} training images for {best_epoch} epoch(s)", flush=True)
    model, optimizer, loss = fresh_model(classes, device, seed)
    full_loader = loader(train, y_train, guided, True, seed, device)
    for epoch in range(1, best_epoch + 1):
        fit_loss = train_epoch(model, full_loader, optimizer, loss, device)
        history.append(dict(phase="refit", epoch=epoch, train_loss=fit_loss))

    history_path = save_csv(pd.DataFrame(history), "cnn", "history", f"{stem}.csv")
    scores = predict(model, loader(test, y_test, guided, False, seed, device), device)
    frames = prediction_frame(test, y_test, scores.argmax(axis=1), scores, classes)
    prediction_path = save_csv(frames, "cnn", "predictions", f"{stem}.csv")
    checkpoint = output_path("cnn", "models", f"{stem}.pt")
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(
        dict(
            state_dict=model.cpu().state_dict(),
            classes=classes,
            image_size=C.CNN_SIZE,
            mask_guided=guided,
            selected_epoch=best_epoch,
        ),
        temporary,
    )
    temporary.replace(checkpoint)
    summary, per_class, confusion = evaluate(frames, classes)
    primary = summary.loc[summary.evaluation.eq("leakage_reduced")].iloc[0]
    print(
        f"  Leakage-reduced test | accuracy {primary.accuracy:.3f} | balanced acc {primary.balanced_accuracy:.3f}"
        f" | macro F1 {primary.macro_f1:.3f}",
        flush=True,
    )
    return (
        summary,
        per_class,
        confusion,
        dict(
            selected_epoch=best_epoch,
            selection_balanced_accuracy=best_score,
            train_images=len(train),
            prediction_file=prediction_path.relative_to(C.OUTPUT_DIR).as_posix(),
            model_file=checkpoint.relative_to(C.OUTPUT_DIR).as_posix(),
            history_file=history_path.relative_to(C.OUTPUT_DIR).as_posix(),
        ),
    )


def main():
    if not C.CNN_SEEDS or C.CNN_EPOCHS < 1 or C.CNN_BATCH_SIZE < 1 or C.CNN_SIZE < 32:
        raise ValueError("Check CNN_SEEDS, CNN_EPOCHS, CNN_BATCH_SIZE and CNN_SIZE in config.py.")
    torch.set_num_threads(C.CNN_CPU_THREADS)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if C.CNN_DEVICE == "auto" else C.CNN_DEVICE
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable in this PyCharm interpreter. Set CNN_DEVICE to 'auto' or 'cpu'."
        )
    print(f"Step 6 | Full-image and mask-guided CNNs | {device}", flush=True)
    manifest = load_manifest()
    cohorts = [("tumor3", (("E21", False), ("E22", True)))]
    if C.CNN_RUN_FOUR_CLASS:
        cohorts.append(("image4", (("E21_4", False),)))
    metrics, per_class, confusions, runs, memberships = [], [], [], [], []
    for cohort, experiments in cohorts:
        train, test, classes = cohort_rows(manifest, cohort)
        y_train, y_test = encode(train, classes), encode(test, classes)
        for seed in dict.fromkeys(C.CNN_SEEDS):
            fit_idx, val_idx = grouped_folds(train, y_train, seed)[0]
            partition = train[["sample_id", "label", "group"]].assign(seed=seed, cohort=cohort, role="fit")
            partition.loc[val_idx, "role"] = "validation"
            memberships.append(partition)
            for experiment, guided in experiments:
                summary, class_table, confusion, files = train_variant(
                    train, test, y_train, y_test, classes, fit_idx, val_idx, experiment, guided, seed, device
                )
                desc = "Mask-guided CNN" if guided else "Full-image CNN"
                tags = dict(experiment=experiment, description=desc, cohort=cohort, seed=seed)
                runs.append(dict(**tags, **files))
                metrics.append(summary.assign(**tags, **files))
                per_class.append(class_table.assign(**tags))
                confusions.append(confusion.assign(**tags))
    for tables, name in ((metrics, "metrics"), (per_class, "per_class"), (confusions, "confusions")):
        save_csv(pd.concat(tables, ignore_index=True), "cnn", f"{name}.csv")
    save_csv(pd.DataFrame(runs), "cnn", "models.csv")
    save_csv(pd.concat(memberships, ignore_index=True), "cnn", "selection_folds.csv")
    save_json(
        dict(
            manifest_sha256=manifest_digest(),
            plane_filter=C.PLANE_FILTER,
            seeds=list(C.CNN_SEEDS),
            epochs=C.CNN_EPOCHS,
            image_size=C.CNN_SIZE,
            batch_size=C.CNN_BATCH_SIZE,
            device=str(device),
            cv_folds=C.CV_FOLDS,
            bootstraps=C.BOOTSTRAPS,
            four_class=C.CNN_RUN_FOUR_CLASS,
        ),
        "cnn",
        "settings.json",
    )
    print(f"Saved checkpoints, epoch histories and metrics in {output_path('cnn', 'metrics.csv').parent}")


if __name__ == "__main__":
    main()
