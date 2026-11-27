"""Duplicate-aware validation and metrics for both ML and CNN stages."""

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import config as C
from data import PLANES


def cohort_rows(manifest, cohort):
    if cohort == "tumor3":
        rows = manifest[manifest.label.isin(C.TUMOR_CLASSES) & manifest.mask_usable]
        classes = C.TUMOR_CLASSES
    else:
        rows = manifest[manifest.label.isin(C.ALL_CLASSES)]
        classes = C.ALL_CLASSES
    if C.PLANE_FILTER is not None:
        if C.PLANE_FILTER not in PLANES:
            raise ValueError(f"PLANE_FILTER must be one of {PLANES} or None.")
        rows = rows.loc[rows.plane.eq(C.PLANE_FILTER)]
    train = rows.loc[rows.split.eq(C.TRAIN_SPLIT)].reset_index(drop=True)
    test = rows.loc[rows.split.eq(C.TEST_SPLIT)].reset_index(drop=True)
    for name, table in (("training", train), ("official test", test)):
        missing = set(classes) - set(table.label)
        if missing:
            raise ValueError(f"{cohort} {name} is missing {sorted(missing)}. See index/coverage.csv.")
    if not (~test.train_overlap).any():
        raise ValueError(
            f"{cohort} has no leakage-reduced test cases. Inspect index/groups.csv and DHASH_MAX_DISTANCE."
        )
    return train, test, list(classes)


def encode(rows, classes):
    return rows.label.map({name: index for index, name in enumerate(classes)}).to_numpy(dtype=int)


def grouped_folds(train, target, seed=C.SEED):
    expected = len(np.unique(target))
    groups = train.group.to_numpy()
    counts = train.groupby("label").group.nunique()
    for count in range(min(C.CV_FOLDS, int(counts.min())), 1, -1):
        for offset in range(12):
            cv = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed + offset)
            folds = list(cv.split(np.zeros(len(target)), target, groups))
            if all(
                len(np.unique(target[fit])) == expected and len(np.unique(target[val])) == expected
                for fit, val in folds
            ):
                return folds
    raise ValueError(
        "Could not form grouped training/validation folds with all classes. Inspect index/groups.csv."
    )


def metric_values(matrix):
    support = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)
    hits = np.diagonal(matrix, axis1=-2, axis2=-1)
    recall = np.divide(hits, support, out=np.full(support.shape, np.nan), where=support > 0)
    precision = np.divide(hits, predicted, out=np.zeros(predicted.shape, dtype=float), where=predicted > 0)
    f1 = np.divide(
        2 * hits,
        support + predicted,
        out=np.zeros(support.shape, dtype=float),
        where=(support + predicted) > 0,
    )
    f1 = np.where(support > 0, f1, np.nan)
    total = support.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = hits.sum(axis=-1) / total
    valid_classes = (support > 0).sum(axis=-1)
    balanced = np.divide(
        np.nansum(recall, axis=-1),
        valid_classes,
        out=np.full(np.shape(accuracy), np.nan),
        where=valid_classes > 0,
    )
    macro_f1 = np.divide(
        np.nansum(f1, axis=-1),
        valid_classes,
        out=np.full(np.shape(accuracy), np.nan),
        where=valid_classes > 0,
    )
    return {"accuracy": accuracy, "balanced_accuracy": balanced, "macro_f1": macro_f1}, precision, recall, f1


@lru_cache(maxsize=8)
def bootstrap_weights(groups, repeats, seed):
    _, group_index = np.unique(groups, return_inverse=True)
    count = int(group_index.max()) + 1
    weights = np.random.default_rng(seed).multinomial(count, np.full(count, 1 / count), size=repeats)
    return group_index, weights.astype(np.float32)


def prediction_frame(rows, truth, predicted, scores, classes):
    table = rows[["sample_id", "plane", "group", "train_overlap"]].copy()
    labels = np.asarray(classes)
    table["truth"] = labels[truth]
    table["prediction"] = labels[predicted]
    for index, label in enumerate(classes):
        table[f"score_{label}"] = scores[:, index]
    return table


def evaluate(predictions, classes, bootstraps=True):
    results, class_rows, confusion_rows = [], [], []
    label_index = {name: index for index, name in enumerate(classes)}
    k = len(classes)
    clean = predictions.loc[~predictions.train_overlap]
    views = [("official", predictions), ("leakage_reduced", clean)]
    views += [(f"leakage_reduced_{plane.lower()}", table) for plane, table in clean.groupby("plane")]
    for split, table in views:
        if table.empty:
            continue
        y = table.truth.map(label_index).to_numpy()
        pred = table.prediction.map(label_index).to_numpy()
        pairs = k * y + pred
        matrix = np.bincount(pairs, minlength=k**2).reshape(k, k)
        metrics, precision, recall, f1 = metric_values(matrix)
        scores = table[[f"score_{label}" for label in classes]].to_numpy()
        support = matrix.sum(axis=1)
        auc = np.nan
        if (support > 0).all():
            auc = np.mean([roc_auc_score(y == index, scores[:, index]) for index in range(k)])
        groups = tuple(table.group)
        entry = dict(
            evaluation=split,
            images=len(table),
            groups=len(set(groups)),
            classes_present=(support > 0).sum(),
            macro_auroc=auc,
            **metrics,
        )
        if bootstraps and C.BOOTSTRAPS > 0 and split in ("official", "leakage_reduced"):
            group_index, weights = bootstrap_weights(groups, C.BOOTSTRAPS, C.SEED)
            group_confusions = np.zeros((weights.shape[1], k**2), dtype=np.float32)
            np.add.at(group_confusions, (group_index, pairs), 1)
            resampled = (weights @ group_confusions).reshape(-1, k, k)
            bootstrap_metrics, *_ = metric_values(resampled)
            # Keep bootstrap replicates with the same represented classes as the original view.
            usable = ((resampled.sum(axis=2) > 0) == (support > 0)).all(axis=1)
            entry["bootstrap_replicates_used"] = int(usable.sum())
            for metric, values in bootstrap_metrics.items():
                lo, hi = np.percentile(values[usable], (2.5, 97.5)) if usable.any() else (np.nan, np.nan)
                entry[f"{metric}_lo"] = lo
                entry[f"{metric}_hi"] = hi
        results.append(entry)
        for index, label in enumerate(classes):
            class_rows.append(
                dict(
                    evaluation=split,
                    label=label,
                    support=support[index],
                    precision=precision[index],
                    recall=recall[index],
                    f1=f1[index],
                )
            )
            for j, predicted in enumerate(classes):
                confusion_rows.append(
                    dict(evaluation=split, truth=label, prediction=predicted, images=matrix[index, j])
                )
    return pd.DataFrame(results), pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)
