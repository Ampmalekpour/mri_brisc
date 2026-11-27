"""Step 5. Fit classical models using grouped CV, then score the official test set."""

import itertools
import json
import math
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

import config as C
from data import feature_settings, load_manifest, manifest_digest, output_path, save_csv, save_json
from evaluation import cohort_rows, encode, evaluate, grouped_folds, prediction_frame
from experiments import EXACT_COLUMNS, FAMILIES, all_experiments, subsets
from models import training_pipeline
from visual import PATCH_DIM


class FeatureStore:
    def __init__(self, manifest):
        self.ids = pd.Index(manifest.sample_id)
        self.groups = {}
        self.patches = None
        self.manifest_sha = manifest_digest()
        self.validated = set()

    def verify_stage(self, group):
        step = 2 if int(group[1:3]) <= 11 else 3 if int(group[1:3]) <= 20 else 4
        if step not in self.validated:
            path = output_path("features", f"step{step}.json")
            if not path.exists():
                raise FileNotFoundError(f"Run feature extraction step {step} first. Missing {path}")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["manifest_sha256"] != self.manifest_sha or record["settings"] != feature_settings(step):
                raise ValueError(
                    f"The index or extraction settings changed. Rerun feature extraction step {step}."
                )
            self.validated.add(step)

    def load(self, group):
        if group not in self.groups:
            self.verify_stage(group)
            path = output_path("features", f"{group}.csv")
            if not path.exists():
                step = 2 if int(group[1:3]) <= 11 else 3 if int(group[1:3]) <= 20 else 4
                raise FileNotFoundError(f"Run feature extraction step {step} first. Missing {path}")
            data = pd.read_csv(path, index_col="sample_id")
            if data.index.has_duplicates or not self.ids.equals(pd.Index(data.index)):
                raise ValueError(f"{path.name} does not match the index. Rerun its feature extraction stage.")
            self.groups[group] = data.astype(float)
        return self.groups[group]

    def load_patches(self):
        if self.patches is None:
            self.verify_stage("F23")
            path = output_path("features", "F23_patches.npz")
            if not path.exists():
                raise FileNotFoundError(f"Run s04_context.py first. Missing {path}")
            with np.load(path) as file:
                unknown = set(file.files) - set(self.ids)
                if unknown:
                    raise ValueError(f"{path.name} does not match the index. Rerun s04_context.py.")
                self.patches = {name: file[name] for name in file.files}
        return self.patches

    def matrix(self, spec, sample_ids):
        parts = defaultdict(dict)
        measurement_columns = {}
        for selector in spec.groups:
            group, _, subset = selector.partition(":")
            data = self.load(group)
            if selector in EXACT_COLUMNS:
                columns = [name for name in EXACT_COLUMNS[selector] if name in data.columns]
            elif selector == "F06:intrinsic":
                columns = [col for col in data if not any(word in col for word in ("head", "image"))]
            elif group == "F21" and subset.endswith("ring_only"):
                prefix = subset.removesuffix("ring_only")
                columns = [
                    col
                    for col in data
                    if col.startswith(prefix) and not col.endswith(("tumor_minus_ring", "tumor_to_ring"))
                ]
            else:
                columns = [name for name in data if name.startswith(subset)]
            if not columns:
                print(f"    {selector} has no columns for the indexed images.", flush=True)
            for column in columns:
                parts[group][column] = data[column]
        frames = []
        for group, columns in parts.items():
            part = pd.DataFrame(columns, index=self.ids).loc[sample_ids].add_prefix(f"{group}__")
            frames.append(part)
            measurement_columns[group] = list(part.columns)
        x = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=sample_ids)
        numeric = list(x.columns)
        if spec.vocabulary:
            empty = np.empty((0, PATCH_DIM), np.float32)
            patches = self.load_patches()
            x["visual_patches"] = [patches.get(name, empty) for name in sample_ids]
            measurement_columns.setdefault("F23", []).append("visual_patches")
        return x, numeric, measurement_columns


def cohort_cache(manifest, kind):
    train, test, classes = cohort_rows(manifest, kind)
    y_train, y_test = encode(train, classes), encode(test, classes)
    folds = grouped_folds(train, y_train)
    memberships = [
        train.iloc[val][["sample_id", "label", "group"]].assign(validation_fold=number)
        for number, (_, val) in enumerate(folds, 1)
    ]
    save_csv(pd.concat(memberships, ignore_index=True), "classical", f"folds_{kind}.csv")
    print(
        f"{kind}: {len(train):,} train, {len(test):,} official test, "
        f"{(~test.train_overlap).sum():,} leakage-reduced test, {len(folds)} grouped folds.",
        flush=True,
    )
    return train, test, classes, y_train, y_test, folds


def fit_model(name, numeric, spec, x_train, y_train, folds):
    pipeline, grid = training_pipeline(
        name, numeric, spec.vocabulary, spec.pca, min(len(fit) for fit, _ in folds)
    )
    search = GridSearchCV(
        pipeline,
        grid or [{}],
        scoring="balanced_accuracy",
        cv=folds,
        n_jobs=C.MODEL_JOBS,
        refit=True,
        error_score="raise",
    )
    with parallel_backend("loky", inner_max_num_threads=1), threadpool_limits(limits=1):
        search.fit(x_train, y_train)
    sd = search.cv_results_["std_test_score"][search.best_index_]
    return search.best_estimator_, search.best_score_, sd, search.best_params_, search.cv_results_


def scores(estimator, x):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)
    return estimator.decision_function(x)


def group_permutation(estimator, x, y, blocks):
    if C.PERMUTATION_REPEATS <= 0 or not len(x):
        return []
    rng = np.random.default_rng(C.SEED)
    base = balanced_accuracy_score(y, estimator.predict(x))
    results = []
    with threadpool_limits(limits=1):
        for group, columns in blocks.items():
            for repeat in range(C.PERMUTATION_REPEATS):
                scrambled = x.copy()
                shuffled = x[columns].iloc[rng.permutation(len(x))].set_axis(x.index)
                scrambled[columns] = shuffled
                score = balanced_accuracy_score(y, estimator.predict(scrambled))
                results.append(
                    dict(
                        group=group,
                        repeat=repeat + 1,
                        baseline=base,
                        permuted_balanced_accuracy=score,
                        balanced_accuracy_drop=base - score,
                    )
                )
    return results


def fit_study(manifest, registry, chosen):
    store = FeatureStore(manifest)
    cache = {}
    metric_rows, class_rows, confusion_rows, permutation_rows, model_rows = [], [], [], [], []
    diagnostic_candidate = next(
        (key for key in ("E20", "E19", "E17", "E16", "E15", "E07", "E06") if key in chosen), None
    )
    if diagnostic_candidate is None:
        diagnostic_candidate = next((key for key in chosen if len(registry[key].groups) > 1), None)

    for experiment_id in chosen:
        spec = registry[experiment_id]
        if spec.cohort not in cache:
            cache[spec.cohort] = cohort_cache(manifest, spec.cohort)
        train, test, classes, y_train, y_test, folds = cache[spec.cohort]
        sample_ids = pd.Index([*train.sample_id, *test.sample_id])
        x, numeric, blocks = store.matrix(spec, sample_ids)
        x_train, x_test = x.iloc[: len(train)], x.iloc[len(train) :]
        names = list(C.MODELS)
        if experiment_id in {"E00", "E03", "E04", "E04N", "E05"} and "tree" not in names:
            names.append("tree")
        print(
            f"\n{experiment_id} | {spec.name} | {len(numeric):,} fixed columns"
            + (f" and {spec.vocabulary} visual encoding" if spec.vocabulary else ""),
            flush=True,
        )
        if not x_train.shape[1] or (not spec.vocabulary and not x_train.notna().any().any()):
            print("  No usable training measurements. See the feature summaries.", flush=True)
            continue
        best = None
        for name in names:
            model, cv, cv_sd, params, cv_results = fit_model(name, numeric, spec, x_train, y_train, folds)
            with threadpool_limits(limits=1):
                predictions = prediction_frame(
                    test, y_test, model.predict(x_test), scores(model, x_test), classes
                )
            stem = f"{experiment_id}_{name}"
            prediction_path = save_csv(predictions, "classical", "predictions", f"{stem}.csv")
            artifact = output_path("classical", "models", f"{stem}.joblib")
            temporary = artifact.with_suffix(".tmp")
            score_type = "probability" if hasattr(model, "predict_proba") else "decision_function"
            joblib.dump(
                dict(
                    pipeline=model,
                    classes=classes,
                    columns=numeric,
                    group_selectors=spec.groups,
                    score_type=score_type,
                ),
                temporary,
            )
            temporary.replace(artifact)
            save_csv(pd.DataFrame(cv_results), "classical", "cv", f"{stem}.csv")
            base = dict(
                experiment=experiment_id,
                description=spec.name,
                cohort=spec.cohort,
                model=name,
                cv_balanced_accuracy=cv,
                cv_sd=cv_sd,
                best_parameters=json.dumps(params),
                train_images=len(train),
                cv_folds=len(folds),
                score_type=score_type,
                prediction_file=prediction_path.relative_to(C.OUTPUT_DIR).as_posix(),
                model_file=artifact.relative_to(C.OUTPUT_DIR).as_posix(),
            )
            model_rows.append(base)
            summary, per_class, confusion = evaluate(predictions, classes)
            metric_rows.append(summary.assign(**base))
            class_rows.append(per_class.assign(experiment=experiment_id, cohort=spec.cohort, model=name))
            confusion_rows.append(confusion.assign(experiment=experiment_id, cohort=spec.cohort, model=name))
            clean = summary.loc[summary.evaluation.eq("leakage_reduced")].iloc[0]
            print(
                f"  {name:<14} CV balanced acc {cv:.3f} | clean acc {clean.accuracy:.3f}, "
                f"balanced acc {clean.balanced_accuracy:.3f}, macro F1 {clean.macro_f1:.3f}",
                flush=True,
            )
            if best is None or cv > best[0]:
                best = (cv, name, model)
        if experiment_id == diagnostic_candidate and best is not None:
            retained = ~test.train_overlap.to_numpy()
            for row in group_permutation(best[2], x_test.iloc[retained], y_test[retained], blocks):
                permutation_rows.append(dict(experiment=experiment_id, model=best[1], **row))

    if not metric_rows:
        raise ValueError("No configured experiment could be fitted. Check the index and feature coverage.")
    for rows, filename in (
        (metric_rows, "metrics"),
        (class_rows, "per_class"),
        (confusion_rows, "confusions"),
    ):
        save_csv(pd.concat(rows, ignore_index=True), "classical", f"{filename}.csv")
    save_csv(pd.DataFrame(model_rows), "classical", "models.csv")
    save_csv(
        pd.DataFrame(
            permutation_rows,
            columns=[
                "experiment",
                "model",
                "group",
                "repeat",
                "baseline",
                "permuted_balanced_accuracy",
                "balanced_accuracy_drop",
            ],
        ),
        "classical",
        "permutation.csv",
    )
    return store, cache


def fit_subsets(manifest, store, cache):
    (train, test, classes, y_train, y_test, folds) = cache.get("tumor3") or cohort_cache(manifest, "tumor3")
    sample_ids = pd.Index([*train.sample_id, *test.sample_id])
    results, cv_scores = [], {}
    for experiment_id, keys, spec in subsets():
        x, numeric, _ = store.matrix(spec, sample_ids)
        x_train, x_test = x.iloc[: len(train)], x.iloc[len(train) :]
        name = C.SUBSET_MODEL if keys else "dummy"
        model, cv, cv_sd, params, _ = fit_model(name, numeric, spec, x_train, y_train, folds)
        with threadpool_limits(limits=1):
            predictions = prediction_frame(
                test, y_test, model.predict(x_test), scores(model, x_test), classes
            )
        report, _, _ = evaluate(predictions, classes, bootstraps=False)
        for row in report.loc[report.evaluation.isin(("official", "leakage_reduced"))].to_dict("records"):
            results.append(
                dict(
                    subset=experiment_id,
                    families=" + ".join(keys),
                    model=name,
                    cv_balanced_accuracy=cv,
                    cv_sd=cv_sd,
                    best_parameters=json.dumps(params),
                    **row,
                )
            )
        cv_scores[frozenset(keys)] = cv
        print(f"  {experiment_id} {spec.name} | CV balanced acc {cv:.3f}", flush=True)
    save_csv(pd.DataFrame(results), "classical", "all_subsets.csv")
    contributions, increments = [], []
    families = list(FAMILIES)
    size = len(families)
    for name in families:
        others = [key for key in families if key != name]
        value = 0.0
        for count in range(size):
            weight = 1 / (size * math.comb(size - 1, count))
            for keys in itertools.combinations(others, count):
                base = frozenset(keys)
                gain = cv_scores[base | {name}] - cv_scores[base]
                value += weight * gain
                increments.append(
                    dict(added_family=name, base_families=" + ".join(keys), cv_balanced_accuracy_gain=gain)
                )
        contributions.append(dict(family=name, shapley_cv_balanced_accuracy=value))
    save_csv(pd.DataFrame(contributions), "classical", "family_shapley.csv")
    save_csv(pd.DataFrame(increments), "classical", "family_increments.csv")


def main():
    manifest = load_manifest()
    if not C.MODELS or C.CV_FOLDS < 2 or not C.VOCABULARY_SIZES or min(C.VOCABULARY_SIZES) < 1:
        raise ValueError("Configure at least one model, two CV folds and a positive visual vocabulary size.")
    registry = all_experiments()
    chosen = list(dict.fromkeys(C.EXPERIMENT_IDS or tuple(registry)))
    if unknown := set(chosen) - set(registry):
        raise ValueError(
            f"Unknown classical experiment IDs: {sorted(unknown)}. E21 and E22 run in s06_train_cnn.py."
        )
    print(f"Step 5 | Classical classifiers | {len(chosen)} experiments", flush=True)
    store, cache = fit_study(manifest, registry, chosen)
    if C.RUN_EXHAUSTIVE_SUBSETS:
        print("\nEight-family subset study | 255 combinations and the baseline", flush=True)
        fit_subsets(manifest, store, cache)
    save_json(
        dict(
            manifest_sha256=manifest_digest(),
            plane_filter=C.PLANE_FILTER,
            experiments=chosen,
            models=list(C.MODELS),
            seed=C.SEED,
            cv_folds=C.CV_FOLDS,
            tuned=C.TUNE_MODELS,
            bootstraps=C.BOOTSTRAPS,
            lda_max_components=C.LDA_MAX_COMPONENTS,
            vocabulary_sizes=list(C.VOCABULARY_SIZES),
            pca_components=list(C.PCA_COMPONENTS),
            exhaustive_subsets=C.RUN_EXHAUSTIVE_SUBSETS,
            feature_settings=feature_settings(),
        ),
        "classical",
        "settings.json",
    )
    print(
        f"Saved held-out metrics, predictions, models and CV results in {output_path('classical', 'metrics.csv').parent}"
    )


if __name__ == "__main__":
    main()
