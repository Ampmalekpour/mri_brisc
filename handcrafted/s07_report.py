"""Step 7. Summarize held-out results and compare predictions on matched cases."""

import json

import numpy as np
import pandas as pd

import config as C
from data import manifest_digest, output_path, save_csv
from evaluation import bootstrap_weights, metric_values


PRIMARY = "leakage_reduced"
MATCHES = ("official", PRIMARY)
INCREMENTS = (
    ("E15", "E16", "Adding tumor position"),
    ("E16", "E17", "Adding local tissue context"),
    ("E17", "E19", "Adding distant tissue"),
    ("E19", "E20", "Adding controls and global descriptors"),
)


def read_csv(*parts):
    path = C.OUTPUT_DIR.joinpath(*parts)
    return pd.read_csv(path) if path.exists() else None


def training_selected(metrics):
    ranked = metrics.loc[metrics.evaluation.eq(PRIMARY)].sort_values(
        "cv_balanced_accuracy", ascending=False, kind="stable"
    )
    return ranked.drop_duplicates(["cohort", "experiment"])


def predictions(path):
    return pd.read_csv(C.OUTPUT_DIR / path).set_index("sample_id", verify_integrity=True)


def pair_rows(a, b, evaluation):
    if set(a.index) != set(b.index):
        raise ValueError(
            "Run files contain different sample IDs. Use the same manifest and cohorts for each stage."
        )
    b = b.reindex(a.index)
    for column in ("truth", "train_overlap", "group"):
        if not a[column].equals(b[column]):
            raise ValueError(
                f"Matched run files disagree on {column}. Rerun stages 5 and 6 with the same index."
            )
    valid = ~a.train_overlap if evaluation == PRIMARY else np.ones(len(a), dtype=bool)
    return pd.DataFrame({"truth": a.truth, "group": a.group, "a": a.prediction, "b": b.prediction}).loc[valid]


def balanced(row, a, b):
    return row.groupby("truth")[[a, b]].mean().mean()


def pair_statistics(rows):
    matching = rows.a.eq(rows.b)
    error_a, error_b = rows.a.ne(rows.truth), rows.b.ne(rows.truth)
    both, either = (error_a & error_b).sum(), (error_a | error_b).sum()
    result = dict(
        images=len(rows),
        group_count=rows.group.nunique(),
        agreement=matching.mean(),
        errors_a=error_a.sum(),
        errors_b=error_b.sum(),
        errors_both=both,
        errors_only_a=(error_a & ~error_b).sum(),
        errors_only_b=(error_b & ~error_a).sum(),
        error_jaccard=both / either if either else np.nan,
        accuracy_difference=(~error_a).mean() - (~error_b).mean(),
    )
    correct = rows.assign(correct_a=~error_a, correct_b=~error_b)
    scores = balanced(correct, "correct_a", "correct_b")
    result["balanced_accuracy_difference"] = scores.correct_a - scores.correct_b
    classes = sorted(set(rows.truth) | set(rows.a) | set(rows.b))
    labels = {label: index for index, label in enumerate(classes)}
    k = len(classes)
    y = rows.truth.map(labels).to_numpy()
    matrices = [
        np.bincount(k * y + rows[column].map(labels).to_numpy(), minlength=k**2).reshape(k, k)
        for column in ("a", "b")
    ]
    a_metrics, *_ = metric_values(matrices[0])
    b_metrics, *_ = metric_values(matrices[1])
    result["macro_f1_difference"] = a_metrics["macro_f1"] - b_metrics["macro_f1"]
    if C.BOOTSTRAPS > 0:
        group_index, weights = bootstrap_weights(tuple(rows.group), C.BOOTSTRAPS, C.SEED)
        matrices = []
        for column in ("a", "b"):
            pair = k * y + rows[column].map(labels).to_numpy()
            matrix = np.zeros((weights.shape[1], k**2), dtype=np.float32)
            np.add.at(matrix, (group_index, pair), 1)
            matrices.append(matrix)
        combined = weights @ np.column_stack(matrices)
        a_counts, b_counts = (chunk.reshape(-1, k, k) for chunk in np.hsplit(combined, 2))
        represented = np.bincount(y, minlength=k) > 0
        valid = ((a_counts.sum(axis=2) > 0) == represented).all(axis=1)
        result["bootstrap_replicates_used"] = valid.sum()
        a_metrics, *_ = metric_values(a_counts)
        b_metrics, *_ = metric_values(b_counts)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = (a_metrics[metric] - b_metrics[metric])[valid]
            lower, upper = np.percentile(values, (2.5, 97.5)) if len(values) else (np.nan, np.nan)
            result[f"{metric}_difference_lo"] = lower
            result[f"{metric}_difference_hi"] = upper
    return result


def compare_cnn(winners, cnn):
    records = []
    for model in winners.itertuples(index=False):
        left = predictions(model.prediction_file)
        baseline = cnn.loc[cnn.cohort.eq(model.cohort) & cnn.evaluation.eq(PRIMARY)]
        for other in baseline.itertuples(index=False):
            right = predictions(other.prediction_file)
            for comparison in MATCHES:
                rows = pair_rows(left, right, comparison)
                records.append(
                    dict(
                        experiment=model.experiment,
                        model=model.model,
                        cohort=model.cohort,
                        cnn=other.experiment,
                        seed=other.seed,
                        evaluation=comparison,
                        **pair_statistics(rows),
                    )
                )
    return pd.DataFrame(records)


def incremental(classical):
    rows = []
    available = classical[classical.evaluation.eq(PRIMARY)]
    for old, new, description in INCREMENTS:
        left, right = (available[available.experiment.eq(experiment)] for experiment in (old, new))
        shared = left.merge(right, on="model", suffixes=("_before", "_after"))
        for model in shared.itertuples(index=False):
            after, before = (
                predictions(model.prediction_file_after),
                predictions(model.prediction_file_before),
            )
            for comparison in MATCHES:
                cases = pair_rows(after, before, comparison)
                rows.append(
                    dict(
                        before=old,
                        after=new,
                        model=model.model,
                        change=description,
                        evaluation=comparison,
                        cv_balanced_accuracy_gain=model.cv_balanced_accuracy_after
                        - model.cv_balanced_accuracy_before,
                        **pair_statistics(cases),
                    )
                )
    return pd.DataFrame(rows)


def markdown_table(frame, columns):
    if frame.empty:
        return "No rows available.\n"
    pretty = frame[columns].map(
        lambda value: (
            "—"
            if pd.isna(value)
            else f"{value:.3f}"
            if isinstance(value, float)
            else str(value).replace("|", "/")
        )
    )
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in pretty.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows]) + "\n"


def plot_top(winners, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = winners.loc[winners.cohort.eq("tumor3")].head(15).iloc[::-1]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(subset) + 2))
    y = np.arange(len(subset))
    ax.barh(y + 0.18, subset.cv_balanced_accuracy, height=0.34, label="Grouped training CV")
    ax.barh(y - 0.18, subset.balanced_accuracy, height=0.34, label="Leakage-reduced test")
    ax.set_yticks(y, [f"{row.experiment} {row.model}" for row in subset.itertuples()])
    ax.set_xlabel("Balanced accuracy")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    classical = read_csv("classical", "metrics.csv")
    cnn = read_csv("cnn", "metrics.csv")
    if classical is None and cnn is None:
        raise FileNotFoundError("Run s05_train_classical.py or s06_train_cnn.py first.")
    run_settings = {}
    for stage, metrics in (("classical", classical), ("cnn", cnn)):
        if metrics is not None:
            path = C.OUTPUT_DIR / stage / "settings.json"
            if not path.exists():
                raise FileNotFoundError(f"Run the {stage} training stage to generate {path.name}.")
            run_settings[stage] = json.loads(path.read_text(encoding="utf-8"))
            if run_settings[stage]["manifest_sha256"] != manifest_digest():
                raise ValueError(f"The {stage} results use a different manifest. Rerun the training stage.")
    print("Step 7 | Held-out metrics and paired comparisons", flush=True)
    coverage = read_csv("index", "coverage.csv")
    sections = [
        "# BRISC handcrafted feature study",
        "",
        "The primary test excludes official test cases in a duplicate or patient group that touches training.",
        "The official split is also reported for comparability. Model selection uses grouped folds from training.",
        "Tumor3 includes glioma, meningioma and pituitary cases with usable annotations.",
        "Image4 uses image-derived inputs and includes the no-tumor class.",
        "",
    ]
    metrics_columns = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"]
    if coverage is not None:
        totals = coverage.groupby(["split", "label"], as_index=False).images.sum()
        sections.extend(
            ["## Indexed sample counts", "", markdown_table(totals, ["split", "label", "images"])]
        )
    if classical is not None:
        winners = training_selected(classical)
        save_csv(winners, "report", "classical_selected.csv")
        plot_top(winners, output_path("report", "top_classical.png"))
        per_class = read_csv("classical", "per_class.csv")
        selected_per_class = per_class.merge(
            winners[["experiment", "cohort", "model"]], on=["experiment", "cohort", "model"]
        )
        save_csv(selected_per_class, "report", "classical_selected_per_class.csv")
        columns = [
            "experiment",
            "description",
            "cohort",
            "model",
            "cv_balanced_accuracy",
            "images",
            *metrics_columns,
        ]
        sections.extend(
            [
                "## Classical models selected by cross-validation",
                "",
                markdown_table(winners, columns),
                "Class-wise precision, recall, and F1 are in `classical_selected_per_class.csv`.",
            ]
        )
        print(
            winners[["experiment", "model", "cv_balanced_accuracy", "balanced_accuracy"]]
            .head(12)
            .to_string(index=False, float_format=lambda number: f"{number:.3f}")
        )
        increments = incremental(classical)
        if not increments.empty:
            save_csv(increments, "report", "named_increments.csv")
            primary = increments[increments.evaluation.eq(PRIMARY)]
            sections.extend(
                [
                    "",
                    "## Paired changes after adding feature groups",
                    "",
                    markdown_table(
                        primary,
                        [
                            "before",
                            "after",
                            "model",
                            "change",
                            "cv_balanced_accuracy_gain",
                            "balanced_accuracy_difference",
                        ],
                    ),
                ]
            )
        permuted = read_csv("classical", "permutation.csv")
        if permuted is not None and not permuted.empty:
            averaged = (
                permuted.groupby(["experiment", "model", "group"])
                .balanced_accuracy_drop.agg(["mean", "std"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            save_csv(averaged, "report", "group_permutation.csv")
            sections.extend(
                [
                    "",
                    "## Group permutation importance on leakage-reduced test",
                    "",
                    markdown_table(averaged, ["experiment", "model", "group", "mean", "std"]),
                ]
            )
        shapley = (
            read_csv("classical", "family_shapley.csv")
            if run_settings["classical"]["exhaustive_subsets"]
            else None
        )
        if shapley is not None:
            sections.extend(
                [
                    "",
                    "## Feature-family Shapley contributions",
                    "",
                    markdown_table(shapley, ["family", "shapley_cv_balanced_accuracy"]),
                ]
            )

    if cnn is not None:
        primary = cnn[cnn.evaluation.eq(PRIMARY)]
        columns = ["experiment", "description", "cohort", "seed", "images", *metrics_columns]
        sections.extend(["", "## CNN baselines by seed", "", markdown_table(primary, columns)])
        average = (
            cnn[cnn.evaluation.isin(MATCHES)]
            .groupby(["experiment", "cohort", "evaluation"])[metrics_columns]
            .agg(["mean", "std"])
        )
        average.columns = [f"{metric}_{stat}" for metric, stat in average.columns]
        save_csv(average.reset_index(), "report", "cnn_seed_summary.csv")
        if classical is not None:
            paired = compare_cnn(winners, cnn)
            if not paired.empty:
                save_csv(paired, "report", "cnn_paired_comparisons.csv")
                sections.extend(
                    [
                        "",
                        "## Matched CNN comparisons",
                        "",
                        "The paired CSV contains handcrafted minus CNN accuracy, grouped confidence intervals, "
                        "prediction agreement and the intersection of model errors for each seed and split.",
                    ]
                )
    sections.extend(
        [
            "",
            "See the stage metrics CSVs for both test splits, per-plane scores and group-bootstrap intervals.",
            "High feature-group performance measures predictive information in that group. "
            "Use the repository's intervention experiments to study which cues a CNN relies on.",
            "",
        ]
    )
    path = output_path("report", "REPORT.md")
    path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved report and comparison tables in {path.parent}")


if __name__ == "__main__":
    main()
