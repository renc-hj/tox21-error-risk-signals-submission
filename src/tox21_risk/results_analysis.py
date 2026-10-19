"""Create final report tables and figures from locked experiment outputs."""

from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tox21_risk.graph_data import file_sha256
from tox21_risk.split_workflow import SPLIT_COLUMNS
from tox21_risk.uncertainty import error_risk_metrics
from tox21_risk.uncertainty_workflow import SIMILARITY_BIN_LABELS


STRATEGY_LABELS = {
    "random": "Random",
    "scaffold": "Scaffold",
    "similarity_cluster": "Similarity-cluster",
}
SIGNAL_LABELS = {
    "structural_distance": "Structural distance",
    "single_pass_uncertainty": "Single-pass uncertainty",
    "mc_dropout_mi": "MC Dropout MI",
}
MODEL_COLORS = {"LR": "#4C78A8", "GCN": "#F58518"}
SIGNAL_COLORS = {
    "structural_distance": "#54A24B",
    "single_pass_uncertainty": "#4C78A8",
    "mc_dropout_mi": "#E45756",
}
MIN_FIGURE_BIN_MOLECULES = 20
MIN_PAIRED_CELL_OBSERVED = 20
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 2026


def cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    *,
    value_column: str,
    cluster_column: str,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Estimate a percentile CI while resampling whole endpoint clusters."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    usable = frame[[cluster_column, value_column]].dropna()
    clusters = usable[cluster_column].drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return float("nan"), float("nan")

    values_by_cluster = {
        cluster: usable.loc[
            usable[cluster_column].eq(cluster), value_column
        ].to_numpy(dtype=float)
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_values = np.concatenate(
            [values_by_cluster[cluster] for cluster in sampled_clusters]
        )
        bootstrap_means[index] = sampled_values.mean()

    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means, [tail_probability, 1.0 - tail_probability]
    )
    return float(lower), float(upper)


def run_results_analysis(
    lr_summary_path: Path,
    gcn_summary_path: Path,
    predictions_path: Path,
    risk_endpoint_metrics_path: Path,
    similarity_bins_path: Path,
    table_dir: Path,
    figure_dir: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Write the final RQ1/RQ2/ablation tables, figures, and prose summary."""

    lr_summary = pd.read_csv(lr_summary_path)
    gcn_summary = pd.read_csv(gcn_summary_path)
    predictions = pd.read_csv(predictions_path)
    risk_metrics = pd.read_csv(risk_endpoint_metrics_path)
    similarity_bins = pd.read_csv(similarity_bins_path)
    if "test" not in set(predictions["partition"]):
        raise ValueError("MC Dropout predictions must contain the test partition")
    predictions["similarity_bin"] = pd.Categorical(
        predictions["similarity_bin"],
        categories=SIMILARITY_BIN_LABELS,
        ordered=True,
    )
    if predictions["similarity_bin"].isna().any():
        raise ValueError("predictions contain an unknown similarity bin")

    model_performance = combine_model_performance(lr_summary, gcn_summary)
    distance_endpoint = distance_relationship_metrics(predictions)
    distance_summary = summarise_distance_relationships(distance_endpoint)
    within_bin_endpoint = within_similarity_bin_metrics(predictions)
    within_bin_summary = summarise_within_bin_metrics(within_bin_endpoint)
    within_bin_paired = paired_within_bin_comparison(
        within_bin_endpoint, minimum_observed=MIN_PAIRED_CELL_OBSERVED
    )
    within_bin_paired_summary = summarise_paired_within_bin_comparison(
        within_bin_paired
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        "model_performance_comparison.csv": model_performance,
        "distance_relationship_endpoint_metrics.csv": distance_endpoint,
        "distance_relationship_summary.csv": distance_summary,
        "within_similarity_bin_endpoint_metrics.csv": within_bin_endpoint,
        "within_similarity_bin_summary.csv": within_bin_summary,
        "within_similarity_bin_paired_summary.csv": within_bin_paired_summary,
    }
    for filename, frame in tables.items():
        frame.to_csv(table_dir / filename, index=False, lineterminator="\n")

    _plot_model_performance(
        model_performance, figure_dir / "model_performance_by_split.png"
    )
    _plot_similarity_relationships(
        similarity_bins, figure_dir / "similarity_error_uncertainty.png"
    )
    _plot_risk_signal_comparison(
        risk_metrics, figure_dir / "error_risk_signal_comparison.png"
    )
    summary_path.write_text(
        _results_summary_markdown(
            model_performance,
            distance_summary,
            risk_metrics,
            within_bin_paired_summary,
        ),
        encoding="utf-8",
    )

    metadata: dict[str, object] = {
        "analysis": "final_results",
        "primary_partition": "test",
        "n_prediction_rows": len(predictions),
        "aggregation_unit": "endpoint-seed",
        "minimum_molecules_per_displayed_similarity_bin": (
            MIN_FIGURE_BIN_MOLECULES
        ),
        "minimum_observed_labels_per_paired_cell": MIN_PAIRED_CELL_OBSERVED,
        "paired_difference_aggregation": (
            "mean_within_endpoint_then_macro_across_endpoints"
        ),
        "paired_difference_ci": {
            "method": "endpoint_cluster_percentile_bootstrap",
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "n_resamples": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "python_version": platform.python_version(),
        "inputs": {
            "lr_summary_sha256": file_sha256(lr_summary_path),
            "gcn_summary_sha256": file_sha256(gcn_summary_path),
            "predictions_sha256": file_sha256(predictions_path),
            "risk_endpoint_metrics_sha256": file_sha256(
                risk_endpoint_metrics_path
            ),
            "similarity_bins_sha256": file_sha256(similarity_bins_path),
        },
        "tables": list(tables),
        "figures": [
            "model_performance_by_split.png",
            "similarity_error_uncertainty.png",
            "error_risk_signal_comparison.png",
        ],
    }
    (table_dir / "results_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def combine_model_performance(
    lr_summary: pd.DataFrame, gcn_summary: pd.DataFrame
) -> pd.DataFrame:
    """Align LR single-fit and GCN repeated-fit performance summaries."""

    _require_columns(
        lr_summary,
        {
            "strategy",
            "partition",
            "macro_auroc",
            "macro_auprc",
            "macro_brier",
        },
        "LR summary",
    )
    _require_columns(
        gcn_summary,
        {
            "strategy",
            "partition",
            "n_seeds",
            "macro_auroc_mean",
            "macro_auroc_std",
            "macro_auprc_mean",
            "macro_auprc_std",
            "macro_brier_mean",
            "macro_brier_std",
        },
        "GCN summary",
    )
    lr = pd.DataFrame(
        {
            "strategy": lr_summary["strategy"],
            "partition": lr_summary["partition"],
            "model": "LR",
            "n_runs": 1,
            "auroc_mean": lr_summary["macro_auroc"],
            "auroc_std": np.nan,
            "auprc_mean": lr_summary["macro_auprc"],
            "auprc_std": np.nan,
            "brier_mean": lr_summary["macro_brier"],
            "brier_std": np.nan,
        }
    )
    gcn = pd.DataFrame(
        {
            "strategy": gcn_summary["strategy"],
            "partition": gcn_summary["partition"],
            "model": "GCN",
            "n_runs": gcn_summary["n_seeds"].astype(int),
            "auroc_mean": gcn_summary["macro_auroc_mean"],
            "auroc_std": gcn_summary["macro_auroc_std"],
            "auprc_mean": gcn_summary["macro_auprc_mean"],
            "auprc_std": gcn_summary["macro_auprc_std"],
            "brier_mean": gcn_summary["macro_brier_mean"],
            "brier_std": gcn_summary["macro_brier_std"],
        }
    )
    return pd.concat([lr, gcn], ignore_index=True)


def filter_supported_similarity_bins(
    summary: pd.DataFrame, *, minimum_molecules: int
) -> pd.DataFrame:
    """Keep adequately supported bins for plotting without altering tables."""

    if minimum_molecules <= 0:
        raise ValueError("minimum_molecules must be positive")
    _require_columns(
        summary, {"n_unique_molecules"}, "similarity-bin summary"
    )
    return summary.loc[
        summary["n_unique_molecules"] >= minimum_molecules
    ].copy()


def distance_relationship_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Measure how structural distance relates separately to BCE and MI."""

    _require_columns(
        predictions,
        {
            "strategy",
            "seed",
            "partition",
            "endpoint",
            "endpoint_bce",
            "structural_distance",
            "mc_dropout_mi",
        },
        "MC Dropout predictions",
    )
    rows: list[dict[str, float | int | str]] = []
    group_columns = ["strategy", "seed", "partition", "endpoint"]
    for keys, group in predictions.groupby(group_columns, sort=False):
        for relationship, outcome in (
            ("structural_distance_vs_bce", "endpoint_bce"),
            ("structural_distance_vs_mi", "mc_dropout_mi"),
        ):
            valid = np.isfinite(group["structural_distance"]) & np.isfinite(
                group[outcome]
            )
            distance = group.loc[valid, "structural_distance"].to_numpy()
            values = group.loc[valid, outcome].to_numpy()
            row = dict(zip(group_columns, keys, strict=True))
            row.update(
                {
                    "relationship": relationship,
                    "n": int(valid.sum()),
                    "spearman": _safe_spearman(distance, values),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_distance_relationships(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate endpoint-seed distance correlations without pooling rows."""

    return (
        metrics.groupby(["strategy", "partition", "relationship"], sort=False)
        .agg(
            n_seed_endpoint=("endpoint", "size"),
            n_valid=("spearman", "count"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
        )
        .reset_index()
    )


def within_similarity_bin_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare MI and naive uncertainty after stratifying by similarity."""

    _require_columns(
        predictions,
        {
            "strategy",
            "seed",
            "partition",
            "endpoint",
            "similarity_bin",
            "endpoint_bce",
            "is_error",
            "single_pass_uncertainty",
            "mc_dropout_mi",
        },
        "MC Dropout predictions",
    )
    frames: list[pd.DataFrame] = []
    group_columns = [
        "strategy",
        "seed",
        "partition",
        "endpoint",
        "similarity_bin",
    ]
    for keys, group in predictions.groupby(
        group_columns, sort=False, observed=True
    ):
        metrics = error_risk_metrics(
            group["endpoint_bce"].to_numpy(),
            group["is_error"].to_numpy(),
            {
                "single_pass_uncertainty": group[
                    "single_pass_uncertainty"
                ].to_numpy(),
                "mc_dropout_mi": group["mc_dropout_mi"].to_numpy(),
            },
        )
        for column, value in zip(group_columns, keys, strict=True):
            metrics[column] = value
        frames.append(metrics)
    columns = [
        *group_columns,
        "signal",
        "n_observed",
        "n_errors",
        "spearman_bce",
        "error_auroc",
        "error_auprc",
    ]
    return pd.concat(frames, ignore_index=True)[columns]


def summarise_within_bin_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise valid endpoint-seed cells for each similarity bin."""

    summary = (
        metrics.groupby(
            ["strategy", "partition", "similarity_bin", "signal"],
            sort=False,
            observed=True,
        )
        .agg(
            n_cells=("endpoint", "size"),
            total_observed=("n_observed", "sum"),
            median_observed=("n_observed", "median"),
            total_errors=("n_errors", "sum"),
            n_valid_spearman=("spearman_bce", "count"),
            spearman_bce_mean=("spearman_bce", "mean"),
            spearman_bce_std=("spearman_bce", "std"),
            n_valid_auroc=("error_auroc", "count"),
            error_auroc_mean=("error_auroc", "mean"),
            error_auroc_std=("error_auroc", "std"),
            n_valid_auprc=("error_auprc", "count"),
            error_auprc_mean=("error_auprc", "mean"),
            error_auprc_std=("error_auprc", "std"),
        )
        .reset_index()
    )
    summary["strategy"] = pd.Categorical(
        summary["strategy"], categories=tuple(SPLIT_COLUMNS), ordered=True
    )
    summary["partition"] = pd.Categorical(
        summary["partition"], categories=("validation", "test"), ordered=True
    )
    summary["similarity_bin"] = pd.Categorical(
        summary["similarity_bin"],
        categories=SIMILARITY_BIN_LABELS,
        ordered=True,
    )
    return summary.sort_values(
        ["strategy", "partition", "similarity_bin", "signal"]
    ).reset_index(drop=True)


def paired_within_bin_comparison(
    metrics: pd.DataFrame, *, minimum_observed: int
) -> pd.DataFrame:
    """Pair naive and MI scores within the same endpoint-seed-similarity cell."""

    if minimum_observed <= 0:
        raise ValueError("minimum_observed must be positive")
    supported = metrics.loc[metrics["n_observed"] >= minimum_observed]
    group_columns = [
        "strategy",
        "seed",
        "partition",
        "endpoint",
        "similarity_bin",
    ]
    value_columns = ["spearman_bce", "error_auroc", "error_auprc"]
    naive = supported.loc[
        supported["signal"].eq("single_pass_uncertainty"),
        [*group_columns, *value_columns],
    ]
    mutual_information = supported.loc[
        supported["signal"].eq("mc_dropout_mi"),
        [*group_columns, *value_columns],
    ]
    paired = naive.merge(
        mutual_information,
        on=group_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_naive", "_mi"),
    )
    for metric in value_columns:
        paired[f"{metric}_difference_naive_minus_mi"] = (
            paired[f"{metric}_naive"] - paired[f"{metric}_mi"]
        )
    return paired


def summarise_paired_within_bin_comparison(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise paired within-bin advantages without pooling predictions."""

    columns = ["strategy", "partition"]
    for short_name in ("spearman", "auroc", "auprc"):
        columns.extend(
            [
                f"n_pairs_{short_name}",
                f"mean_difference_naive_minus_mi_{short_name}",
                f"fraction_naive_better_{short_name}",
            ]
        )
    columns.extend(
        [
            "n_endpoint_clusters_auroc",
            "mean_difference_naive_minus_mi_auroc_ci_lower",
            "mean_difference_naive_minus_mi_auroc_ci_upper",
        ]
    )
    rows: list[dict[str, float | int | str]] = []
    for keys, group in paired.groupby(["strategy", "partition"], sort=False):
        row: dict[str, float | int | str] = {
            "strategy": keys[0],
            "partition": keys[1],
        }
        for short_name, metric in (
            ("spearman", "spearman_bce"),
            ("auroc", "error_auroc"),
            ("auprc", "error_auprc"),
        ):
            difference_column = f"{metric}_difference_naive_minus_mi"
            differences = group[difference_column].dropna()
            endpoint_differences = (
                group.loc[group[difference_column].notna()]
                .groupby("endpoint", sort=False)[difference_column]
                .mean()
            )
            row[f"n_pairs_{short_name}"] = len(differences)
            row[f"mean_difference_naive_minus_mi_{short_name}"] = (
                float(endpoint_differences.mean())
                if len(endpoint_differences)
                else float("nan")
            )
            row[f"fraction_naive_better_{short_name}"] = (
                float((differences > 0.0).mean())
                if len(differences)
                else float("nan")
            )
        auroc_difference_column = "error_auroc_difference_naive_minus_mi"
        auroc_rows = group.loc[group[auroc_difference_column].notna()]
        endpoint_auroc = (
            auroc_rows.groupby("endpoint", sort=False)[auroc_difference_column]
            .mean()
            .rename(auroc_difference_column)
            .reset_index()
        )
        row["n_endpoint_clusters_auroc"] = len(endpoint_auroc)
        ci_lower, ci_upper = cluster_bootstrap_mean_ci(
            endpoint_auroc,
            value_column=auroc_difference_column,
            cluster_column="endpoint",
            n_bootstrap=BOOTSTRAP_REPLICATES,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
            seed=BOOTSTRAP_SEED,
        )
        row["mean_difference_naive_minus_mi_auroc_ci_lower"] = ci_lower
        row["mean_difference_naive_minus_mi_auroc_ci_upper"] = ci_upper
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _plot_model_performance(performance: pd.DataFrame, path: Path) -> None:
    test = performance.loc[performance["partition"].eq("test")]
    strategies = _ordered_values(test["strategy"], tuple(SPLIT_COLUMNS))
    models = _ordered_values(test["model"], ("LR", "GCN"))
    positions = np.arange(len(strategies), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("auroc", "auprc"),
        ("Test AUROC", "Test AUPRC"),
        strict=True,
    ):
        for model_index, model in enumerate(models):
            rows = test.loc[test["model"].eq(model)].set_index("strategy")
            means = [rows.loc[strategy, f"{metric}_mean"] for strategy in strategies]
            standard_deviations = np.array(
                [rows.loc[strategy, f"{metric}_std"] for strategy in strategies],
                dtype=float,
            )
            offset = (model_index - (len(models) - 1) / 2) * width
            axis.bar(
                positions + offset,
                means,
                width,
                yerr=np.nan_to_num(standard_deviations, nan=0.0),
                capsize=3,
                color=MODEL_COLORS[model],
                label=model,
            )
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(positions, [STRATEGY_LABELS[value] for value in strategies])
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro score")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, frameon=False)
    _save_figure(fig, path)


def _plot_similarity_relationships(similarity_bins: pd.DataFrame, path: Path) -> None:
    _require_columns(
        similarity_bins,
        {
            "strategy",
            "partition",
            "similarity_bin",
            "mean_max_train_tanimoto",
            "mean_endpoint_bce",
            "mean_mc_dropout_mi",
        },
        "similarity-bin summary",
    )
    test = filter_supported_similarity_bins(
        similarity_bins.loc[similarity_bins["partition"].eq("test")],
        minimum_molecules=MIN_FIGURE_BIN_MOLECULES,
    )
    strategies = _ordered_values(test["strategy"], tuple(SPLIT_COLUMNS))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for strategy, color in zip(strategies, colors, strict=False):
        rows = test.loc[test["strategy"].eq(strategy)].sort_values(
            "mean_max_train_tanimoto"
        )
        axes[0].plot(
            rows["mean_max_train_tanimoto"],
            rows["mean_endpoint_bce"],
            marker="o",
            color=color,
            label=STRATEGY_LABELS[strategy],
        )
        axes[1].plot(
            rows["mean_max_train_tanimoto"],
            rows["mean_mc_dropout_mi"],
            marker="o",
            color=color,
            label=STRATEGY_LABELS[strategy],
        )
    axes[0].set_title("Prediction error by similarity")
    axes[0].set_ylabel("Mean endpoint BCE")
    axes[1].set_title("MC Dropout MI by similarity")
    axes[1].set_ylabel("Mean MI (nats)")
    for axis in axes:
        axis.set_xlabel("Mean maximum training-set Tanimoto")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, frameon=False)
    fig.suptitle(
        f"Bins with at least {MIN_FIGURE_BIN_MOLECULES} unique molecules",
        fontsize=10,
        y=1.02,
    )
    _save_figure(fig, path)


def _plot_risk_signal_comparison(risk_metrics: pd.DataFrame, path: Path) -> None:
    test = risk_metrics.loc[risk_metrics["partition"].eq("test")]
    summary = (
        test.groupby(["strategy", "signal"], sort=False)
        .agg(
            error_auroc_mean=("error_auroc", "mean"),
            error_auroc_std=("error_auroc", "std"),
            error_auprc_mean=("error_auprc", "mean"),
            error_auprc_std=("error_auprc", "std"),
        )
        .reset_index()
    )
    strategies = _ordered_values(summary["strategy"], tuple(SPLIT_COLUMNS))
    signals = _ordered_values(summary["signal"], tuple(SIGNAL_LABELS))
    positions = np.arange(len(strategies), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("error_auroc", "error_auprc"),
        ("Error-detection AUROC", "Error-detection AUPRC"),
        strict=True,
    ):
        for signal_index, signal in enumerate(signals):
            rows = summary.loc[summary["signal"].eq(signal)].set_index("strategy")
            means = [rows.loc[strategy, f"{metric}_mean"] for strategy in strategies]
            deviations = [
                rows.loc[strategy, f"{metric}_std"] for strategy in strategies
            ]
            offset = (signal_index - (len(signals) - 1) / 2) * width
            axis.bar(
                positions + offset,
                means,
                width,
                yerr=np.nan_to_num(deviations, nan=0.0),
                capsize=3,
                color=SIGNAL_COLORS[signal],
                label=SIGNAL_LABELS[signal],
            )
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(positions, [STRATEGY_LABELS[value] for value in strategies])
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean across endpoint-seed cells")
    axes[1].legend(frameon=False, fontsize=9)
    _save_figure(fig, path)


def _results_summary_markdown(
    model_performance: pd.DataFrame,
    distance_summary: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    within_bin_paired_summary: pd.DataFrame,
) -> str:
    test_models = model_performance.loc[model_performance["partition"].eq("test")]
    test_distance = distance_summary.loc[distance_summary["partition"].eq("test")]
    test_risk = (
        risk_metrics.loc[risk_metrics["partition"].eq("test")]
        .groupby(["strategy", "signal"], sort=False)
        .agg(
            error_auroc=("error_auroc", "mean"),
            error_auprc=("error_auprc", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Results summary",
        "",
        "The test partition is the primary reporting partition. Validation results are retained as a consistency check.",
        "",
        "## RQ1 — Split-level performance",
        "",
    ]
    for row in test_models.itertuples(index=False):
        lines.append(
            f"- {STRATEGY_LABELS.get(row.strategy, row.strategy)} {row.model}: "
            f"AUROC {row.auroc_mean:.3f}; AUPRC {row.auprc_mean:.3f}."
        )
    lines.extend(
        [
            "",
            "## RQ2 — Structural distance, error, and uncertainty",
            "",
        ]
    )
    for row in test_distance.itertuples(index=False):
        label = row.relationship.replace("structural_distance_vs_", "distance–")
        lines.append(
            f"- {STRATEGY_LABELS.get(row.strategy, row.strategy)} {label}: "
            f"mean Spearman ρ = {row.spearman_mean:.3f} "
            f"across {row.n_valid} valid endpoint-seed cells."
        )
    lines.extend(["", "## Ablation — Error-risk signals", ""])
    for row in test_risk.itertuples(index=False):
        lines.append(
            f"- {STRATEGY_LABELS.get(row.strategy, row.strategy)} "
            f"{SIGNAL_LABELS.get(row.signal, row.signal)}: error AUROC "
            f"{row.error_auroc:.3f}; error AUPRC {row.error_auprc:.3f}."
        )
    for row in within_bin_paired_summary.loc[
        within_bin_paired_summary["partition"].eq("test")
    ].itertuples(index=False):
        if np.isfinite(row.mean_difference_naive_minus_mi_auroc_ci_lower):
            ci_text = (
                f"95% endpoint-cluster bootstrap CI "
                f"{row.mean_difference_naive_minus_mi_auroc_ci_lower:.3f} to "
                f"{row.mean_difference_naive_minus_mi_auroc_ci_upper:.3f}; "
                f"{row.n_endpoint_clusters_auroc} endpoints"
            )
        else:
            ci_text = (
                "95% endpoint-cluster bootstrap CI not estimable from "
                f"{row.n_endpoint_clusters_auroc} endpoint"
            )
        lines.append(
            f"- Within similarity bins, naive uncertainty had higher error "
            f"AUROC than MI in {row.fraction_naive_better_auroc:.1%} of "
            f"{row.n_pairs_auroc} valid "
            f"{STRATEGY_LABELS.get(row.strategy, row.strategy)} paired cells "
            f"(mean AUROC difference {row.mean_difference_naive_minus_mi_auroc:.3f}; "
            f"{ci_text})."
        )
    valid_within_bin_pairs = int(
        within_bin_paired_summary.loc[
            within_bin_paired_summary["partition"].eq("test"), "n_pairs_auroc"
        ].sum()
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Similarity-bin comparisons are descriptive rather than exact molecule matching. Extreme bins may contain few molecules, so their means should not be interpreted alone.",
            f"The similarity figure displays only bins containing at least {MIN_FIGURE_BIN_MOLECULES} unique molecules; all bins remain in the tables.",
            f"Paired within-bin conclusions require at least {MIN_PAIRED_CELL_OBSERVED} observed labels per endpoint-seed-bin cell.",
            "Paired AUROC differences are averaged within each endpoint before the 12 endpoint means are macro-averaged.",
            f"The 95% endpoint-cluster bootstrap CIs use {BOOTSTRAP_REPLICATES:,} percentile-bootstrap resamples of whole endpoints, preserving dependence among seeds and similarity bins from the same endpoint.",
            f"Within-bin paired error AUROC was defined for {valid_within_bin_pairs} endpoint-seed-bin cells in the test analyses.",
            "Single-pass uncertainty is mathematically derived from the same probability used to calculate BCE; its BCE correlation is therefore not independent evidence. Error-detection AUROC/AUPRC provide the more direct ablation comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def _ordered_values(values: pd.Series, preferred: tuple[str, ...]) -> list[str]:
    observed = set(values.dropna().astype(str))
    ordered = [value for value in preferred if value in observed]
    return [*ordered, *sorted(observed.difference(ordered))]


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(first, second).statistic)


def _require_columns(
    frame: pd.DataFrame, required: set[str], description: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{description} is missing columns: {', '.join(missing)}")
