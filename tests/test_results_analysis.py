from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tox21_risk.results_analysis import (
    cluster_bootstrap_mean_ci,
    combine_model_performance,
    distance_relationship_metrics,
    filter_supported_similarity_bins,
    paired_within_bin_comparison,
    summarise_distance_relationships,
    summarise_paired_within_bin_comparison,
    summarise_within_bin_metrics,
    within_similarity_bin_metrics,
    run_results_analysis,
)


def test_cluster_bootstrap_ci_resamples_endpoints_and_is_reproducible() -> None:
    paired_differences = pd.DataFrame(
        {
            "endpoint": ["task_a", "task_a", "task_b", "task_b"],
            "difference": [0.10, 0.10, 0.30, 0.30],
        }
    )

    first = cluster_bootstrap_mean_ci(
        paired_differences,
        value_column="difference",
        cluster_column="endpoint",
        n_bootstrap=10_000,
        confidence_level=0.95,
        seed=2026,
    )
    second = cluster_bootstrap_mean_ci(
        paired_differences,
        value_column="difference",
        cluster_column="endpoint",
        n_bootstrap=10_000,
        confidence_level=0.95,
        seed=2026,
    )

    assert first == pytest.approx((0.10, 0.30))
    assert second == first


def test_cluster_bootstrap_ci_requires_two_endpoint_clusters() -> None:
    one_endpoint = pd.DataFrame(
        {"endpoint": ["task_a", "task_a"], "difference": [0.10, 0.20]}
    )

    lower, upper = cluster_bootstrap_mean_ci(
        one_endpoint,
        value_column="difference",
        cluster_column="endpoint",
        n_bootstrap=100,
        confidence_level=0.95,
        seed=2026,
    )

    assert np.isnan(lower)
    assert np.isnan(upper)


def test_paired_summary_reports_endpoint_cluster_bootstrap_ci() -> None:
    paired = pd.DataFrame(
        {
            "strategy": ["random"] * 4,
            "partition": ["test"] * 4,
            "endpoint": ["task_a", "task_a", "task_b", "task_b"],
            "spearman_bce_difference_naive_minus_mi": [0.1, 0.1, 0.3, 0.3],
            "error_auroc_difference_naive_minus_mi": [0.1, 0.1, 0.3, 0.3],
            "error_auprc_difference_naive_minus_mi": [0.1, 0.1, 0.3, 0.3],
        }
    )

    summary = summarise_paired_within_bin_comparison(paired)

    assert summary.loc[0, "n_endpoint_clusters_auroc"] == 2
    assert summary.loc[
        0, "mean_difference_naive_minus_mi_auroc_ci_lower"
    ] == pytest.approx(0.10)
    assert summary.loc[
        0, "mean_difference_naive_minus_mi_auroc_ci_upper"
    ] == pytest.approx(0.30)


def test_paired_summary_gives_each_endpoint_equal_weight() -> None:
    paired = pd.DataFrame(
        {
            "strategy": ["scaffold"] * 4,
            "partition": ["test"] * 4,
            "endpoint": ["task_a", "task_b", "task_b", "task_b"],
            "spearman_bce_difference_naive_minus_mi": [0.0, 1.0, 1.0, 1.0],
            "error_auroc_difference_naive_minus_mi": [0.0, 1.0, 1.0, 1.0],
            "error_auprc_difference_naive_minus_mi": [0.0, 1.0, 1.0, 1.0],
        }
    )

    summary = summarise_paired_within_bin_comparison(paired)

    assert summary.loc[0, "n_pairs_auroc"] == 4
    assert summary.loc[0, "n_endpoint_clusters_auroc"] == 2
    assert summary.loc[
        0, "mean_difference_naive_minus_mi_auroc"
    ] == pytest.approx(0.5)
    assert summary.loc[
        0, "mean_difference_naive_minus_mi_auroc_ci_lower"
    ] == pytest.approx(0.0)
    assert summary.loc[
        0, "mean_difference_naive_minus_mi_auroc_ci_upper"
    ] == pytest.approx(1.0)


def test_combine_model_performance_aligns_lr_and_gcn_columns() -> None:
    lr = pd.DataFrame(
        {
            "strategy": ["random"],
            "partition": ["test"],
            "n_endpoints": [12],
            "macro_auroc": [0.75],
            "macro_auprc": [0.30],
            "macro_brier": [0.12],
        }
    )
    gcn = pd.DataFrame(
        {
            "strategy": ["random"],
            "partition": ["test"],
            "n_seeds": [3],
            "macro_auroc_mean": [0.80],
            "macro_auroc_std": [0.02],
            "macro_auprc_mean": [0.35],
            "macro_auprc_std": [0.03],
            "macro_brier_mean": [0.08],
            "macro_brier_std": [0.01],
        }
    )

    result = combine_model_performance(lr, gcn)

    assert result[["model", "n_runs"]].to_dict("records") == [
        {"model": "LR", "n_runs": 1},
        {"model": "GCN", "n_runs": 3},
    ]
    assert result.loc[result["model"].eq("LR"), "auroc_mean"].item() == 0.75
    assert np.isnan(result.loc[result["model"].eq("LR"), "auroc_std"].item())
    assert result.loc[result["model"].eq("GCN"), "auroc_std"].item() == 0.02


def test_filter_supported_similarity_bins_keeps_threshold_and_above() -> None:
    frame = pd.DataFrame(
        {
            "similarity_bin": ["0.0-0.2", "0.2-0.4", "0.4-0.6"],
            "n_unique_molecules": [19, 20, 21],
        }
    )

    result = filter_supported_similarity_bins(frame, minimum_molecules=20)

    assert result["similarity_bin"].tolist() == ["0.2-0.4", "0.4-0.6"]


def test_distance_relationship_metrics_measure_bce_and_mi_separately() -> None:
    predictions = _prediction_frame()

    result = distance_relationship_metrics(predictions).set_index("relationship")

    assert result.loc["structural_distance_vs_bce", "n"] == 4
    assert result.loc["structural_distance_vs_bce", "spearman"] == pytest.approx(1.0)
    assert result.loc["structural_distance_vs_mi", "n"] == 5
    assert result.loc["structural_distance_vs_mi", "spearman"] == pytest.approx(1.0)

    summary = summarise_distance_relationships(result.reset_index())
    assert len(summary) == 2
    assert set(summary["n_valid"]) == {1}


def test_within_similarity_bin_metrics_compare_mi_with_naive_uncertainty() -> None:
    predictions = _prediction_frame().iloc[:4].copy()
    predictions["similarity_bin"] = "0.2-0.4"

    result = within_similarity_bin_metrics(predictions).set_index("signal")

    assert set(result.index) == {"single_pass_uncertainty", "mc_dropout_mi"}
    assert result.loc["mc_dropout_mi", "spearman_bce"] == pytest.approx(1.0)
    assert result.loc["mc_dropout_mi", "error_auroc"] == pytest.approx(1.0)
    assert result.loc["single_pass_uncertainty", "error_auroc"] == pytest.approx(0.0)

    summary = summarise_within_bin_metrics(result.reset_index())
    assert len(summary) == 2
    assert set(summary["n_valid_auroc"]) == {1}
    assert set(summary["total_observed"]) == {4}
    assert set(summary["total_errors"]) == {2}

    paired = paired_within_bin_comparison(
        result.reset_index(), minimum_observed=1
    )
    assert paired.loc[0, "error_auroc_difference_naive_minus_mi"] == pytest.approx(
        -1.0
    )
    paired_summary = summarise_paired_within_bin_comparison(paired)
    assert paired_summary.loc[0, "n_pairs_auroc"] == 1
    assert paired_summary.loc[0, "fraction_naive_better_auroc"] == 0.0
    assert paired_within_bin_comparison(
        result.reset_index(), minimum_observed=5
    ).empty


def test_run_results_analysis_writes_tables_figures_and_summary(tmp_path) -> None:
    lr_path = tmp_path / "lr.csv"
    pd.DataFrame(
        {
            "strategy": ["random"],
            "partition": ["test"],
            "n_endpoints": [1],
            "macro_auroc": [0.75],
            "macro_auprc": [0.30],
            "macro_brier": [0.12],
        }
    ).to_csv(lr_path, index=False)
    gcn_path = tmp_path / "gcn.csv"
    pd.DataFrame(
        {
            "strategy": ["random"],
            "partition": ["test"],
            "n_seeds": [3],
            "macro_bce_mean": [0.2],
            "macro_bce_std": [0.01],
            "macro_auroc_mean": [0.80],
            "macro_auroc_std": [0.02],
            "macro_auprc_mean": [0.35],
            "macro_auprc_std": [0.03],
            "macro_brier_mean": [0.08],
            "macro_brier_std": [0.01],
        }
    ).to_csv(gcn_path, index=False)
    predictions = _prediction_frame()
    predictions["similarity_bin"] = "0.2-0.4"
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    risk_metrics_path = tmp_path / "risk_metrics.csv"
    pd.DataFrame(
        {
            "strategy": ["random"] * 3,
            "seed": [2026] * 3,
            "partition": ["test"] * 3,
            "endpoint": ["task_a"] * 3,
            "signal": [
                "structural_distance",
                "single_pass_uncertainty",
                "mc_dropout_mi",
            ],
            "n_observed": [4] * 3,
            "n_errors": [2] * 3,
            "spearman_bce": [1.0, -1.0, 1.0],
            "error_auroc": [1.0, 0.0, 1.0],
            "error_auprc": [1.0, 0.42, 1.0],
        }
    ).to_csv(risk_metrics_path, index=False)
    similarity_bins_path = tmp_path / "similarity_bins.csv"
    pd.DataFrame(
        {
            "strategy": ["random"],
            "partition": ["test"],
            "similarity_bin": ["0.2-0.4"],
            "n_endpoint_rows": [4],
            "n_unique_molecules": [4],
            "mean_max_train_tanimoto": [0.3],
            "mean_endpoint_bce": [0.5],
            "error_rate": [0.5],
            "mean_single_pass_uncertainty": [0.5],
            "mean_mc_dropout_mi": [0.05],
        }
    ).to_csv(similarity_bins_path, index=False)
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"
    summary_path = tmp_path / "results_summary.md"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metadata = run_results_analysis(
            lr_summary_path=lr_path,
            gcn_summary_path=gcn_path,
            predictions_path=predictions_path,
            risk_endpoint_metrics_path=risk_metrics_path,
            similarity_bins_path=similarity_bins_path,
            table_dir=table_dir,
            figure_dir=figure_dir,
            summary_path=summary_path,
        )

    expected_tables = {
        "model_performance_comparison.csv",
        "distance_relationship_endpoint_metrics.csv",
        "distance_relationship_summary.csv",
        "within_similarity_bin_endpoint_metrics.csv",
        "within_similarity_bin_summary.csv",
        "within_similarity_bin_paired_summary.csv",
        "results_analysis_metadata.json",
    }
    assert expected_tables == {path.name for path in table_dir.iterdir()}
    expected_figures = {
        "model_performance_by_split.png",
        "similarity_error_uncertainty.png",
        "error_risk_signal_comparison.png",
    }
    assert expected_figures == {path.name for path in figure_dir.iterdir()}
    for path in figure_dir.iterdir():
        with Image.open(path) as image:
            assert image.width >= 800
            assert image.height >= 400
    assert "RQ1" in summary_path.read_text(encoding="utf-8")
    assert "95% endpoint-cluster bootstrap CI" in summary_path.read_text(
        encoding="utf-8"
    )
    paired_summary = pd.read_csv(
        table_dir / "within_similarity_bin_paired_summary.csv"
    )
    assert {
        "n_endpoint_clusters_auroc",
        "mean_difference_naive_minus_mi_auroc_ci_lower",
        "mean_difference_naive_minus_mi_auroc_ci_upper",
    }.issubset(paired_summary.columns)
    assert not any("No artists with labels" in str(item.message) for item in caught)
    assert metadata["n_prediction_rows"] == 5
    assert metadata["primary_partition"] == "test"
    assert metadata["paired_difference_ci"] == {
        "method": "endpoint_cluster_percentile_bootstrap",
        "confidence_level": 0.95,
        "n_resamples": 10_000,
        "seed": 2026,
    }
    assert metadata["paired_difference_aggregation"] == (
        "mean_within_endpoint_then_macro_across_endpoints"
    )


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mol_id": [f"mol_{index}" for index in range(5)],
            "strategy": ["random"] * 5,
            "seed": [2026] * 5,
            "partition": ["test"] * 5,
            "endpoint": ["task_a"] * 5,
            "label": [0.0, 0.0, 1.0, 1.0, np.nan],
            "endpoint_bce": [0.1, 0.2, 0.8, 0.9, np.nan],
            "is_error": [0.0, 0.0, 1.0, 1.0, np.nan],
            "structural_distance": [0.1, 0.2, 0.7, 0.9, 1.0],
            "single_pass_uncertainty": [0.9, 0.8, 0.2, 0.1, 0.0],
            "mc_dropout_mi": [0.01, 0.02, 0.08, 0.09, 0.10],
        }
    )
