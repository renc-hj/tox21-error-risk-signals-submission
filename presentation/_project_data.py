"""Small, reproducible inputs for the research-focused Lectrace presentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from tox21_risk.fingerprints import morgan_fingerprint_matrix
from tox21_risk.gcn_model import Tox21GCN
from tox21_risk.graph_data import ATOM_FEATURE_DIM, smiles_to_graph
from tox21_risk.results_analysis import (
    paired_within_bin_comparison,
    summarise_paired_within_bin_comparison,
)
from tox21_risk.splitting import (
    SplitConfig,
    bemis_murcko_scaffolds,
    butina_cluster_ids,
    group_assignments,
    random_assignments,
)
from tox21_risk.uncertainty import (
    binary_classification_error,
    binary_cross_entropy_per_label,
    mc_dropout_mutual_information,
    single_pass_uncertainty,
)


ROOT = Path(__file__).resolve().parents[1]

DEMO_SMILES = (
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "Oc1ccccc1",
    "C1CCCCC1",
    "CC1CCCCC1",
    "c1ccncc1",
    "Cc1ccncc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "O=C(O)c1ccccc1O",
    "CCO",
    "CCCO",
)


def build_split_demo() -> SimpleNamespace:
    """Apply the submitted split functions to a compact molecule set."""

    config = SplitConfig()
    fingerprints = morgan_fingerprint_matrix(DEMO_SMILES)
    scaffolds = bemis_murcko_scaffolds(DEMO_SMILES)
    cluster_ids = butina_cluster_ids(fingerprints, config)
    return SimpleNamespace(
        config=config,
        smiles=DEMO_SMILES,
        fingerprints=fingerprints,
        scaffolds=scaffolds,
        cluster_ids=cluster_ids,
        random_assignments=random_assignments(len(DEMO_SMILES), config),
        scaffold_assignments=group_assignments(scaffolds, config),
        cluster_assignments=group_assignments(cluster_ids, config),
    )


def build_split_presentation_state() -> dict[str, object]:
    """Return only the split values that are useful on the presentation screen."""

    state = build_split_demo()
    return {
        "molecule_count": len(state.smiles),
        "fingerprint_shape": tuple(state.fingerprints.shape),
        "scaffold_groups": state.scaffolds,
        "similarity_cluster_ids": state.cluster_ids.tolist(),
        "random_partition": state.random_assignments.tolist(),
        "scaffold_partition": state.scaffold_assignments.tolist(),
        "similarity_cluster_partition": state.cluster_assignments.tolist(),
    }


def build_uncertainty_demo() -> SimpleNamespace:
    """Apply the submitted uncertainty functions to a hand-checkable example."""

    labels = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=float
    )
    single_probabilities = np.array(
        [[0.90, 0.20], [0.55, 0.52], [0.48, 0.80], [0.10, 0.45]], dtype=float
    )
    offsets = np.linspace(-0.08, 0.08, 50)[:, None, None]
    direction = np.array(
        [[1.0, -0.4], [0.8, 1.0], [-1.0, 0.5], [-0.5, 1.0]], dtype=float
    )[None, :, :]
    mc_probabilities = np.clip(
        single_probabilities[None, :, :] + offsets * direction,
        1e-4,
        1.0 - 1e-4,
    )
    return SimpleNamespace(
        labels=labels,
        single_probabilities=single_probabilities,
        mc_probabilities=mc_probabilities,
        single_pass_uncertainty=single_pass_uncertainty(single_probabilities),
        mutual_information=mc_dropout_mutual_information(mc_probabilities),
        is_error=binary_classification_error(labels, single_probabilities),
        endpoint_bce=binary_cross_entropy_per_label(labels, single_probabilities),
    )


def build_uncertainty_presentation_state() -> dict[str, object]:
    """Return compact probability-to-risk values for the variable panel."""

    state = build_uncertainty_demo()
    return {
        "labels": state.labels.tolist(),
        "single_pass_probabilities": state.single_probabilities.tolist(),
        "mc_dropout_passes_shape": tuple(state.mc_probabilities.shape),
        "single_pass_uncertainty": state.single_pass_uncertainty.round(3).tolist(),
        "mc_dropout_mutual_information": state.mutual_information.round(5).tolist(),
        "prediction_error": state.is_error.tolist(),
        "binary_cross_entropy": state.endpoint_bce.round(3).tolist(),
        "illustrative_endpoint": {
            "true_label": int(state.labels[2, 0]),
            "deterministic_probability": float(state.single_probabilities[2, 0]),
            "predicted_label_at_0.5": int(state.single_probabilities[2, 0] >= 0.5),
            "single_pass_uncertainty": round(
                float(state.single_pass_uncertainty[2, 0]), 3
            ),
            "mc_dropout_mutual_information": round(
                float(state.mutual_information[2, 0]), 5
            ),
        },
    }


def build_gcn_forward_demo() -> dict[str, object]:
    """Expose the submitted GCN's shape changes on one deterministic example."""

    torch.manual_seed(6329)
    graph = smiles_to_graph("demo_molecule", "CCO", [0.0] * 12)
    batch = Batch.from_data_list([graph])
    model = Tox21GCN(in_channels=ATOM_FEATURE_DIM, num_tasks=12)
    model.eval()
    layer_output_shapes: list[tuple[int, ...]] = []
    pooled_shape: tuple[int, ...] | None = None

    def capture_layer_shape(_module, _inputs, output) -> None:
        layer_output_shapes.append(tuple(output.shape))

    def capture_head_input(_module, inputs, _output) -> None:
        nonlocal pooled_shape
        pooled_shape = tuple(inputs[0].shape)

    hooks = [layer.register_forward_hook(capture_layer_shape) for layer in model.convolutions]
    hooks.append(model.output.register_forward_hook(capture_head_input))
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch)
        probabilities = torch.sigmoid(logits)
    for hook in hooks:
        hook.remove()
    return {
        "molecule": "CCO",
        "atom_feature_shape": tuple(batch.x.shape),
        "directed_edge_shape": tuple(batch.edge_index.shape),
        "gcn_layers": len(model.convolutions),
        "layer_output_shapes": layer_output_shapes,
        "pooled_molecule_shape": pooled_shape,
        "endpoint_probability_shape": tuple(probabilities.shape),
        "endpoint_probabilities": probabilities.squeeze(0).round(decimals=3).tolist(),
    }


def load_headline_results() -> dict[str, dict[str, float]]:
    """Load test error-detection AUROC values from the tracked result table."""

    table = pd.read_csv(ROOT / "reports" / "tables" / "risk_signal_summary.csv")
    selected = table.loc[
        (table["partition"] == "test")
        & table["signal"].isin(
            ["single_pass_uncertainty", "mc_dropout_mi"]
        )
    ]
    results: dict[str, dict[str, float]] = {}
    for strategy, rows in selected.groupby("strategy", sort=False):
        by_signal = rows.set_index("signal")["error_auroc_mean"]
        results[str(strategy)] = {
            "single_pass": round(
                float(by_signal["single_pass_uncertainty"]), 3
            ),
            "mc_dropout_mi": round(float(by_signal["mc_dropout_mi"]), 3),
        }
    return results


def load_model_performance() -> list[dict[str, float | str]]:
    """Return GCN test AUROC and AUPRC values for the three fixed splits."""

    table = pd.read_csv(
        ROOT / "reports" / "tables" / "model_performance_comparison.csv"
    )
    rows = table.loc[(table["partition"] == "test") & (table["model"] == "GCN")]
    return [
        {
            "split": str(row.strategy).replace("_", "-"),
            "AUROC": round(float(row.auroc_mean), 3),
            "AUPRC": round(float(row.auprc_mean), 3),
        }
        for row in rows.itertuples(index=False)
    ]


def load_within_similarity_summary() -> list[dict[str, float | str]]:
    """Recompute the report's paired test comparison within similarity bins."""

    table = pd.read_csv(
        ROOT / "reports" / "tables" / "within_similarity_bin_endpoint_metrics.csv"
    )
    paired = paired_within_bin_comparison(table, minimum_observed=20)
    summary = summarise_paired_within_bin_comparison(paired)
    test_rows = summary.loc[summary["partition"].eq("test")]
    return [
        {
            "split": str(row.strategy).replace("_", "-"),
            "single_pass_better_fraction": round(
                float(row.fraction_naive_better_auroc), 3
            ),
            "mean_auroc_difference": round(
                float(row.mean_difference_naive_minus_mi_auroc), 3
            ),
            "95% CI": (
                round(float(row.mean_difference_naive_minus_mi_auroc_ci_lower), 3),
                round(float(row.mean_difference_naive_minus_mi_auroc_ci_upper), 3),
            ),
        }
        for row in test_rows.itertuples(index=False)
    ]
