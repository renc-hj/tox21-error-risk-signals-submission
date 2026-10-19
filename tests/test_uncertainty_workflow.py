from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

from tox21_risk.fingerprints import morgan_fingerprint_matrix
from tox21_risk.gcn_model import GCNConfig, Tox21GCN
from tox21_risk.graph_data import ATOM_FEATURE_DIM, build_graphs, save_graph_cache
from tox21_risk.split_workflow import SPLIT_COLUMNS
from tox21_risk.uncertainty_workflow import (
    SIMILARITY_BIN_LABELS,
    _similarity_bin_summary,
    run_uncertainty_workflow,
)


TASKS = ("task_a", "task_b")


def test_similarity_bin_summary_uses_increasing_similarity_order() -> None:
    predictions = pd.DataFrame(
        {
            "strategy": ["random", "random"],
            "partition": ["test", "test"],
            "similarity_bin": pd.Categorical(
                ["0.8-1.0", "0.0-0.2"],
                categories=SIMILARITY_BIN_LABELS,
                ordered=True,
            ),
            "label": [0.0, 1.0],
            "endpoint_bce": [0.1, 0.2],
            "mol_id": ["high", "low"],
            "max_train_tanimoto": [0.9, 0.1],
            "is_error": [0.0, 0.0],
            "single_pass_uncertainty": [0.1, 0.2],
            "mc_dropout_mi": [0.01, 0.02],
        }
    )

    result = _similarity_bin_summary(predictions)

    assert result["similarity_bin"].astype(str).tolist() == [
        "0.0-0.2",
        "0.8-1.0",
    ]


def test_uncertainty_workflow_writes_complete_error_risk_contract(tmp_path) -> None:
    molecules = pd.DataFrame(
        {
            "mol_id": [f"mol_{index}" for index in range(6)],
            "standardized_smiles": ["CC", "CCC", "CO", "CN", "CF", "CCl"],
            "task_a": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "task_b": [1.0, 0.0, np.nan, 0.0, 1.0, 0.0],
        }
    )
    molecule_path = tmp_path / "molecules.csv"
    molecules.to_csv(molecule_path, index=False)
    fingerprint_path = tmp_path / "fingerprints.npy"
    np.save(
        fingerprint_path,
        morgan_fingerprint_matrix(molecules["standardized_smiles"].tolist()),
    )
    graph_path = tmp_path / "graphs.pt"
    save_graph_cache(
        build_graphs(molecules, TASKS), graph_path, molecule_path, TASKS
    )
    assignments = pd.DataFrame(
        {
            "mol_id": molecules["mol_id"],
            **{
                column: [
                    "train",
                    "train",
                    "validation",
                    "validation",
                    "test",
                    "test",
                ]
                for column in SPLIT_COLUMNS.values()
            },
        }
    )
    assignment_path = tmp_path / "assignments.csv"
    assignments.to_csv(assignment_path, index=False)
    config = replace(GCNConfig(), hidden_channels=8, batch_size=2)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_config = asdict(config)
    checkpoint_config["seeds"] = list(config.seeds)
    for strategy in SPLIT_COLUMNS:
        for seed in config.seeds:
            torch.manual_seed(seed)
            model = Tox21GCN(
                ATOM_FEATURE_DIM,
                len(TASKS),
                hidden_channels=config.hidden_channels,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
            path = checkpoint_dir / strategy / f"seed_{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": checkpoint_config,
                    "seed": seed,
                    "split_strategy": strategy,
                    "input_width": ATOM_FEATURE_DIM,
                    "endpoint_order": list(TASKS),
                    "best_epoch": 1,
                    "best_validation_bce": 0.5,
                },
                path,
            )

    predictions_path = tmp_path / "mc_predictions.csv"
    endpoint_metrics_path = tmp_path / "endpoint_metrics.csv"
    summary_path = tmp_path / "summary.csv"
    similarity_bins_path = tmp_path / "similarity_bins.csv"
    metadata_path = tmp_path / "metadata.json"

    metadata = run_uncertainty_workflow(
        molecule_path=molecule_path,
        fingerprint_path=fingerprint_path,
        graph_cache_path=graph_path,
        assignment_path=assignment_path,
        checkpoint_dir=checkpoint_dir,
        predictions_path=predictions_path,
        endpoint_metrics_path=endpoint_metrics_path,
        summary_path=summary_path,
        similarity_bins_path=similarity_bins_path,
        metadata_path=metadata_path,
        tasks=TASKS,
        config=config,
        n_passes=3,
    )

    predictions = pd.read_csv(predictions_path)
    endpoint_metrics = pd.read_csv(endpoint_metrics_path)
    summary = pd.read_csv(summary_path)
    similarity_bins = pd.read_csv(similarity_bins_path)
    assert len(predictions) == 72
    assert set(predictions.columns) == {
        "mol_id",
        "strategy",
        "seed",
        "partition",
        "endpoint",
        "label",
        "single_probability",
        "mc_mean_probability",
        "endpoint_bce",
        "is_error",
        "max_train_tanimoto",
        "structural_distance",
        "single_pass_uncertainty",
        "mc_dropout_mi",
        "similarity_bin",
    }
    assert predictions["max_train_tanimoto"].notna().all()
    assert predictions.loc[predictions["label"].isna(), "endpoint_bce"].isna().all()
    assert (predictions["mc_dropout_mi"] >= 0.0).all()
    assert len(endpoint_metrics) == 108
    assert set(endpoint_metrics["signal"]) == {
        "structural_distance",
        "single_pass_uncertainty",
        "mc_dropout_mi",
    }
    assert len(summary) == 18
    assert not similarity_bins.empty
    bin_order = {
        label: index
        for index, label in enumerate(
            ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        )
    }
    for _, group in similarity_bins.groupby(["strategy", "partition"], sort=False):
        positions = [bin_order[value] for value in group["similarity_bin"]]
        assert positions == sorted(positions)
    assert metadata["n_passes"] == 3
    assert metadata["n_fits"] == 9
    assert metadata["n_prediction_rows"] == 72
    assert metadata["mutual_information_unit"] == "nats"
