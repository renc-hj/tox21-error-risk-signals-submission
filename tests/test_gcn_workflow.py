from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tox21_risk.gcn_model import GCNConfig, Tox21GCN
from tox21_risk.gcn_training import FitResult
from tox21_risk.gcn_workflow import run_gcn_workflow, validate_gcn_artifacts
from tox21_risk.graph_data import (
    ATOM_FEATURE_DIM,
    build_graphs,
    load_graph_cache,
    save_graph_cache,
)


TASKS = tuple(f"task_{index}" for index in range(12))
SPLIT_COLUMNS = (
    "random_split",
    "scaffold_split",
    "similarity_cluster_split",
)


@pytest.fixture
def aligned_artifacts(tmp_path):
    labels = np.tile([[0.0] * 12, [1.0] * 12], (3, 1))
    molecules = pd.DataFrame(
        {
            "mol_id": [f"mol_{index}" for index in range(6)],
            "standardized_smiles": ["CC", "CCC", "CO", "CN", "CF", "CCl"],
            **{task: labels[:, index] for index, task in enumerate(TASKS)},
        }
    )
    molecule_path = tmp_path / "molecules.csv"
    molecules.to_csv(molecule_path, index=False)
    graphs = build_graphs(molecules, TASKS)
    cache_path = tmp_path / "graphs.pt"
    save_graph_cache(graphs, cache_path, molecule_path, TASKS)
    assignments = pd.DataFrame(
        {
            "mol_id": molecules["mol_id"],
            **{
                column: ["train", "train", "validation", "validation", "test", "test"]
                for column in SPLIT_COLUMNS
            },
        }
    )
    return molecules, assignments, load_graph_cache(cache_path), molecule_path, cache_path


@pytest.mark.parametrize("fault", ["reordered", "missing", "duplicate"])
def test_validate_gcn_artifacts_rejects_mol_id_misalignment(
    aligned_artifacts, fault
) -> None:
    molecules, assignments, cache, molecule_path, _ = aligned_artifacts
    if fault == "reordered":
        assignments = assignments.iloc[::-1].reset_index(drop=True)
    elif fault == "missing":
        assignments = assignments.iloc[:-1].copy()
    else:
        assignments.loc[1, "mol_id"] = assignments.loc[0, "mol_id"]

    with pytest.raises(ValueError, match="mol_id"):
        validate_gcn_artifacts(molecules, assignments, cache, TASKS, molecule_path)


def test_validate_gcn_artifacts_rejects_stale_source_hash(
    aligned_artifacts,
) -> None:
    molecules, assignments, cache, molecule_path, _ = aligned_artifacts
    cache["molecule_file_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        validate_gcn_artifacts(molecules, assignments, cache, TASKS, molecule_path)


def test_validate_gcn_artifacts_rejects_graph_smiles_mismatch(
    aligned_artifacts,
) -> None:
    molecules, assignments, cache, molecule_path, _ = aligned_artifacts
    cache["graphs"][0].standardized_smiles = "N"

    with pytest.raises(ValueError, match="standardized SMILES"):
        validate_gcn_artifacts(molecules, assignments, cache, TASKS, molecule_path)


def test_validate_gcn_artifacts_rejects_wrong_schema_or_endpoint_order(
    aligned_artifacts,
) -> None:
    molecules, assignments, cache, molecule_path, _ = aligned_artifacts
    cache["feature_schema_version"] = 999
    with pytest.raises(ValueError, match="schema version"):
        validate_gcn_artifacts(molecules, assignments, cache, TASKS, molecule_path)

    cache["feature_schema_version"] = 1
    cache["endpoint_order"] = list(reversed(TASKS))
    with pytest.raises(ValueError, match="endpoint order"):
        validate_gcn_artifacts(molecules, assignments, cache, TASKS, molecule_path)


def test_validate_gcn_artifacts_rejects_invalid_split_feature_or_label(
    aligned_artifacts,
) -> None:
    molecules, assignments, cache, molecule_path, _ = aligned_artifacts
    bad_assignments = assignments.copy()
    bad_assignments.loc[0, "random_split"] = "holdout"
    with pytest.raises(ValueError, match="random.*partition"):
        validate_gcn_artifacts(
            molecules, bad_assignments, cache, TASKS, molecule_path
        )

    original_x = cache["graphs"][0].x
    cache["graphs"][0].x = original_x[:, :-1]
    with pytest.raises(ValueError, match="feature width"):
        validate_gcn_artifacts(molecules, assignments, cache, TASKS, molecule_path)
    cache["graphs"][0].x = original_x

    bad_molecules = molecules.copy()
    bad_molecules.loc[0, TASKS[0]] = 2.0
    with pytest.raises(ValueError, match="binary or missing"):
        validate_gcn_artifacts(
            bad_molecules, assignments, cache, TASKS, molecule_path
        )


def _tiny_trainer(
    train_graphs,
    validation_graphs,
    tasks,
    config,
    seed,
    split_strategy,
    checkpoint_path,
):
    torch.manual_seed(seed)
    model = Tox21GCN(
        ATOM_FEATURE_DIM,
        len(tasks),
        hidden_channels=8,
        num_layers=3,
        dropout=0.2,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "split_strategy": split_strategy}, checkpoint_path)
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [0.8, 0.7],
            "validation_macro_bce": [0.75, 0.70],
            "validation_macro_auroc": [0.5, 0.6],
        }
    )
    return FitResult(
        model=model,
        history=history,
        best_epoch=2,
        best_validation_bce=0.70,
    )


def test_run_gcn_workflow_writes_complete_nine_run_contract(
    aligned_artifacts, tmp_path
) -> None:
    molecules, assignments, _, molecule_path, cache_path = aligned_artifacts
    assignment_path = tmp_path / "assignments.csv"
    assignments.to_csv(assignment_path, index=False)
    config = replace(
        GCNConfig(),
        hidden_channels=8,
        batch_size=2,
        max_epochs=2,
        patience=1,
    )
    tables = tmp_path / "tables"
    predictions_path = tmp_path / "predictions.csv"
    history_path = tmp_path / "history.csv"
    checkpoint_dir = tmp_path / "checkpoints"

    metadata = run_gcn_workflow(
        molecule_path=molecule_path,
        graph_cache_path=cache_path,
        assignment_path=assignment_path,
        endpoint_metrics_path=tables / "endpoint.csv",
        split_summary_path=tables / "split.csv",
        training_summary_path=tables / "training.csv",
        metadata_path=tables / "metadata.json",
        predictions_path=predictions_path,
        history_path=history_path,
        checkpoint_dir=checkpoint_dir,
        tasks=TASKS,
        config=config,
        fit_fn=_tiny_trainer,
    )

    endpoint_metrics = pd.read_csv(tables / "endpoint.csv")
    split_summary = pd.read_csv(tables / "split.csv")
    training_summary = pd.read_csv(tables / "training.csv")
    predictions = pd.read_csv(predictions_path)
    history = pd.read_csv(history_path)
    assert len(endpoint_metrics) == 216
    assert len(split_summary) == 6
    assert set(split_summary["n_seeds"]) == {3}
    assert len(training_summary) == 9
    assert len(predictions) == 432
    assert set(predictions.columns) == {
        "mol_id",
        "strategy",
        "seed",
        "partition",
        "endpoint",
        "label",
        "probability",
    }
    assert len(history) == 18
    assert set(history["seed"]) == {2026, 2027, 2028}
    assert len(list(checkpoint_dir.glob("*/*.pt"))) == 9
    assert metadata["n_fits"] == 9
    assert metadata["selection_metric"] == "validation_macro_endpoint_bce"
    assert len(metadata["molecule_file_sha256"]) == 64
    assert len(metadata["graph_cache_file_sha256"]) == 64
    assert len(metadata["assignment_file_sha256"]) == 64
