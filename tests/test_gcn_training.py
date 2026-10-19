from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from tox21_risk.gcn_model import GCNConfig, Tox21GCN
from tox21_risk.gcn_training import (
    evaluate_model,
    fit_model,
    masked_bce_with_logits,
)
from tox21_risk.graph_data import ATOM_FEATURE_DIM, smiles_to_graph


TASKS = ("task_a", "task_b")


def _graphs():
    return [
        smiles_to_graph("mol_1", "CC", [0.0, 0.0]),
        smiles_to_graph("mol_2", "CCC", [1.0, 0.0]),
        smiles_to_graph("mol_3", "CO", [0.0, 1.0]),
        smiles_to_graph("mol_4", "CN", [1.0, 1.0]),
    ]


def test_masked_bce_matches_hand_selected_observed_values() -> None:
    logits = torch.tensor([[0.0, 2.0], [-2.0, 1.0]])
    targets = torch.tensor([[0.0, float("nan")], [1.0, 1.0]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0, -2.0, 1.0]), torch.tensor([0.0, 1.0, 1.0])
    )

    actual = masked_bce_with_logits(logits, targets)

    assert torch.allclose(actual, expected)


def test_masked_bce_rejects_all_missing_batch() -> None:
    with pytest.raises(ValueError, match="no observed labels"):
        masked_bce_with_logits(
            torch.zeros((2, 2)), torch.full((2, 2), float("nan"))
        )


def test_evaluate_model_excludes_missing_labels_from_endpoint_metrics() -> None:
    graphs = _graphs()
    graphs[0].y[0, 1] = float("nan")
    model = Tox21GCN(ATOM_FEATURE_DIM, len(TASKS), hidden_channels=8)
    loader = DataLoader(graphs, batch_size=4, shuffle=False)

    result = evaluate_model(model, loader, TASKS, torch.device("cpu"))

    metrics = result.endpoint_metrics.set_index("endpoint")
    assert metrics.loc["task_a", "n"] == 4
    assert metrics.loc["task_b", "n"] == 3
    assert math.isfinite(result.macro_endpoint_bce)
    assert result.labels.shape == (4, 2)
    assert np.isnan(result.labels[0, 1])
    assert result.mol_ids == ["mol_1", "mol_2", "mol_3", "mol_4"]


def test_evaluate_model_names_endpoint_with_one_observed_class() -> None:
    graphs = _graphs()
    for graph in graphs:
        graph.y[0, 1] = 0.0
    model = Tox21GCN(ATOM_FEATURE_DIM, len(TASKS), hidden_channels=8)

    with pytest.raises(ValueError, match="task_b.*both classes"):
        evaluate_model(
            model,
            DataLoader(graphs, batch_size=4, shuffle=False),
            TASKS,
            torch.device("cpu"),
        )


def test_fit_model_saves_and_restores_best_checkpoint(tmp_path) -> None:
    config = GCNConfig(
        seeds=(7,),
        hidden_channels=8,
        num_layers=3,
        dropout=0.2,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
        max_epochs=3,
        patience=2,
        device="cpu",
    )
    checkpoint_path = tmp_path / "best.pt"

    result = fit_model(
        _graphs(),
        _graphs(),
        TASKS,
        config,
        seed=7,
        split_strategy="random",
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["seed"] == 7
    assert checkpoint["split_strategy"] == "random"
    assert checkpoint["endpoint_order"] == list(TASKS)
    assert checkpoint["input_width"] == ATOM_FEATURE_DIM
    assert checkpoint["best_epoch"] == result.best_epoch
    assert math.isfinite(result.best_validation_bce)
    assert 1 <= len(result.history) <= 3
    assert set(result.history.columns) == {
        "epoch",
        "train_loss",
        "validation_macro_bce",
        "validation_macro_auroc",
    }
