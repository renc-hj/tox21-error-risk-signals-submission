from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from tox21_risk.gcn_model import Tox21GCN
from tox21_risk.graph_data import ATOM_FEATURE_DIM, smiles_to_graph
from tox21_risk.uncertainty import (
    binary_classification_error,
    binary_cross_entropy_per_label,
    error_risk_metrics,
    mc_dropout_predict,
    mc_dropout_mutual_information,
    single_pass_uncertainty,
)


def test_mc_dropout_mutual_information_matches_hand_calculation() -> None:
    probabilities = np.array([[[0.1]], [[0.9]]], dtype=float)

    result = mc_dropout_mutual_information(probabilities)

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(0.3680642072)


def test_mc_dropout_mutual_information_is_zero_for_identical_passes() -> None:
    probabilities = np.full((5, 2, 3), 0.25, dtype=float)

    result = mc_dropout_mutual_information(probabilities)

    assert np.array_equal(result, np.zeros((2, 3)))


def test_endpoint_error_values_exclude_missing_labels() -> None:
    labels = np.array([1.0, 0.0, np.nan])
    probabilities = np.array([0.8, 0.8, 0.1])

    bce = binary_cross_entropy_per_label(labels, probabilities)
    is_error = binary_classification_error(labels, probabilities)

    assert bce[:2] == pytest.approx([-np.log(0.8), -np.log(0.2)])
    assert np.isnan(bce[2])
    assert is_error[:2].tolist() == [0.0, 1.0]
    assert np.isnan(is_error[2])


def test_single_pass_uncertainty_is_highest_at_half_probability() -> None:
    probabilities = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

    result = single_pass_uncertainty(probabilities)

    assert result == pytest.approx([0.0, 0.2, 0.5, 0.2, 0.0])


def test_mc_dropout_predict_is_reproducible_and_restores_eval_mode() -> None:
    graphs = [
        smiles_to_graph("mol_1", "CC", [0.0, 1.0]),
        smiles_to_graph("mol_2", "CCC", [1.0, np.nan]),
        smiles_to_graph("mol_3", "CO", [0.0, 0.0]),
    ]
    torch.manual_seed(7)
    model = Tox21GCN(
        ATOM_FEATURE_DIM,
        2,
        hidden_channels=8,
        num_layers=3,
        dropout=0.8,
    )
    model.eval()
    loader = DataLoader(graphs, batch_size=2, shuffle=False)

    first = mc_dropout_predict(
        model, loader, torch.device("cpu"), n_passes=6, seed=99
    )
    second = mc_dropout_predict(
        model, loader, torch.device("cpu"), n_passes=6, seed=99
    )

    assert first.mol_ids == ["mol_1", "mol_2", "mol_3"]
    assert first.labels.shape == (3, 2)
    assert first.single_probabilities.shape == (3, 2)
    assert first.mean_probabilities.shape == (3, 2)
    assert first.mutual_information.shape == (3, 2)
    assert np.array_equal(first.mean_probabilities, second.mean_probabilities)
    assert np.array_equal(first.mutual_information, second.mutual_information)
    assert (first.mutual_information >= 0.0).all()
    assert (first.mutual_information > 0.0).any()
    assert not model.training


def test_mc_dropout_predict_requires_multiple_passes() -> None:
    graph = smiles_to_graph("mol_1", "CC", [0.0])
    model = Tox21GCN(ATOM_FEATURE_DIM, 1, hidden_channels=8)

    with pytest.raises(ValueError, match="at least two"):
        mc_dropout_predict(
            model,
            DataLoader([graph], batch_size=1),
            torch.device("cpu"),
            n_passes=1,
            seed=1,
        )


def test_error_risk_metrics_rewards_signal_that_orders_errors_first() -> None:
    bce = np.array([0.1, 0.2, 0.8, 1.0])
    is_error = np.array([0.0, 0.0, 1.0, 1.0])
    signals = {
        "perfect": np.array([0.1, 0.2, 0.8, 0.9]),
        "reversed": np.array([0.9, 0.8, 0.2, 0.1]),
    }

    result = error_risk_metrics(bce, is_error, signals).set_index("signal")

    assert result.loc["perfect", "n_observed"] == 4
    assert result.loc["perfect", "n_errors"] == 2
    assert result.loc["perfect", "spearman_bce"] == pytest.approx(1.0)
    assert result.loc["perfect", "error_auroc"] == pytest.approx(1.0)
    assert result.loc["perfect", "error_auprc"] == pytest.approx(1.0)
    assert result.loc["reversed", "spearman_bce"] == pytest.approx(-1.0)
    assert result.loc["reversed", "error_auroc"] == pytest.approx(0.0)


def test_error_risk_metrics_marks_single_error_class_as_undefined() -> None:
    result = error_risk_metrics(
        np.array([0.1, 0.2]),
        np.array([0.0, 0.0]),
        {"risk": np.array([0.1, 0.2])},
    )

    assert np.isnan(result.loc[0, "error_auroc"])
    assert np.isnan(result.loc[0, "error_auprc"])
