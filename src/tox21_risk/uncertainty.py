"""MC Dropout inference and endpoint-level error-risk calculations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.loader import DataLoader


_PROBABILITY_EPSILON = 1e-7


@dataclass(frozen=True)
class MCDropoutResult:
    """Deterministic predictions and MC Dropout summaries for one partition."""

    labels: np.ndarray
    single_probabilities: np.ndarray
    mean_probabilities: np.ndarray
    mutual_information: np.ndarray
    mol_ids: list[str]


def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_passes: int,
    seed: int,
) -> MCDropoutResult:
    """Run one deterministic pass and reproducible stochastic dropout passes."""

    if n_passes < 2:
        raise ValueError("MC Dropout requires at least two passes")
    original_training_mode = model.training
    model.to(device)
    try:
        model.eval()
        labels, single_probabilities, mol_ids = _predict_loader(
            model, loader, device
        )
        draws: list[np.ndarray] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.train()
            for _ in range(n_passes):
                pass_labels, pass_probabilities, pass_mol_ids = _predict_loader(
                    model, loader, device
                )
                if pass_mol_ids != mol_ids or not np.array_equal(
                    pass_labels, labels, equal_nan=True
                ):
                    raise ValueError(
                        "evaluation loader order changed across MC passes"
                    )
                draws.append(pass_probabilities)
    finally:
        model.train(original_training_mode)

    stacked = np.stack(draws, axis=0)
    return MCDropoutResult(
        labels=labels,
        single_probabilities=single_probabilities,
        mean_probabilities=stacked.mean(axis=0),
        mutual_information=mc_dropout_mutual_information(stacked),
        mol_ids=mol_ids,
    )


def mc_dropout_mutual_information(probabilities: np.ndarray) -> np.ndarray:
    """Return Bernoulli mutual information from pass × molecule × task values."""

    draws = np.asarray(probabilities, dtype=float)
    if draws.ndim != 3 or draws.shape[0] < 2:
        raise ValueError(
            "MC probabilities must have shape (passes, molecules, endpoints) "
            "with at least two passes"
        )
    _validate_probabilities(draws)
    predictive_entropy = _bernoulli_entropy(draws.mean(axis=0))
    expected_entropy = _bernoulli_entropy(draws).mean(axis=0)
    information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    information[np.abs(information) < 1e-12] = 0.0
    return information


def binary_cross_entropy_per_label(
    labels: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Return one BCE value per observed label and NaN for missing labels."""

    targets, predictions = _aligned_label_probabilities(labels, probabilities)
    clipped = np.clip(predictions, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    losses = np.full(targets.shape, np.nan, dtype=float)
    observed = ~np.isnan(targets)
    losses[observed] = -(
        targets[observed] * np.log(clipped[observed])
        + (1.0 - targets[observed]) * np.log(1.0 - clipped[observed])
    )
    return losses


def binary_classification_error(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Return 1 for a thresholded error, 0 for correct, and NaN if unlabelled."""

    targets, predictions = _aligned_label_probabilities(labels, probabilities)
    if not 0.0 < threshold < 1.0:
        raise ValueError("classification threshold must be in (0, 1)")
    errors = np.full(targets.shape, np.nan, dtype=float)
    observed = ~np.isnan(targets)
    errors[observed] = (
        (predictions[observed] >= threshold).astype(float) != targets[observed]
    ).astype(float)
    return errors


def single_pass_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """Convert distance from 0.5 into a risk-oriented uncertainty score."""

    predictions = np.asarray(probabilities, dtype=float)
    _validate_probabilities(predictions)
    return 0.5 - np.abs(predictions - 0.5)


def error_risk_metrics(
    endpoint_bce: np.ndarray,
    is_error: np.ndarray,
    signals: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Score risk signals against continuous BCE and thresholded errors."""

    losses = np.asarray(endpoint_bce, dtype=float)
    errors = np.asarray(is_error, dtype=float)
    if losses.shape != errors.shape:
        raise ValueError("endpoint BCE and error indicators must have the same shape")
    rows: list[dict[str, float | int | str]] = []
    for name, values in signals.items():
        risk = np.asarray(values, dtype=float)
        if risk.shape != losses.shape:
            raise ValueError(f"{name} must have the same shape as endpoint BCE")
        observed = np.isfinite(losses) & np.isfinite(errors) & np.isfinite(risk)
        selected_losses = losses[observed]
        selected_errors = errors[observed]
        selected_risk = risk[observed]
        n_observed = int(observed.sum())
        if n_observed >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                correlation = float(
                    spearmanr(selected_risk, selected_losses).statistic
                )
        else:
            correlation = float("nan")
        if np.unique(selected_errors).size == 2:
            error_auroc = float(roc_auc_score(selected_errors, selected_risk))
            error_auprc = float(
                average_precision_score(selected_errors, selected_risk)
            )
        else:
            error_auroc = float("nan")
            error_auprc = float("nan")
        rows.append(
            {
                "signal": name,
                "n_observed": n_observed,
                "n_errors": int(selected_errors.sum()),
                "spearman_bce": correlation,
                "error_auroc": error_auroc,
                "error_auprc": error_auprc,
            }
        )
    return pd.DataFrame(rows)


def _bernoulli_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        probabilities, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON
    )
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def _aligned_label_probabilities(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(labels, dtype=float)
    predictions = np.asarray(probabilities, dtype=float)
    if targets.shape != predictions.shape:
        raise ValueError("labels and probabilities must have the same shape")
    observed = targets[~np.isnan(targets)]
    if not np.isin(observed, (0.0, 1.0)).all():
        raise ValueError("labels must be binary or missing")
    _validate_probabilities(predictions)
    return targets, predictions


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")


def _predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    label_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    mol_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            label_parts.append(batch.y.detach().cpu().numpy())
            probability_parts.append(torch.sigmoid(logits).cpu().numpy())
            mol_ids.extend(str(mol_id) for mol_id in batch.mol_id)
    if not label_parts:
        raise ValueError("evaluation loader contains no graphs")
    labels = np.concatenate(label_parts, axis=0)
    probabilities = np.concatenate(probability_parts, axis=0)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and model predictions must have the same shape")
    return labels, probabilities, mol_ids
