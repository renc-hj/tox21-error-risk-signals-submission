"""Endpoint-specific Logistic Regression baseline and probability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True, slots=True)
class LRConfig:
    """Locked Logistic Regression choices used by the baseline."""

    seed: int = 2026
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    solver: str = "liblinear"
    penalty: str = "l2"
    class_weight: str = "balanced"
    max_iter: int = 5_000
    selection_metric: str = "auroc"

    def __post_init__(self) -> None:
        if not self.c_values or any(value <= 0 for value in self.c_values):
            raise ValueError("c_values must contain positive values")
        if tuple(sorted(self.c_values)) != self.c_values:
            raise ValueError("c_values must be in ascending order")
        if self.solver != "liblinear" or self.penalty != "l2":
            raise ValueError("the baseline requires liblinear with L2 penalty")
        if self.class_weight != "balanced":
            raise ValueError("class_weight must be 'balanced'")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.selection_metric != "auroc":
            raise ValueError("selection_metric must be 'auroc'")


@dataclass(frozen=True, slots=True)
class EndpointModelSelection:
    """Selected fitted model and its validation evidence."""

    model: LogisticRegression
    c_value: float
    validation_metrics: dict[str, float | int]
    n_train: int
    n_validation: int


def load_lr_config(path: Path) -> LRConfig:
    """Load and validate the small YAML configuration for the LR baseline."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("LR configuration must be a YAML mapping")
    candidates = raw.get("c_values")
    if not isinstance(candidates, list) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in candidates
    ):
        raise ValueError("c_values must be a list of numbers")
    return LRConfig(
        seed=_integer(raw, "seed"),
        c_values=tuple(float(value) for value in candidates),
        solver=_string(raw, "solver"),
        penalty=_string(raw, "penalty"),
        class_weight=_string(raw, "class_weight"),
        max_iter=_integer(raw, "max_iter"),
        selection_metric=_string(raw, "selection_metric"),
    )


def binary_probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | int]:
    """Calculate binary metrics after excluding missing labels."""

    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("labels and probabilities must be aligned 1D arrays")
    observed = ~np.isnan(y)
    y = y[observed]
    p = p[observed]
    if len(y) == 0:
        raise ValueError("no observed labels remain after masking")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("observed labels must be 0 or 1")
    y = y.astype(int)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    if len(np.unique(y)) != 2:
        raise ValueError("probability metrics require both classes")
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    return {
        "n": len(y),
        "positive": positive,
        "negative": negative,
        "positive_prevalence": positive / len(y),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def select_endpoint_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    endpoint: str,
    config: LRConfig,
) -> EndpointModelSelection:
    """Select `C` using validation AUROC and return the fitted train-only model."""

    train_features, train_labels = _observed_partition(
        x_train, y_train, endpoint, "training"
    )
    validation_features, validation_labels = _observed_partition(
        x_validation, y_validation, endpoint, "validation"
    )

    best_model: LogisticRegression | None = None
    best_c: float | None = None
    best_metrics: dict[str, float | int] | None = None
    best_score = -np.inf
    for c_value in config.c_values:
        model = LogisticRegression(
            C=c_value,
            solver=config.solver,
            l1_ratio=0.0,
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            random_state=config.seed,
        )
        model.fit(train_features, train_labels)
        probabilities = model.predict_proba(validation_features)[:, 1]
        metrics = binary_probability_metrics(validation_labels, probabilities)
        score = float(metrics[config.selection_metric])
        if score > best_score:
            best_model = model
            best_c = c_value
            best_metrics = metrics
            best_score = score

    if best_model is None or best_c is None or best_metrics is None:
        raise RuntimeError(f"no LR model was selected for {endpoint}")
    return EndpointModelSelection(
        model=best_model,
        c_value=best_c,
        validation_metrics=best_metrics,
        n_train=len(train_labels),
        n_validation=len(validation_labels),
    )


def _observed_partition(
    features: np.ndarray,
    labels: np.ndarray,
    endpoint: str,
    partition: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features)
    y = np.asarray(labels, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"{endpoint} {partition} features and labels are not aligned")
    observed = ~np.isnan(y)
    x = x[observed]
    y = y[observed]
    if len(y) == 0:
        raise ValueError(f"{endpoint} {partition} partition has no observed labels")
    if not np.isin(y, (0, 1)).all():
        raise ValueError(
            f"{endpoint} {partition} labels must be 0 or 1"
        )
    y = y.astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{endpoint} {partition} partition has one class")
    return x, y


def _integer(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value
