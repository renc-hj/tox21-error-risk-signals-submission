from pathlib import Path
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification

from tox21_risk.lr_baseline import (
    LRConfig,
    binary_probability_metrics,
    load_lr_config,
    select_endpoint_model,
)


def test_load_lr_config_reads_locked_baseline_choices(tmp_path: Path) -> None:
    path = tmp_path / "lr.yaml"
    path.write_text(
        """
seed: 2026
c_values: [0.01, 0.1, 1.0, 10.0]
solver: liblinear
penalty: l2
class_weight: balanced
max_iter: 5000
selection_metric: auroc
""".strip(),
        encoding="utf-8",
    )

    assert load_lr_config(path) == LRConfig()


def test_lr_config_rejects_invalid_candidates() -> None:
    with pytest.raises(ValueError, match="c_values"):
        LRConfig(c_values=(0.1, -1.0))


def test_binary_metrics_mask_missing_labels() -> None:
    y = np.array([0.0, 1.0, np.nan, 1.0])
    probability = np.array([0.1, 0.8, 0.99, 0.6])

    metrics = binary_probability_metrics(y, probability)

    assert metrics["n"] == 3
    assert metrics["positive"] == 2
    assert metrics["negative"] == 1
    assert metrics["positive_prevalence"] == pytest.approx(2 / 3)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["brier"] == pytest.approx(0.07)


def test_binary_metrics_reject_labels_other_than_zero_or_one() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        binary_probability_metrics(
            np.array([0.0, 2.0]), np.array([0.2, 0.8])
        )


def test_endpoint_model_selects_c_using_validation_auroc() -> None:
    features, labels = make_classification(
        n_samples=80,
        n_features=10,
        n_informative=3,
        n_redundant=2,
        class_sep=0.5,
        flip_y=0.15,
        random_state=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        selected = select_endpoint_model(
            features[:50],
            labels[:50].astype(float),
            features[50:],
            labels[50:].astype(float),
            endpoint="endpoint_a",
            config=LRConfig(),
        )

    assert selected.c_value == 0.01
    assert selected.validation_metrics["auroc"] == pytest.approx(
        0.8348214285714286
    )
    assert selected.n_train == 50
    assert selected.n_validation == 30


def test_endpoint_model_masks_missing_training_and_validation_labels() -> None:
    features, labels = make_classification(
        n_samples=80,
        n_features=8,
        n_informative=3,
        random_state=7,
    )
    train_labels = labels[:50].astype(float)
    validation_labels = labels[50:].astype(float)
    train_labels[[0, 4]] = np.nan
    validation_labels[3] = np.nan

    selected = select_endpoint_model(
        features[:50],
        train_labels,
        features[50:],
        validation_labels,
        endpoint="endpoint_a",
        config=LRConfig(c_values=(1.0,)),
    )

    assert selected.n_train == 48
    assert selected.n_validation == 29


def test_endpoint_model_rejects_one_class_partition_with_context() -> None:
    features = np.arange(40, dtype=float).reshape(10, 4)

    with pytest.raises(ValueError, match="endpoint_a training.*one class"):
        select_endpoint_model(
            features[:6],
            np.zeros(6),
            features[6:],
            np.array([0, 1, 0, 1], dtype=float),
            endpoint="endpoint_a",
            config=LRConfig(c_values=(1.0,)),
        )


def test_endpoint_model_rejects_nonbinary_training_labels() -> None:
    features = np.arange(48, dtype=float).reshape(12, 4)

    with pytest.raises(ValueError, match="endpoint_a training.*0 or 1"):
        select_endpoint_model(
            features[:8],
            np.array([0, 1, 0, 1, 0, 1, 0, 2], dtype=float),
            features[8:],
            np.array([0, 1, 0, 1], dtype=float),
            endpoint="endpoint_a",
            config=LRConfig(c_values=(1.0,)),
        )
