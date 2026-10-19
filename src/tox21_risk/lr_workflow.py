"""Run the LR baseline against the repository's fixed Tox21 artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tox21_risk.fingerprints import morgan_fingerprint_matrix
from tox21_risk.lr_baseline import (
    LRConfig,
    binary_probability_metrics,
    select_endpoint_model,
)
from tox21_risk.split_workflow import FINGERPRINT_RADIUS, SPLIT_COLUMNS


def run_lr_workflow(
    molecule_path: Path,
    fingerprint_path: Path,
    assignment_path: Path,
    metrics_path: Path,
    summary_path: Path,
    predictions_path: Path,
    metadata_path: Path,
    tasks: Sequence[str],
    config: LRConfig,
) -> dict[str, object]:
    """Train endpoint-specific LR models and write reproducible results."""

    molecules = pd.read_csv(molecule_path)
    fingerprints = np.load(fingerprint_path, allow_pickle=False)
    assignments = pd.read_csv(assignment_path, keep_default_na=False)
    _validate_artifacts(molecules, fingerprints, assignments, tasks)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for strategy, split_column in SPLIT_COLUMNS.items():
        split_values = assignments[split_column].to_numpy()
        train_mask = split_values == "train"
        validation_mask = split_values == "validation"

        for endpoint in tasks:
            labels = molecules[endpoint].to_numpy(dtype=float)
            selected = select_endpoint_model(
                fingerprints[train_mask],
                labels[train_mask],
                fingerprints[validation_mask],
                labels[validation_mask],
                endpoint=endpoint,
                config=config,
            )
            iterations = int(np.asarray(selected.model.n_iter_).max())

            for partition in ("validation", "test"):
                partition_mask = split_values == partition
                partition_probabilities = selected.model.predict_proba(
                    fingerprints[partition_mask]
                )[:, 1]
                try:
                    metrics = binary_probability_metrics(
                        labels[partition_mask], partition_probabilities
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{endpoint} {strategy} {partition}: {error}"
                    ) from error
                metric_rows.append(
                    {
                        "strategy": strategy,
                        "endpoint": endpoint,
                        "partition": partition,
                        "c_value": selected.c_value,
                        "n_train": selected.n_train,
                        "n_partition": metrics["n"],
                        "positive": metrics["positive"],
                        "negative": metrics["negative"],
                        "positive_prevalence": metrics[
                            "positive_prevalence"
                        ],
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "brier": metrics["brier"],
                        "iterations": iterations,
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "mol_id": molecules.loc[partition_mask, "mol_id"],
                            "strategy": strategy,
                            "endpoint": endpoint,
                            "partition": partition,
                            "label": labels[partition_mask],
                            "probability": partition_probabilities,
                        }
                    )
                )

    endpoint_metrics = pd.DataFrame(metric_rows)
    split_summary = (
        endpoint_metrics.groupby(["strategy", "partition"], sort=False)
        .agg(
            n_endpoints=("endpoint", "nunique"),
            macro_auroc=("auroc", "mean"),
            macro_auprc=("auprc", "mean"),
            macro_brier=("brier", "mean"),
        )
        .reset_index()
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metadata: dict[str, object] = {
        "model": "LogisticRegression",
        "n_molecules": len(molecules),
        "n_endpoints": len(tasks),
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_bits": int(fingerprints.shape[1]),
        "seed": config.seed,
        "c_values": list(config.c_values),
        "selection_metric": config.selection_metric,
        "solver": config.solver,
        "penalty": config.penalty,
        "class_weight": config.class_weight,
        "max_iter": config.max_iter,
        "missing_label_policy": "excluded per endpoint",
        "molecule_file_sha256": _sha256(molecule_path),
        "fingerprint_file_sha256": _sha256(fingerprint_path),
        "assignment_file_sha256": _sha256(assignment_path),
    }

    for path in (
        metrics_path,
        summary_path,
        predictions_path,
        metadata_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    endpoint_metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    split_summary.to_csv(summary_path, index=False, lineterminator="\n")
    predictions.to_csv(predictions_path, index=False, lineterminator="\n")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _validate_artifacts(
    molecules: pd.DataFrame,
    fingerprints: np.ndarray,
    assignments: pd.DataFrame,
    tasks: Sequence[str],
) -> None:
    molecule_columns = {"mol_id", "standardized_smiles", *tasks}
    missing_molecule_columns = sorted(molecule_columns.difference(molecules.columns))
    if missing_molecule_columns:
        raise ValueError(
            "molecule table is missing columns: "
            + ", ".join(missing_molecule_columns)
        )
    assignment_columns = {"mol_id", *SPLIT_COLUMNS.values()}
    missing_assignment_columns = sorted(
        assignment_columns.difference(assignments.columns)
    )
    if missing_assignment_columns:
        raise ValueError(
            "assignment table is missing columns: "
            + ", ".join(missing_assignment_columns)
        )
    if molecules["mol_id"].isna().any() or not molecules["mol_id"].is_unique:
        raise ValueError("molecule mol_id values must be present and unique")
    if assignments["mol_id"].isna().any() or not assignments["mol_id"].is_unique:
        raise ValueError("assignment mol_id values must be present and unique")
    if not molecules["mol_id"].equals(assignments["mol_id"]):
        raise ValueError(
            "molecule and assignment mol_id values must appear in the same order"
        )
    if len(fingerprints) != len(molecules):
        raise ValueError("molecule and fingerprint row counts must match")
    expected = morgan_fingerprint_matrix(
        molecules["standardized_smiles"].tolist(),
        radius=FINGERPRINT_RADIUS,
        n_bits=fingerprints.shape[1],
    )
    if not np.array_equal(fingerprints, expected):
        raise ValueError("fingerprint rows are not aligned with the molecule table")
    expected_partitions = {"train", "validation", "test"}
    for strategy, split_column in SPLIT_COLUMNS.items():
        observed = set(assignments[split_column])
        if observed != expected_partitions:
            raise ValueError(
                f"{strategy} split must contain train, validation, and test"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
