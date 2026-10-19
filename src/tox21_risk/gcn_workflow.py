"""Run the fixed nine-fit Tox21 GCN experiment and write result tables."""

from __future__ import annotations

import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdkit
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tox21_risk.gcn_model import GCNConfig
from tox21_risk.gcn_training import FitResult, evaluate_model, fit_model
from tox21_risk.graph_data import (
    ATOM_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    file_sha256,
    load_graph_cache,
)
from tox21_risk.split_workflow import SPLIT_COLUMNS


FitFunction = Callable[
    [Sequence[Data], Sequence[Data], Sequence[str], GCNConfig, int, str, Path],
    FitResult,
]


def run_gcn_workflow(
    molecule_path: Path,
    graph_cache_path: Path,
    assignment_path: Path,
    endpoint_metrics_path: Path,
    split_summary_path: Path,
    training_summary_path: Path,
    metadata_path: Path,
    predictions_path: Path,
    history_path: Path,
    checkpoint_dir: Path,
    tasks: Sequence[str],
    config: GCNConfig,
    fit_fn: FitFunction = fit_model,
) -> dict[str, object]:
    """Train all strategy-seed combinations and save the fixed output contract."""

    molecules = pd.read_csv(molecule_path)
    assignments = pd.read_csv(assignment_path, keep_default_na=False)
    cache = load_graph_cache(graph_cache_path)
    validate_gcn_artifacts(molecules, assignments, cache, tasks, molecule_path)
    graphs: list[Data] = cache["graphs"]

    endpoint_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    device = torch.device(config.device)

    for strategy, split_column in SPLIT_COLUMNS.items():
        split_values = assignments[split_column].to_numpy()
        partitions = {
            partition: [
                graph
                for graph, assigned in zip(graphs, split_values, strict=True)
                if assigned == partition
            ]
            for partition in ("train", "validation", "test")
        }
        for seed in config.seeds:
            checkpoint_path = checkpoint_dir / strategy / f"seed_{seed}.pt"
            fit = fit_fn(
                partitions["train"],
                partitions["validation"],
                tasks,
                config,
                seed,
                strategy,
                checkpoint_path,
            )
            training_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "n_train": len(partitions["train"]),
                    "n_validation": len(partitions["validation"]),
                    "best_epoch": fit.best_epoch,
                    "epochs_ran": len(fit.history),
                    "best_validation_macro_bce": fit.best_validation_bce,
                }
            )
            history = fit.history.copy()
            history.insert(0, "seed", seed)
            history.insert(0, "strategy", strategy)
            history_frames.append(history)

            for partition in ("validation", "test"):
                loader = DataLoader(
                    partitions[partition],
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                evaluation = evaluate_model(fit.model, loader, tasks, device)
                metrics = evaluation.endpoint_metrics.copy()
                metrics.insert(0, "partition", partition)
                metrics.insert(0, "seed", seed)
                metrics.insert(0, "strategy", strategy)
                endpoint_frames.append(metrics)
                prediction_frames.append(
                    _prediction_frame(evaluation, tasks, strategy, seed, partition)
                )
            print(
                f"Completed {strategy} seed {seed}: best epoch "
                f"{fit.best_epoch}, validation macro BCE "
                f"{fit.best_validation_bce:.6f}",
                flush=True,
            )

    endpoint_metrics = pd.concat(endpoint_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    histories = pd.concat(history_frames, ignore_index=True)
    training_summary = pd.DataFrame(training_rows)
    seed_macro = (
        endpoint_metrics.groupby(["strategy", "seed", "partition"], sort=False)
        .agg(
            macro_bce=("bce", "mean"),
            macro_auroc=("auroc", "mean"),
            macro_auprc=("auprc", "mean"),
            macro_brier=("brier", "mean"),
        )
        .reset_index()
    )
    split_summary = (
        seed_macro.groupby(["strategy", "partition"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            macro_bce_mean=("macro_bce", "mean"),
            macro_bce_std=("macro_bce", "std"),
            macro_auroc_mean=("macro_auroc", "mean"),
            macro_auroc_std=("macro_auroc", "std"),
            macro_auprc_mean=("macro_auprc", "mean"),
            macro_auprc_std=("macro_auprc", "std"),
            macro_brier_mean=("macro_brier", "mean"),
            macro_brier_std=("macro_brier", "std"),
        )
        .reset_index()
    )

    config_values = asdict(config)
    config_values["seeds"] = list(config.seeds)
    metadata: dict[str, object] = {
        "model": "Tox21GCN",
        "n_molecules": len(molecules),
        "n_endpoints": len(tasks),
        "n_fits": len(SPLIT_COLUMNS) * len(config.seeds),
        "endpoint_order": list(tasks),
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "bond_features": False,
        "missing_label_policy": "excluded with endpoint masks",
        "selection_metric": "validation_macro_endpoint_bce",
        "config": config_values,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "rdkit_version": rdkit.__version__,
        "molecule_file_sha256": file_sha256(molecule_path),
        "graph_cache_file_sha256": file_sha256(graph_cache_path),
        "assignment_file_sha256": file_sha256(assignment_path),
    }
    for path in (
        endpoint_metrics_path,
        split_summary_path,
        training_summary_path,
        metadata_path,
        predictions_path,
        history_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    endpoint_metrics.to_csv(endpoint_metrics_path, index=False, lineterminator="\n")
    split_summary.to_csv(split_summary_path, index=False, lineterminator="\n")
    training_summary.to_csv(
        training_summary_path, index=False, lineterminator="\n"
    )
    predictions.to_csv(predictions_path, index=False, lineterminator="\n")
    histories.to_csv(history_path, index=False, lineterminator="\n")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def validate_gcn_artifacts(
    molecules: pd.DataFrame,
    assignments: pd.DataFrame,
    cache: dict[str, Any],
    tasks: Sequence[str],
    molecule_path: Path,
) -> None:
    """Reject stale, reordered, or schema-incompatible experiment inputs."""

    required_molecule_columns = {"mol_id", "standardized_smiles", *tasks}
    missing_molecule_columns = sorted(
        required_molecule_columns.difference(molecules.columns)
    )
    if missing_molecule_columns:
        raise ValueError(
            "molecule table is missing columns: "
            + ", ".join(missing_molecule_columns)
        )
    required_assignment_columns = {"mol_id", *SPLIT_COLUMNS.values()}
    missing_assignment_columns = sorted(
        required_assignment_columns.difference(assignments.columns)
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

    labels = molecules[list(tasks)].to_numpy(dtype=float)
    observed = labels[~np.isnan(labels)]
    if not np.isin(observed, (0.0, 1.0)).all():
        raise ValueError("endpoint labels must be binary or missing")
    if cache.get("molecule_file_sha256") != file_sha256(molecule_path):
        raise ValueError("graph cache molecule SHA-256 does not match molecule file")
    if cache.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("graph cache feature schema version is incompatible")
    if cache.get("feature_dim") != ATOM_FEATURE_DIM:
        raise ValueError("graph cache feature width is incompatible")
    if cache.get("endpoint_order") != list(tasks):
        raise ValueError("graph cache endpoint order is incompatible")
    if cache.get("molecule_count") != len(molecules):
        raise ValueError("graph cache molecule count does not match molecule table")
    if cache.get("mol_ids") != molecules["mol_id"].astype(str).tolist():
        raise ValueError("graph cache mol_id order does not match molecule table")

    graphs = cache.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != len(molecules):
        raise ValueError("graph cache graph count does not match molecule table")
    for row_index, graph in enumerate(graphs):
        if str(graph.mol_id) != str(molecules.iloc[row_index]["mol_id"]):
            raise ValueError("graph mol_id order does not match molecule table")
        if str(graph.standardized_smiles) != str(
            molecules.iloc[row_index]["standardized_smiles"]
        ):
            raise ValueError(
                "graph standardized SMILES does not match molecule table"
            )
        if graph.x.ndim != 2 or graph.x.shape[1] != ATOM_FEATURE_DIM:
            raise ValueError("graph feature width does not match locked schema")
        graph_labels = graph.y.detach().cpu().numpy().reshape(-1)
        if graph_labels.shape != (len(tasks),) or not np.allclose(
            graph_labels, labels[row_index], equal_nan=True
        ):
            raise ValueError("graph labels do not match molecule table")

    expected_partitions = {"train", "validation", "test"}
    for strategy, split_column in SPLIT_COLUMNS.items():
        if set(assignments[split_column]) != expected_partitions:
            raise ValueError(
                f"{strategy} split partition values must be train, validation, test"
            )


def _prediction_frame(
    evaluation: Any,
    tasks: Sequence[str],
    strategy: str,
    seed: int,
    partition: str,
) -> pd.DataFrame:
    n_molecules = len(evaluation.mol_ids)
    return pd.DataFrame(
        {
            "mol_id": np.repeat(evaluation.mol_ids, len(tasks)),
            "strategy": strategy,
            "seed": seed,
            "partition": partition,
            "endpoint": np.tile(list(tasks), n_molecules),
            "label": evaluation.labels.reshape(-1),
            "probability": evaluation.probabilities.reshape(-1),
        }
    )
