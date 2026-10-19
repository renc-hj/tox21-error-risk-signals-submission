"""Run MC Dropout and compare endpoint-level error-risk signals."""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
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

from tox21_risk.fingerprints import morgan_fingerprint_matrix
from tox21_risk.gcn_model import GCNConfig, Tox21GCN
from tox21_risk.gcn_workflow import validate_gcn_artifacts
from tox21_risk.graph_data import ATOM_FEATURE_DIM, file_sha256, load_graph_cache
from tox21_risk.split_workflow import FINGERPRINT_RADIUS, SPLIT_COLUMNS
from tox21_risk.splitting import maximum_training_similarity
from tox21_risk.uncertainty import (
    MCDropoutResult,
    binary_classification_error,
    binary_cross_entropy_per_label,
    error_risk_metrics,
    mc_dropout_predict,
    single_pass_uncertainty,
)


SIGNAL_COLUMNS = (
    "structural_distance",
    "single_pass_uncertainty",
    "mc_dropout_mi",
)
SIMILARITY_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SIMILARITY_BIN_LABELS = (
    "0.0-0.2",
    "0.2-0.4",
    "0.4-0.6",
    "0.6-0.8",
    "0.8-1.0",
)


def run_uncertainty_workflow(
    molecule_path: Path,
    fingerprint_path: Path,
    graph_cache_path: Path,
    assignment_path: Path,
    checkpoint_dir: Path,
    predictions_path: Path,
    endpoint_metrics_path: Path,
    summary_path: Path,
    similarity_bins_path: Path,
    metadata_path: Path,
    tasks: Sequence[str],
    config: GCNConfig,
    *,
    n_passes: int = 50,
    classification_threshold: float = 0.5,
) -> dict[str, object]:
    """Create the endpoint-level table and aggregate three risk signals."""

    if n_passes < 2:
        raise ValueError("MC Dropout requires at least two passes")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification threshold must be in (0, 1)")

    molecules = pd.read_csv(molecule_path)
    fingerprints = np.load(fingerprint_path, allow_pickle=False)
    assignments = pd.read_csv(assignment_path, keep_default_na=False)
    cache = load_graph_cache(graph_cache_path)
    validate_gcn_artifacts(molecules, assignments, cache, tasks, molecule_path)
    _validate_fingerprints(molecules, fingerprints)
    graphs: list[Data] = cache["graphs"]
    device = torch.device(config.device)

    prediction_frames: list[pd.DataFrame] = []
    checkpoint_manifest: list[dict[str, object]] = []
    for strategy, split_column in SPLIT_COLUMNS.items():
        split_values = assignments[split_column].to_numpy()
        max_similarity = maximum_training_similarity(
            fingerprints,
            split_values,
            target_partitions=("validation", "test"),
        )
        similarity_by_mol_id = dict(
            zip(molecules["mol_id"].astype(str), max_similarity, strict=True)
        )
        partitions = {
            partition: [
                graph
                for graph, assigned in zip(graphs, split_values, strict=True)
                if assigned == partition
            ]
            for partition in ("validation", "test")
        }
        for seed in config.seeds:
            checkpoint_path = checkpoint_dir / strategy / f"seed_{seed}.pt"
            checkpoint = _load_checkpoint(
                checkpoint_path, strategy, seed, tasks, config
            )
            model = Tox21GCN(
                int(checkpoint["input_width"]),
                len(tasks),
                hidden_channels=config.hidden_channels,
                num_layers=config.num_layers,
                dropout=config.dropout,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            checkpoint_manifest.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "sha256": file_sha256(checkpoint_path),
                }
            )

            for partition_index, partition in enumerate(("validation", "test")):
                loader = DataLoader(
                    partitions[partition],
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                result = mc_dropout_predict(
                    model,
                    loader,
                    device,
                    n_passes=n_passes,
                    seed=seed + 100_000 * (partition_index + 1),
                )
                prediction_frames.append(
                    _prediction_frame(
                        result,
                        tasks,
                        strategy,
                        seed,
                        partition,
                        similarity_by_mol_id,
                        classification_threshold,
                    )
                )
            print(
                f"Completed MC Dropout for {strategy} seed {seed} "
                f"({n_passes} passes per partition)",
                flush=True,
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["similarity_bin"] = pd.cut(
        predictions["max_train_tanimoto"],
        bins=SIMILARITY_BIN_EDGES,
        labels=SIMILARITY_BIN_LABELS,
        include_lowest=True,
    )
    if predictions["similarity_bin"].isna().any():
        raise ValueError("some Tanimoto similarities fall outside [0, 1]")
    endpoint_metrics = _endpoint_signal_metrics(predictions)
    summary = _signal_summary(endpoint_metrics)
    similarity_bins = _similarity_bin_summary(predictions)

    for path in (
        predictions_path,
        endpoint_metrics_path,
        summary_path,
        similarity_bins_path,
        metadata_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, lineterminator="\n")
    endpoint_metrics.to_csv(
        endpoint_metrics_path, index=False, lineterminator="\n"
    )
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    similarity_bins.to_csv(
        similarity_bins_path, index=False, lineterminator="\n"
    )

    metadata: dict[str, object] = {
        "analysis": "endpoint_error_risk_signals",
        "n_molecules": len(molecules),
        "n_endpoints": len(tasks),
        "n_fits": len(SPLIT_COLUMNS) * len(config.seeds),
        "n_passes": n_passes,
        "n_prediction_rows": len(predictions),
        "endpoint_order": list(tasks),
        "signals": list(SIGNAL_COLUMNS),
        "target_error": "single-pass endpoint-level binary cross-entropy",
        "structural_distance_definition": "1 - maximum training-set Tanimoto",
        "single_pass_uncertainty_definition": "0.5 - abs(p - 0.5)",
        "mc_dropout_mi_definition": (
            "entropy(mean probability) - mean(entropy(pass probability))"
        ),
        "classification_threshold": classification_threshold,
        "mutual_information_unit": "nats",
        "similarity_metric": "Morgan radius-2 Tanimoto",
        "similarity_bins": list(SIMILARITY_BIN_LABELS),
        "mc_seed_rule": "training seed + 100000 for validation; +200000 for test",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "rdkit_version": rdkit.__version__,
        "molecule_file_sha256": file_sha256(molecule_path),
        "fingerprint_file_sha256": file_sha256(fingerprint_path),
        "graph_cache_file_sha256": file_sha256(graph_cache_path),
        "assignment_file_sha256": file_sha256(assignment_path),
        "checkpoint_manifest": checkpoint_manifest,
        "predictions_file_sha256": file_sha256(predictions_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _prediction_frame(
    result: MCDropoutResult,
    tasks: Sequence[str],
    strategy: str,
    seed: int,
    partition: str,
    similarity_by_mol_id: dict[str, float],
    classification_threshold: float,
) -> pd.DataFrame:
    n_molecules = len(result.mol_ids)
    labels = result.labels.reshape(-1)
    single = result.single_probabilities.reshape(-1)
    similarities = np.repeat(
        [similarity_by_mol_id[mol_id] for mol_id in result.mol_ids], len(tasks)
    )
    return pd.DataFrame(
        {
            "mol_id": np.repeat(result.mol_ids, len(tasks)),
            "strategy": strategy,
            "seed": seed,
            "partition": partition,
            "endpoint": np.tile(list(tasks), n_molecules),
            "label": labels,
            "single_probability": single,
            "mc_mean_probability": result.mean_probabilities.reshape(-1),
            "endpoint_bce": binary_cross_entropy_per_label(labels, single),
            "is_error": binary_classification_error(
                labels, single, threshold=classification_threshold
            ),
            "max_train_tanimoto": similarities,
            "structural_distance": 1.0 - similarities,
            "single_pass_uncertainty": single_pass_uncertainty(single),
            "mc_dropout_mi": result.mutual_information.reshape(-1),
        }
    )


def _endpoint_signal_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    group_columns = ["strategy", "seed", "partition", "endpoint"]
    for keys, group in predictions.groupby(group_columns, sort=False):
        metrics = error_risk_metrics(
            group["endpoint_bce"].to_numpy(),
            group["is_error"].to_numpy(),
            {name: group[name].to_numpy() for name in SIGNAL_COLUMNS},
        )
        for column, value in zip(group_columns, keys, strict=True):
            metrics[column] = value
        frames.append(metrics)
    columns = [
        *group_columns,
        "signal",
        "n_observed",
        "n_errors",
        "spearman_bce",
        "error_auroc",
        "error_auprc",
    ]
    return pd.concat(frames, ignore_index=True)[columns]


def _signal_summary(endpoint_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        endpoint_metrics.groupby(["strategy", "partition", "signal"], sort=False)
        .agg(
            n_seed_endpoint=("endpoint", "size"),
            spearman_bce_mean=("spearman_bce", "mean"),
            spearman_bce_std=("spearman_bce", "std"),
            error_auroc_mean=("error_auroc", "mean"),
            error_auroc_std=("error_auroc", "std"),
            error_auprc_mean=("error_auprc", "mean"),
            error_auprc_std=("error_auprc", "std"),
        )
        .reset_index()
    )


def _similarity_bin_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    observed = predictions.loc[predictions["label"].notna()].copy()
    summary = (
        observed.groupby(
            ["strategy", "partition", "similarity_bin"],
            sort=False,
            observed=True,
        )
        .agg(
            n_endpoint_rows=("endpoint_bce", "size"),
            n_unique_molecules=("mol_id", "nunique"),
            mean_max_train_tanimoto=("max_train_tanimoto", "mean"),
            mean_endpoint_bce=("endpoint_bce", "mean"),
            error_rate=("is_error", "mean"),
            mean_single_pass_uncertainty=("single_pass_uncertainty", "mean"),
            mean_mc_dropout_mi=("mc_dropout_mi", "mean"),
        )
        .reset_index()
    )
    summary["strategy"] = pd.Categorical(
        summary["strategy"], categories=tuple(SPLIT_COLUMNS), ordered=True
    )
    summary["partition"] = pd.Categorical(
        summary["partition"], categories=("validation", "test"), ordered=True
    )
    return summary.sort_values(
        ["strategy", "partition", "similarity_bin"]
    ).reset_index(drop=True)


def _validate_fingerprints(
    molecules: pd.DataFrame, fingerprints: np.ndarray
) -> None:
    if fingerprints.ndim != 2 or len(fingerprints) != len(molecules):
        raise ValueError("fingerprints must align one-to-one with molecules")
    expected = morgan_fingerprint_matrix(
        molecules["standardized_smiles"].tolist(),
        radius=FINGERPRINT_RADIUS,
        n_bits=fingerprints.shape[1],
    )
    if not np.array_equal(fingerprints, expected):
        raise ValueError("fingerprint rows are not aligned with the molecule table")


def _load_checkpoint(
    path: Path,
    strategy: str,
    seed: int,
    tasks: Sequence[str],
    config: GCNConfig,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing GCN checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid checkpoint payload: {path}")
    expected_config = asdict(config)
    expected_config["seeds"] = list(config.seeds)
    expected = {
        "seed": seed,
        "split_strategy": strategy,
        "input_width": ATOM_FEATURE_DIM,
        "endpoint_order": list(tasks),
        "config": expected_config,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"checkpoint {key} does not match requested analysis")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("checkpoint is missing model_state_dict")
    return checkpoint
