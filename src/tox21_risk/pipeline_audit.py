"""Independent consistency checks for the completed Tox21 pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


AuditResult = dict[str, object]
PARTITIONS = {"train", "validation", "test"}
LOCKED_EXPECTATIONS: dict[str, object] = {
    "split.fingerprint_radius": 2,
    "split.fingerprint_bits": 1024,
    "split.similarity_threshold": 0.6,
    "split.distance_cutoff": 0.4,
    "lr.n_endpoints": 12,
    "lr.fingerprint_radius": 2,
    "lr.fingerprint_bits": 1024,
    "lr.c_values": [0.01, 0.1, 1.0, 10.0],
    "lr.selection_metric": "auroc",
    "lr.class_weight": "balanced",
    "gcn.n_endpoints": 12,
    "gcn.n_fits": 9,
    "gcn.atom_feature_dim": 166,
    "gcn.bond_features": False,
    "gcn.selection_metric": "validation_macro_endpoint_bce",
    "gcn.config.seeds": [2026, 2027, 2028],
    "gcn.config.hidden_channels": 128,
    "gcn.config.num_layers": 3,
    "gcn.config.dropout": 0.2,
    "gcn.config.batch_size": 64,
    "gcn.config.learning_rate": 0.001,
    "gcn.config.weight_decay": 0.00001,
    "gcn.config.max_epochs": 200,
    "gcn.config.patience": 20,
    "gcn.config.device": "cpu",
    "mc_dropout.n_endpoints": 12,
    "mc_dropout.n_fits": 9,
    "mc_dropout.n_passes": 50,
    "mc_dropout.classification_threshold": 0.5,
    "mc_dropout.signals": [
        "structural_distance",
        "single_pass_uncertainty",
        "mc_dropout_mi",
    ],
    "results.primary_partition": "test",
    "results.minimum_observed_labels_per_paired_cell": 20,
    "results.paired_difference_aggregation": (
        "mean_within_endpoint_then_macro_across_endpoints"
    ),
    "results.paired_difference_ci.method": (
        "endpoint_cluster_percentile_bootstrap"
    ),
    "results.paired_difference_ci.confidence_level": 0.95,
    "results.paired_difference_ci.n_resamples": 10_000,
    "results.paired_difference_ci.seed": 2026,
}


def audit_experiment_contract(
    metadata: Mapping[str, Mapping[str, object]],
) -> AuditResult:
    """Check the locked settings that define the reported experiment."""

    mismatches: dict[str, dict[str, object]] = {}
    for dotted_key, expected in LOCKED_EXPECTATIONS.items():
        actual = _nested_value(metadata, dotted_key)
        if actual != expected:
            mismatches[dotted_key] = {"expected": expected, "actual": actual}
    return {
        "name": "experiment contract",
        "passed": not mismatches,
        "details": {
            "n_locked_values": len(LOCKED_EXPECTATIONS),
            "mismatches": mismatches,
        },
    }


def run_pipeline_audit(repo_root: Path, output_path: Path) -> dict[str, object]:
    """Audit the complete saved pipeline without retraining any model."""

    root = repo_root.resolve()
    tables = root / "reports" / "tables"
    metadata = {
        name: _read_json(tables / name)
        for name in (
            "split_metadata.json",
            "lr_metadata.json",
            "gcn_metadata.json",
            "mc_dropout_metadata.json",
            "results_analysis_metadata.json",
        )
    }
    paths = {
        "molecules": Path("data/processed/tox21_molecules.csv"),
        "fingerprints": Path("data/processed/morgan_r2_1024.npy"),
        "assignments": Path("data/processed/split_assignments.csv"),
        "graphs": Path("data/processed/tox21_graphs.pt"),
        "predictions": Path("data/processed/mc_dropout_predictions.csv"),
    }

    molecules = pd.read_csv(root / paths["molecules"])
    assignments = pd.read_csv(
        root / paths["assignments"], keep_default_na=False
    )
    checks: list[AuditResult] = [
        audit_molecule_identity(
            molecules,
            expected_count=int(metadata["split_metadata.json"]["n_molecules"]),
        ),
        audit_split_integrity(assignments),
        audit_experiment_contract(
            {
                "split": metadata["split_metadata.json"],
                "lr": metadata["lr_metadata.json"],
                "gcn": metadata["gcn_metadata.json"],
                "mc_dropout": metadata["mc_dropout_metadata.json"],
                "results": metadata["results_analysis_metadata.json"],
            }
        ),
    ]

    artifact_specs = (
        (
            "Split molecules",
            paths["molecules"],
            metadata["split_metadata.json"]["molecule_file_sha256"],
        ),
        (
            "Split fingerprints",
            paths["fingerprints"],
            metadata["split_metadata.json"]["fingerprint_file_sha256"],
        ),
        (
            "LR molecules",
            paths["molecules"],
            metadata["lr_metadata.json"]["molecule_file_sha256"],
        ),
        (
            "LR fingerprints",
            paths["fingerprints"],
            metadata["lr_metadata.json"]["fingerprint_file_sha256"],
        ),
        (
            "LR assignments",
            paths["assignments"],
            metadata["lr_metadata.json"]["assignment_file_sha256"],
        ),
        (
            "GCN molecules",
            paths["molecules"],
            metadata["gcn_metadata.json"]["molecule_file_sha256"],
        ),
        (
            "GCN assignments",
            paths["assignments"],
            metadata["gcn_metadata.json"]["assignment_file_sha256"],
        ),
        (
            "GCN graph cache",
            paths["graphs"],
            metadata["gcn_metadata.json"]["graph_cache_file_sha256"],
        ),
        (
            "MC Dropout molecules",
            paths["molecules"],
            metadata["mc_dropout_metadata.json"]["molecule_file_sha256"],
        ),
        (
            "MC Dropout fingerprints",
            paths["fingerprints"],
            metadata["mc_dropout_metadata.json"]["fingerprint_file_sha256"],
        ),
        (
            "MC Dropout assignments",
            paths["assignments"],
            metadata["mc_dropout_metadata.json"]["assignment_file_sha256"],
        ),
        (
            "MC Dropout graph cache",
            paths["graphs"],
            metadata["mc_dropout_metadata.json"]["graph_cache_file_sha256"],
        ),
        (
            "MC Dropout predictions",
            paths["predictions"],
            metadata["mc_dropout_metadata.json"]["predictions_file_sha256"],
        ),
    )
    checks.extend(
        _audit_repo_hash(name, root, relative, str(expected))
        for name, relative, expected in artifact_specs
    )

    results_inputs = {
        "lr_summary_sha256": Path("reports/tables/lr_split_summary.csv"),
        "gcn_summary_sha256": Path("reports/tables/gcn_split_summary.csv"),
        "predictions_sha256": paths["predictions"],
        "risk_endpoint_metrics_sha256": Path(
            "reports/tables/risk_signal_endpoint_metrics.csv"
        ),
        "similarity_bins_sha256": Path(
            "reports/tables/risk_signal_similarity_bins.csv"
        ),
    }
    recorded_inputs = metadata["results_analysis_metadata.json"]["inputs"]
    checks.extend(
        _audit_repo_hash(
            f"Results input {key}", root, relative, str(recorded_inputs[key])
        )
        for key, relative in results_inputs.items()
    )

    checks.append(
        audit_checkpoint_manifest(
            root / "artifacts" / "gcn",
            metadata["mc_dropout_metadata.json"]["checkpoint_manifest"],
            expected_count=int(metadata["gcn_metadata.json"]["n_fits"]),
        )
    )
    failed_checks = [
        str(check["name"]) for check in checks if not bool(check["passed"])
    ]
    report: dict[str, object] = {
        "status": "pass" if not failed_checks else "fail",
        "n_checks": len(checks),
        "failed_checks": failed_checks,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def audit_file_hash(name: str, path: Path, expected_sha256: str) -> AuditResult:
    """Check that an artifact exists and matches its recorded digest."""

    exists = path.is_file()
    actual_sha256 = _sha256(path) if exists else None
    return {
        "name": name,
        "passed": exists and actual_sha256 == expected_sha256,
        "details": {
            "path": str(path),
            "exists": exists,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
        },
    }


def audit_molecule_identity(
    molecules: pd.DataFrame, *, expected_count: int
) -> AuditResult:
    """Check row count and uniqueness after standardisation and deduplication."""

    required = {"mol_id", "standardized_smiles"}
    missing = sorted(required.difference(molecules.columns))
    if missing:
        return {
            "name": "molecule identity",
            "passed": False,
            "details": {"missing_columns": missing},
        }
    details = {
        "expected_count": expected_count,
        "row_count": len(molecules),
        "unique_mol_ids": int(molecules["mol_id"].nunique(dropna=False)),
        "unique_standardized_smiles": int(
            molecules["standardized_smiles"].nunique(dropna=False)
        ),
    }
    return {
        "name": "molecule identity",
        "passed": all(value == expected_count for value in details.values()),
        "details": details,
    }


def audit_split_integrity(assignments: pd.DataFrame) -> AuditResult:
    """Check IDs, partition labels, and group leakage in fixed splits."""

    split_columns = {
        "random": "random_split",
        "scaffold": "scaffold_split",
        "similarity_cluster": "similarity_cluster_split",
    }
    required = {
        "mol_id",
        "scaffold",
        "cluster_id",
        *split_columns.values(),
    }
    missing = sorted(required.difference(assignments.columns))
    if missing:
        return {
            "name": "split integrity",
            "passed": False,
            "details": {"missing_columns": missing},
        }

    partition_sets = {
        strategy: sorted(set(assignments[column]))
        for strategy, column in split_columns.items()
    }
    scaffold_overlap = _group_overlap_count(
        assignments, "scaffold", "scaffold_split"
    )
    cluster_overlap = _group_overlap_count(
        assignments, "cluster_id", "similarity_cluster_split"
    )
    details = {
        "row_count": len(assignments),
        "unique_mol_ids": int(assignments["mol_id"].nunique(dropna=False)),
        "partition_sets": partition_sets,
        "scaffold_overlap_count": scaffold_overlap,
        "cluster_overlap_count": cluster_overlap,
    }
    return {
        "name": "split integrity",
        "passed": (
            details["row_count"] == details["unique_mol_ids"]
            and all(set(values) == PARTITIONS for values in partition_sets.values())
            and scaffold_overlap == 0
            and cluster_overlap == 0
        ),
        "details": details,
    }


def audit_checkpoint_manifest(
    checkpoint_dir: Path,
    manifest: Sequence[Mapping[str, object]],
    *,
    expected_count: int,
) -> AuditResult:
    """Check every recorded strategy-seed checkpoint and digest."""

    missing: list[str] = []
    hash_mismatches: list[str] = []
    for item in manifest:
        relative = Path(str(item["strategy"])) / f"seed_{int(item['seed'])}.pt"
        path = checkpoint_dir / relative
        if not path.is_file():
            missing.append(str(relative))
        elif _sha256(path) != str(item["sha256"]):
            hash_mismatches.append(str(relative))
    details = {
        "expected_count": expected_count,
        "manifest_count": len(manifest),
        "missing": missing,
        "hash_mismatches": hash_mismatches,
    }
    return {
        "name": "checkpoint manifest",
        "passed": (
            len(manifest) == expected_count
            and not missing
            and not hash_mismatches
        ),
        "details": details,
    }


def _group_overlap_count(
    assignments: pd.DataFrame, group_column: str, split_column: str
) -> int:
    return int(
        (
            assignments.groupby(group_column, dropna=False)[split_column]
            .nunique()
            .gt(1)
        ).sum()
    )


def _audit_repo_hash(
    name: str, root: Path, relative_path: Path, expected_sha256: str
) -> AuditResult:
    result = audit_file_hash(name, root / relative_path, expected_sha256)
    details = dict(result["details"])
    details["path"] = str(relative_path)
    result["details"] = details
    return result


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return payload


def _nested_value(
    values: Mapping[str, object], dotted_key: str
) -> object:
    current: object = values
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
