from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from tox21_risk.pipeline_audit import (
    audit_checkpoint_manifest,
    audit_experiment_contract,
    audit_file_hash,
    audit_molecule_identity,
    audit_split_integrity,
    run_pipeline_audit,
)


def test_audit_file_hash_detects_stale_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("current data\n", encoding="utf-8")

    result = audit_file_hash("prediction hash", artifact, "0" * 64)

    assert result["name"] == "prediction hash"
    assert result["passed"] is False
    assert result["details"]["exists"] is True
    assert result["details"]["expected_sha256"] == "0" * 64
    assert result["details"]["actual_sha256"] == _sha256(artifact)


def test_audit_molecule_identity_detects_duplicate_standardized_smiles() -> None:
    molecules = pd.DataFrame(
        {
            "mol_id": ["mol_1", "mol_2", "mol_3"],
            "standardized_smiles": ["CC", "CC", "CO"],
        }
    )

    result = audit_molecule_identity(molecules, expected_count=3)

    assert result["passed"] is False
    assert result["details"] == {
        "expected_count": 3,
        "row_count": 3,
        "unique_mol_ids": 3,
        "unique_standardized_smiles": 2,
    }


def test_audit_split_integrity_detects_scaffold_leakage() -> None:
    assignments = pd.DataFrame(
        {
            "mol_id": ["a", "b", "c", "d", "e", "f"],
            "scaffold": ["s1", "s1", "s2", "s3", "s4", "s5"],
            "cluster_id": [1, 1, 2, 3, 4, 5],
            "random_split": [
                "train",
                "validation",
                "test",
                "train",
                "validation",
                "test",
            ],
            "scaffold_split": [
                "train",
                "test",
                "validation",
                "train",
                "validation",
                "test",
            ],
            "similarity_cluster_split": [
                "train",
                "train",
                "validation",
                "test",
                "validation",
                "test",
            ],
        }
    )

    result = audit_split_integrity(assignments)

    assert result["passed"] is False
    assert result["details"]["scaffold_overlap_count"] == 1
    assert result["details"]["cluster_overlap_count"] == 0


def test_audit_checkpoint_manifest_detects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "random" / "seed_2026.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"checkpoint")
    manifest = [
        {
            "strategy": "random",
            "seed": 2026,
            "sha256": _sha256(existing),
        },
        {
            "strategy": "scaffold",
            "seed": 2026,
            "sha256": "0" * 64,
        },
    ]

    result = audit_checkpoint_manifest(tmp_path, manifest, expected_count=2)

    assert result["passed"] is False
    assert result["details"]["manifest_count"] == 2
    assert result["details"]["missing"] == ["scaffold/seed_2026.pt"]
    assert result["details"]["hash_mismatches"] == []


def test_audit_experiment_contract_detects_wrong_mc_pass_count() -> None:
    metadata = _locked_contract_metadata()
    metadata["mc_dropout"]["n_passes"] = 49

    result = audit_experiment_contract(metadata)

    assert result["passed"] is False
    assert result["details"]["mismatches"] == {
        "mc_dropout.n_passes": {"expected": 50, "actual": 49}
    }


def test_run_pipeline_audit_writes_report_and_detects_stale_predictions(
    tmp_path: Path,
) -> None:
    _write_consistent_audit_bundle(tmp_path)
    output = tmp_path / "reports" / "tables" / "pipeline_audit.json"

    passed = run_pipeline_audit(tmp_path, output)

    assert passed["status"] == "pass"
    assert passed["failed_checks"] == []
    assert any(
        check["name"] == "experiment contract" for check in passed["checks"]
    )
    assert json.loads(output.read_text(encoding="utf-8")) == passed

    predictions = tmp_path / "data" / "processed" / "mc_dropout_predictions.csv"
    predictions.write_text("stale predictions\n", encoding="utf-8")
    failed = run_pipeline_audit(tmp_path, output)

    assert failed["status"] == "fail"
    assert "MC Dropout predictions" in failed["failed_checks"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_consistent_audit_bundle(root: Path) -> None:
    processed = root / "data" / "processed"
    tables = root / "reports" / "tables"
    checkpoint_root = root / "artifacts" / "gcn"
    processed.mkdir(parents=True)
    tables.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True)

    molecule_path = processed / "tox21_molecules.csv"
    pd.DataFrame(
        {
            "mol_id": ["a", "b", "c"],
            "standardized_smiles": ["C", "CC", "CCC"],
        }
    ).to_csv(molecule_path, index=False)
    assignment_path = processed / "split_assignments.csv"
    pd.DataFrame(
        {
            "mol_id": ["a", "b", "c"],
            "scaffold": ["s1", "s2", "s3"],
            "cluster_id": [1, 2, 3],
            "random_split": ["train", "validation", "test"],
            "scaffold_split": ["train", "validation", "test"],
            "similarity_cluster_split": ["train", "validation", "test"],
        }
    ).to_csv(assignment_path, index=False)

    fingerprint_path = processed / "morgan_r2_1024.npy"
    fingerprint_path.write_bytes(b"fingerprints")
    graph_path = processed / "tox21_graphs.pt"
    graph_path.write_bytes(b"graphs")
    prediction_path = processed / "mc_dropout_predictions.csv"
    prediction_path.write_text("predictions\n", encoding="utf-8")
    checkpoint_manifest = []
    for strategy in ("random", "scaffold", "similarity_cluster"):
        for seed in (2026, 2027, 2028):
            checkpoint_path = checkpoint_root / strategy / f"seed_{seed}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(f"{strategy}-{seed}".encode())
            checkpoint_manifest.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "sha256": _sha256(checkpoint_path),
                }
            )

    analysis_inputs = {
        "lr_summary_sha256": tables / "lr_split_summary.csv",
        "gcn_summary_sha256": tables / "gcn_split_summary.csv",
        "risk_endpoint_metrics_sha256": tables
        / "risk_signal_endpoint_metrics.csv",
        "similarity_bins_sha256": tables / "risk_signal_similarity_bins.csv",
    }
    for path in analysis_inputs.values():
        path.write_text(path.name + "\n", encoding="utf-8")

    common = {
        "molecule_file_sha256": _sha256(molecule_path),
        "fingerprint_file_sha256": _sha256(fingerprint_path),
        "assignment_file_sha256": _sha256(assignment_path),
        "graph_cache_file_sha256": _sha256(graph_path),
    }
    contract = _locked_contract_metadata()
    metadata = {
        "split_metadata.json": {
            **contract["split"],
            "n_molecules": 3,
            "molecule_file_sha256": common["molecule_file_sha256"],
            "fingerprint_file_sha256": common["fingerprint_file_sha256"],
        },
        "lr_metadata.json": {
            **contract["lr"],
            "n_molecules": 3,
            **{
                key: common[key]
                for key in (
                    "molecule_file_sha256",
                    "fingerprint_file_sha256",
                    "assignment_file_sha256",
                )
            },
        },
        "gcn_metadata.json": {
            **contract["gcn"],
            "n_molecules": 3,
            **{
                key: common[key]
                for key in (
                    "molecule_file_sha256",
                    "assignment_file_sha256",
                    "graph_cache_file_sha256",
                )
            },
        },
        "mc_dropout_metadata.json": {
            **contract["mc_dropout"],
            "n_molecules": 3,
            **common,
            "predictions_file_sha256": _sha256(prediction_path),
            "checkpoint_manifest": checkpoint_manifest,
        },
        "results_analysis_metadata.json": {
            **contract["results"],
            "inputs": {
                **{
                    key: _sha256(path) for key, path in analysis_inputs.items()
                },
                "predictions_sha256": _sha256(prediction_path),
            }
        },
    }
    for filename, payload in metadata.items():
        (tables / filename).write_text(
            json.dumps(payload), encoding="utf-8"
        )


def _locked_contract_metadata() -> dict[str, dict[str, object]]:
    return {
        "split": {
            "fingerprint_radius": 2,
            "fingerprint_bits": 1024,
            "similarity_threshold": 0.6,
            "distance_cutoff": 0.4,
        },
        "lr": {
            "n_endpoints": 12,
            "fingerprint_radius": 2,
            "fingerprint_bits": 1024,
            "c_values": [0.01, 0.1, 1.0, 10.0],
            "selection_metric": "auroc",
            "class_weight": "balanced",
        },
        "gcn": {
            "n_endpoints": 12,
            "n_fits": 9,
            "atom_feature_dim": 166,
            "bond_features": False,
            "selection_metric": "validation_macro_endpoint_bce",
            "config": {
                "seeds": [2026, 2027, 2028],
                "hidden_channels": 128,
                "num_layers": 3,
                "dropout": 0.2,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "max_epochs": 200,
                "patience": 20,
                "device": "cpu",
            },
        },
        "mc_dropout": {
            "n_endpoints": 12,
            "n_fits": 9,
            "n_passes": 50,
            "classification_threshold": 0.5,
            "signals": [
                "structural_distance",
                "single_pass_uncertainty",
                "mc_dropout_mi",
            ],
        },
        "results": {
            "primary_partition": "test",
            "minimum_observed_labels_per_paired_cell": 20,
            "paired_difference_aggregation": (
                "mean_within_endpoint_then_macro_across_endpoints"
            ),
            "paired_difference_ci": {
                "method": "endpoint_cluster_percentile_bootstrap",
                "confidence_level": 0.95,
                "n_resamples": 10_000,
                "seed": 2026,
            },
        },
    }
