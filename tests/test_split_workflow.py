import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tox21_risk.fingerprints import morgan_fingerprint_matrix
from tox21_risk.split_workflow import run_split_workflow
from tox21_risk.splitting import SplitConfig


SMILES = [
    "CCO",
    "CCN",
    "CCCl",
    "c1ccccc1",
    "Oc1ccccc1",
    "C1CCCCC1",
    "N1CCCCC1",
    "c1ccncc1",
    "C1CCCC1",
    "c1ccccc1C",
]


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    molecules = pd.DataFrame(
        {
            "mol_id": [f"mol_{index:06d}" for index in range(len(SMILES))],
            "standardized_smiles": SMILES,
            "task_a": [1, 0, np.nan, 1, 0, 0, 1, np.nan, 0, 1],
            "task_b": [0, 0, 1, 0, np.nan, 1, 0, 0, 1, 0],
        }
    )
    fingerprints = morgan_fingerprint_matrix(SMILES)

    molecule_path = tmp_path / "molecules.csv"
    fingerprint_path = tmp_path / "fingerprints.npy"
    molecules.to_csv(molecule_path, index=False)
    np.save(fingerprint_path, fingerprints, allow_pickle=False)
    return molecule_path, fingerprint_path


def _output_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "assignments_path": tmp_path / "processed" / "split_assignments.csv",
        "summary_path": tmp_path / "reports" / "split_summary.csv",
        "prevalence_path": tmp_path / "reports" / "split_endpoint_prevalence.csv",
        "similarity_path": tmp_path / "reports" / "split_similarity_summary.csv",
        "metadata_path": tmp_path / "reports" / "split_metadata.json",
    }


def test_split_workflow_writes_verified_assignments_and_reports(
    tmp_path: Path,
) -> None:
    molecule_path, fingerprint_path = _inputs(tmp_path)
    paths = _output_paths(tmp_path)

    run_split_workflow(
        molecule_path=molecule_path,
        fingerprint_path=fingerprint_path,
        tasks=("task_a", "task_b"),
        config=SplitConfig(),
        **paths,
    )

    assignments = pd.read_csv(paths["assignments_path"])
    assert len(assignments) == 10
    assert assignments["mol_id"].is_unique
    assert {
        "random_split",
        "scaffold_split",
        "similarity_cluster_split",
        "scaffold",
        "cluster_id",
    }.issubset(assignments.columns)
    assert assignments.groupby("scaffold")["scaffold_split"].nunique().max() == 1
    assert (
        assignments.groupby("cluster_id")["similarity_cluster_split"]
        .nunique()
        .max()
        == 1
    )

    summary = pd.read_csv(paths["summary_path"])
    assert set(summary["strategy"]) == {
        "random",
        "scaffold",
        "similarity_cluster",
    }
    assert (summary["molecule_overlap_count"] == 0).all()
    assert summary.loc[
        summary["strategy"] == "scaffold", "scaffold_overlap_count"
    ].iloc[0] == 0
    assert summary.loc[
        summary["strategy"] == "similarity_cluster", "cluster_overlap_count"
    ].iloc[0] == 0

    prevalence = pd.read_csv(paths["prevalence_path"])
    task_a = prevalence[prevalence["endpoint"] == "task_a"]
    assert task_a.groupby("strategy")["missing"].sum().eq(2).all()

    similarity = pd.read_csv(paths["similarity_path"])
    assert similarity["test_max_tanimoto_mean"].between(0, 1).all()
    assert similarity["test_max_tanimoto_median"].between(0, 1).all()

    metadata = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["n_molecules"] == 10
    assert metadata["fingerprint_radius"] == 2
    assert metadata["fingerprint_bits"] == 1_024
    assert metadata["similarity_threshold"] == 0.6
    assert metadata["distance_cutoff"] == 0.4


def test_split_workflow_rejects_duplicate_molecule_ids(tmp_path: Path) -> None:
    molecule_path, fingerprint_path = _inputs(tmp_path)
    molecules = pd.read_csv(molecule_path)
    molecules.loc[1, "mol_id"] = molecules.loc[0, "mol_id"]
    molecules.to_csv(molecule_path, index=False)

    with pytest.raises(ValueError, match="mol_id"):
        run_split_workflow(
            molecule_path=molecule_path,
            fingerprint_path=fingerprint_path,
            tasks=("task_a", "task_b"),
            config=SplitConfig(),
            **_output_paths(tmp_path),
        )


def test_split_workflow_rejects_fingerprint_row_mismatch(tmp_path: Path) -> None:
    molecule_path, fingerprint_path = _inputs(tmp_path)
    fingerprints = np.load(fingerprint_path, allow_pickle=False)
    np.save(fingerprint_path, fingerprints[:-1], allow_pickle=False)

    with pytest.raises(ValueError, match="row count"):
        run_split_workflow(
            molecule_path=molecule_path,
            fingerprint_path=fingerprint_path,
            tasks=("task_a", "task_b"),
            config=SplitConfig(),
            **_output_paths(tmp_path),
        )


def test_split_workflow_rejects_misaligned_fingerprint_rows(tmp_path: Path) -> None:
    molecule_path, fingerprint_path = _inputs(tmp_path)
    fingerprints = np.load(fingerprint_path, allow_pickle=False)
    fingerprints[[0, 3]] = fingerprints[[3, 0]]
    np.save(fingerprint_path, fingerprints, allow_pickle=False)

    with pytest.raises(ValueError, match="aligned"):
        run_split_workflow(
            molecule_path=molecule_path,
            fingerprint_path=fingerprint_path,
            tasks=("task_a", "task_b"),
            config=SplitConfig(),
            **_output_paths(tmp_path),
        )
