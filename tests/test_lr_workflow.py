import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tox21_risk.fingerprints import morgan_fingerprint_matrix
from tox21_risk.lr_baseline import LRConfig
from tox21_risk.lr_workflow import run_lr_workflow


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
    "O=C=O",
    "CC(=O)O",
]


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    molecule_ids = [f"mol_{index:06d}" for index in range(len(SMILES))]
    molecules = pd.DataFrame(
        {
            "mol_id": molecule_ids,
            "standardized_smiles": SMILES,
            "task_a": [0, 1, 0, 1, np.nan, 1, 0, 1, 0, 1, 0, 1],
            "task_b": [1, 0, 1, 0, 1, np.nan, 1, 0, 1, 0, 1, 0],
        }
    )
    split = ["train"] * 8 + ["validation"] * 2 + ["test"] * 2
    assignments = pd.DataFrame(
        {
            "mol_id": molecule_ids,
            "random_split": split,
            "scaffold_split": split,
            "similarity_cluster_split": split,
        }
    )
    molecule_path = tmp_path / "molecules.csv"
    fingerprint_path = tmp_path / "fingerprints.npy"
    assignment_path = tmp_path / "assignments.csv"
    molecules.to_csv(molecule_path, index=False)
    np.save(
        fingerprint_path,
        morgan_fingerprint_matrix(SMILES),
        allow_pickle=False,
    )
    assignments.to_csv(assignment_path, index=False)
    return molecule_path, fingerprint_path, assignment_path


def _outputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "metrics_path": tmp_path / "reports" / "lr_endpoint_metrics.csv",
        "summary_path": tmp_path / "reports" / "lr_split_summary.csv",
        "predictions_path": tmp_path / "processed" / "lr_predictions.csv",
        "metadata_path": tmp_path / "reports" / "lr_metadata.json",
    }


def test_lr_workflow_trains_every_endpoint_on_all_three_splits(
    tmp_path: Path,
) -> None:
    molecule_path, fingerprint_path, assignment_path = _inputs(tmp_path)
    outputs = _outputs(tmp_path)

    run_lr_workflow(
        molecule_path=molecule_path,
        fingerprint_path=fingerprint_path,
        assignment_path=assignment_path,
        tasks=("task_a", "task_b"),
        config=LRConfig(c_values=(1.0,)),
        **outputs,
    )

    metrics = pd.read_csv(outputs["metrics_path"])
    assert len(metrics) == 3 * 2 * 2
    assert set(metrics["strategy"]) == {
        "random",
        "scaffold",
        "similarity_cluster",
    }
    assert set(metrics["partition"]) == {"validation", "test"}
    assert metrics["endpoint"].nunique() == 2
    assert metrics["c_value"].eq(1.0).all()
    assert metrics.loc[metrics["endpoint"] == "task_a", "n_train"].eq(7).all()
    assert metrics[["auroc", "auprc", "brier"]].notna().all().all()

    summary = pd.read_csv(outputs["summary_path"])
    assert len(summary) == 3 * 2
    assert summary["n_endpoints"].eq(2).all()
    for row in summary.itertuples(index=False):
        expected = metrics.loc[
            (metrics["strategy"] == row.strategy)
            & (metrics["partition"] == row.partition),
            "auroc",
        ].mean()
        assert row.macro_auroc == pytest.approx(expected)

    predictions = pd.read_csv(outputs["predictions_path"])
    assert len(predictions) == 3 * 2 * 4
    assert predictions["probability"].between(0, 1).all()
    assert set(predictions["partition"]) == {"validation", "test"}

    metadata = json.loads(outputs["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["n_molecules"] == 12
    assert metadata["n_endpoints"] == 2
    assert metadata["c_values"] == [1.0]
    assert metadata["missing_label_policy"] == "excluded per endpoint"


def test_lr_workflow_rejects_shuffled_assignment_rows(tmp_path: Path) -> None:
    molecule_path, fingerprint_path, assignment_path = _inputs(tmp_path)
    assignments = pd.read_csv(assignment_path).sample(frac=1, random_state=5)
    assignments.to_csv(assignment_path, index=False)

    with pytest.raises(ValueError, match="same order"):
        run_lr_workflow(
            molecule_path=molecule_path,
            fingerprint_path=fingerprint_path,
            assignment_path=assignment_path,
            tasks=("task_a", "task_b"),
            config=LRConfig(c_values=(1.0,)),
            **_outputs(tmp_path),
        )


def test_lr_workflow_rejects_misaligned_fingerprints(tmp_path: Path) -> None:
    molecule_path, fingerprint_path, assignment_path = _inputs(tmp_path)
    fingerprints = np.load(fingerprint_path, allow_pickle=False)
    fingerprints[[0, 3]] = fingerprints[[3, 0]]
    np.save(fingerprint_path, fingerprints, allow_pickle=False)

    with pytest.raises(ValueError, match="aligned"):
        run_lr_workflow(
            molecule_path=molecule_path,
            fingerprint_path=fingerprint_path,
            assignment_path=assignment_path,
            tasks=("task_a", "task_b"),
            config=LRConfig(c_values=(1.0,)),
            **_outputs(tmp_path),
        )
