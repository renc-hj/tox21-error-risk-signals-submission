from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tox21_risk.config import load_data_config
from tox21_risk.workflow import run_preprocessing


FIXTURE = Path("tests/fixtures/tox21_mini.csv")


def _run_once(tmp_path: Path, name: str):
    raw = pd.read_csv(FIXTURE)
    config = replace(
        load_data_config(Path("configs/data.yaml")), expected_raw_rows=len(raw)
    )
    output_dir = tmp_path / name
    paths = {
        "processed_path": output_dir / "molecules.csv",
        "audit_path": output_dir / "audit.csv",
        "summary_path": output_dir / "summary.json",
        "fingerprint_path": output_dir / "fingerprints.npy",
    }
    summary = run_preprocessing(FIXTURE, config=config, **paths)
    return summary, paths


def test_preprocessing_is_repeatable_and_preserves_missing_labels(
    tmp_path: Path,
) -> None:
    first_summary, first = _run_once(tmp_path, "first")
    second_summary, second = _run_once(tmp_path, "second")

    assert first_summary == second_summary
    assert first["processed_path"].read_bytes() == second["processed_path"].read_bytes()
    assert first["audit_path"].read_bytes() == second["audit_path"].read_bytes()
    assert first["summary_path"].read_bytes() == second["summary_path"].read_bytes()
    np.testing.assert_array_equal(
        np.load(first["fingerprint_path"]), np.load(second["fingerprint_path"])
    )

    molecules = pd.read_csv(first["processed_path"])
    assert molecules["SR-p53"].isna().all()
    conflicted = molecules.loc[molecules["standardized_smiles"] == "CCN"]
    assert conflicted["NR-AR"].isna().all()
    assert molecules.loc[
        molecules["standardized_smiles"] == "CCO", "NR-AR-LBD"
    ].item() == 0.0


def test_preprocessing_summary_and_audit_reconcile_raw_rows(tmp_path: Path) -> None:
    summary, paths = _run_once(tmp_path, "run")
    saved_summary = json.loads(paths["summary_path"].read_text())
    audit = pd.read_csv(paths["audit_path"])
    fingerprints = np.load(paths["fingerprint_path"])

    assert summary == saved_summary
    assert summary["raw_rows"] == 8
    assert summary["standardization_failed_rows"] == 2
    assert summary["duplicate_rows_removed"] == 3
    assert summary["endpoint_conflicts"] == 1
    assert summary["final_molecules"] == 3
    assert summary["raw_rows"] == (
        summary["standardization_failed_rows"]
        + summary["duplicate_rows_removed"]
        + summary["final_molecules"]
    )
    assert len(audit) == summary["raw_rows"]
    assert (audit["status"] == "failed").sum() == 2
    assert fingerprints.shape == (summary["final_molecules"], 1_024)
