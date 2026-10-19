from pathlib import Path

import numpy as np
import pandas as pd

from tox21_risk.eda import build_endpoint_summary, generate_eda_outputs


TASKS = ("task_a", "task_b", "task_all_missing")


def _molecules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mol_id": ["mol_000001", "mol_000002", "mol_000003"],
            "standardized_smiles": ["CCO", "CCN", "c1ccccc1"],
            "inchikey": ["ETHANOL", "ETHYLAMINE", "BENZENE"],
            "source_row_count": [2, 1, 1],
            "task_a": [1.0, 0.0, 1.0],
            "task_b": [0.0, np.nan, 1.0],
            "task_all_missing": [np.nan, np.nan, np.nan],
        }
    )


def _audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_number": [1, 2, 3, 4, 5],
            "standardized_smiles": ["CCO", "CCO", "CCN", "c1ccccc1", np.nan],
            "status": ["success", "success", "success", "success", "failed"],
            "conflict_count": [0.0, 0.0, 1.0, 0.0, np.nan],
        }
    )


def test_endpoint_summary_counts_observed_labels_and_uses_observed_denominator() -> None:
    summary = build_endpoint_summary(_molecules(), TASKS).set_index("endpoint")

    assert summary.loc["task_a"].to_dict() == {
        "positive": 2.0,
        "negative": 1.0,
        "missing": 0.0,
        "observed": 3.0,
        "positive_prevalence": 2 / 3,
    }
    assert summary.loc["task_b", "positive"] == 1
    assert summary.loc["task_b", "negative"] == 1
    assert summary.loc["task_b", "missing"] == 1
    assert summary.loc["task_b", "observed"] == 2
    assert summary.loc["task_b", "positive_prevalence"] == 0.5
    assert summary.loc["task_all_missing", "observed"] == 0
    assert np.isnan(summary.loc["task_all_missing", "positive_prevalence"])


def test_generate_eda_outputs_creates_tables_and_five_nonempty_figures(
    tmp_path: Path,
) -> None:
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"

    outputs = generate_eda_outputs(
        _molecules(), _audit(), TASKS, table_dir, figure_dir, seed=2026
    )

    expected = {
        table_dir / "endpoint_summary.csv",
        table_dir / "dataset_summary.json",
        figure_dir / "label_availability.png",
        figure_dir / "positive_prevalence.png",
        figure_dir / "observed_endpoints_per_molecule.png",
        figure_dir / "atom_count_distribution.png",
        figure_dir / "example_structures.png",
    }
    assert set(outputs) == expected
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)

    endpoint_table = pd.read_csv(table_dir / "endpoint_summary.csv")
    dataset_table = pd.read_json(
        table_dir / "dataset_summary.json", typ="series"
    )
    assert len(endpoint_table) == len(TASKS)
    assert dataset_table["molecule_count"] == 3
    assert dataset_table["standardization_failed_rows"] == 1
