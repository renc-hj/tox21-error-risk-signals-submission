import numpy as np
import pandas as pd
import pandas.testing as pdt

from tox21_risk.deduplication import resolve_multitask_duplicates


TASKS = ("task_a", "task_b", "task_all_missing")


def _duplicate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "standardized_smiles": "CCO",
                "inchikey": "ETHANOL",
                "task_a": 1.0,
                "task_b": np.nan,
                "task_all_missing": np.nan,
            },
            {
                "standardized_smiles": "CCO",
                "inchikey": "ETHANOL",
                "task_a": np.nan,
                "task_b": 0.0,
                "task_all_missing": np.nan,
            },
            {
                "standardized_smiles": "CCO",
                "inchikey": "ETHANOL",
                "task_a": np.nan,
                "task_b": 0.0,
                "task_all_missing": np.nan,
            },
            {
                "standardized_smiles": "CCN",
                "inchikey": "ETHYLAMINE",
                "task_a": 0.0,
                "task_b": 1.0,
                "task_all_missing": np.nan,
            },
            {
                "standardized_smiles": "CCN",
                "inchikey": "ETHYLAMINE",
                "task_a": 1.0,
                "task_b": 1.0,
                "task_all_missing": np.nan,
            },
            {
                "standardized_smiles": "CCC",
                "inchikey": "PROPANE",
                "task_a": np.nan,
                "task_b": np.nan,
                "task_all_missing": np.nan,
            },
        ]
    )


def test_duplicate_resolution_merges_complementary_labels_and_masks_conflict() -> None:
    result = resolve_multitask_duplicates(_duplicate_frame(), TASKS)
    molecules = result.molecules.set_index("standardized_smiles")

    assert molecules.loc["CCO", "task_a"] == 1.0
    assert molecules.loc["CCO", "task_b"] == 0.0
    assert pd.isna(molecules.loc["CCN", "task_a"])
    assert molecules.loc["CCN", "task_b"] == 1.0
    assert molecules["task_all_missing"].isna().all()

    conflict = result.audit.set_index("standardized_smiles")
    assert bool(conflict.loc["CCN", "conflict_task_a"])
    assert not bool(conflict.loc["CCN", "conflict_task_b"])
    assert conflict.loc["CCO", "complementary_label_merge_count"] == 1


def test_duplicate_summary_reconciles_all_input_rows() -> None:
    result = resolve_multitask_duplicates(_duplicate_frame(), TASKS)

    assert result.summary == {
        "input_rows": 6,
        "unique_molecules": 3,
        "duplicate_rows_removed": 3,
        "duplicate_groups": 2,
        "exact_duplicate_rows": 1,
        "complementary_label_merges": 1,
        "endpoint_conflicts": 1,
        "molecules_with_conflicts": 1,
    }
    assert result.summary["input_rows"] == (
        result.summary["unique_molecules"]
        + result.summary["duplicate_rows_removed"]
    )


def test_duplicate_resolution_is_deterministic_and_assigns_stable_ids() -> None:
    first = resolve_multitask_duplicates(_duplicate_frame(), TASKS)
    second = resolve_multitask_duplicates(_duplicate_frame(), TASKS)

    pdt.assert_frame_equal(first.molecules, second.molecules)
    pdt.assert_frame_equal(first.audit, second.audit)
    assert first.molecules["standardized_smiles"].tolist() == ["CCC", "CCN", "CCO"]
    assert first.molecules["mol_id"].tolist() == [
        "mol_000001",
        "mol_000002",
        "mol_000003",
    ]
