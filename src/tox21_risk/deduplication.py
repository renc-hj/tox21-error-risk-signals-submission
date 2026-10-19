"""Endpoint-wise duplicate resolution that preserves missing Tox21 labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Unique molecules plus molecule-level and aggregate audit records."""

    molecules: pd.DataFrame
    audit: pd.DataFrame
    summary: dict[str, int]


def resolve_multitask_duplicates(
    frame: pd.DataFrame, tasks: Sequence[str]
) -> DeduplicationResult:
    """Merge duplicate identities while resolving each endpoint independently."""

    task_names = tuple(tasks)
    _validate_input(frame, task_names)
    working = frame.copy()
    for task in task_names:
        working[task] = pd.to_numeric(working[task], errors="raise")
        observed_values = set(working[task].dropna().unique().tolist())
        if not observed_values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"{task} contains values other than 0, 1, or missing")

    exact_duplicate_rows = int(
        working.duplicated(
            subset=["standardized_smiles", "inchikey", *task_names], keep="first"
        ).sum()
    )

    molecule_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    for standardized_smiles, group in working.groupby(
        "standardized_smiles", sort=True, dropna=False
    ):
        resolved_labels: dict[str, float] = {}
        conflict_flags: dict[str, bool] = {}
        for task in task_names:
            observed = sorted(set(group[task].dropna().astype(float).tolist()))
            has_conflict = len(observed) > 1
            conflict_flags[task] = has_conflict
            resolved_labels[task] = observed[0] if len(observed) == 1 else np.nan

        resolved_observed = sum(
            not pd.isna(resolved_labels[task]) for task in task_names
        )
        most_complete_source_row = int(group[list(task_names)].notna().sum(axis=1).max())
        complementary_merges = max(0, resolved_observed - most_complete_source_row)
        inchikey = _representative_inchikey(group["inchikey"])
        source_row_count = len(group)
        group_exact_duplicates = int(
            group.duplicated(
                subset=["standardized_smiles", "inchikey", *task_names],
                keep="first",
            ).sum()
        )

        molecule_records.append(
            {
                "standardized_smiles": str(standardized_smiles),
                "inchikey": inchikey,
                "source_row_count": source_row_count,
                **resolved_labels,
            }
        )
        audit_records.append(
            {
                "standardized_smiles": str(standardized_smiles),
                "source_row_count": source_row_count,
                "is_duplicate_group": source_row_count > 1,
                "exact_duplicate_rows": group_exact_duplicates,
                "complementary_label_merge_count": complementary_merges,
                "conflict_count": sum(conflict_flags.values()),
                **{
                    f"conflict_{task}": conflict_flags[task] for task in task_names
                },
            }
        )

    molecules = pd.DataFrame.from_records(molecule_records).sort_values(
        "standardized_smiles", kind="stable"
    )
    molecules = molecules.reset_index(drop=True)
    molecules.insert(
        0,
        "mol_id",
        [f"mol_{index:06d}" for index in range(1, len(molecules) + 1)],
    )
    audit = (
        pd.DataFrame.from_records(audit_records)
        .sort_values("standardized_smiles", kind="stable")
        .reset_index(drop=True)
    )

    duplicate_groups = int((audit["source_row_count"] > 1).sum())
    endpoint_conflicts = int(audit["conflict_count"].sum())
    summary = {
        "input_rows": len(working),
        "unique_molecules": len(molecules),
        "duplicate_rows_removed": len(working) - len(molecules),
        "duplicate_groups": duplicate_groups,
        "exact_duplicate_rows": exact_duplicate_rows,
        "complementary_label_merges": int(
            audit["complementary_label_merge_count"].sum()
        ),
        "endpoint_conflicts": endpoint_conflicts,
        "molecules_with_conflicts": int((audit["conflict_count"] > 0).sum()),
    }
    return DeduplicationResult(molecules=molecules, audit=audit, summary=summary)


def _validate_input(frame: pd.DataFrame, tasks: tuple[str, ...]) -> None:
    required = {"standardized_smiles", "inchikey", *tasks}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"deduplication input is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("deduplication input must contain at least one row")
    if frame["standardized_smiles"].isna().any():
        raise ValueError("standardized_smiles must not be missing")


def _representative_inchikey(values: pd.Series) -> str | None:
    candidates = sorted(
        {str(value) for value in values.dropna().tolist() if str(value).strip()}
    )
    return candidates[0] if candidates else None
