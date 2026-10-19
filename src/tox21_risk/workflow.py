"""Orchestrate standardization, deduplication, and fingerprint generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tox21_risk.config import DataConfig
from tox21_risk.deduplication import resolve_multitask_duplicates
from tox21_risk.fingerprints import morgan_fingerprint_matrix
from tox21_risk.standardization import standardize_smiles


def run_preprocessing(
    raw_path: Path,
    processed_path: Path,
    audit_path: Path,
    summary_path: Path,
    fingerprint_path: Path,
    config: DataConfig,
) -> dict[str, object]:
    """Run the deterministic preprocessing workflow and save local artifacts."""

    raw = pd.read_csv(raw_path, compression="infer")
    _validate_raw_frame(raw, config)

    audit_records: list[dict[str, object]] = []
    successful_records: list[dict[str, object]] = []
    for source_index, row in raw.iterrows():
        source_smiles = row[config.smiles_column]
        normalized_source = None if pd.isna(source_smiles) else str(source_smiles)
        result = standardize_smiles(normalized_source)
        audit_records.append(
            {
                "source_row_number": int(source_index) + 1,
                "original_smiles": result.original_smiles,
                "standardized_smiles": result.standardized_smiles,
                "inchikey": result.inchikey,
                "status": result.status,
                "failure_reason": result.failure_reason,
                "transformation_log": json.dumps(list(result.transformation_log)),
            }
        )
        if result.status == "success":
            successful_records.append(
                {
                    "standardized_smiles": result.standardized_smiles,
                    "inchikey": result.inchikey,
                    **{task: row[task] for task in config.tasks},
                }
            )

    if not successful_records:
        raise ValueError("no molecules remained after standardization")

    standardized = pd.DataFrame.from_records(successful_records)
    deduplicated = resolve_multitask_duplicates(standardized, config.tasks)
    molecules = deduplicated.molecules
    fingerprints = morgan_fingerprint_matrix(
        molecules["standardized_smiles"].tolist(),
        radius=config.fingerprint_radius,
        n_bits=config.fingerprint_bits,
    )

    audit = pd.DataFrame.from_records(audit_records).merge(
        deduplicated.audit, on="standardized_smiles", how="left", sort=False
    )
    failed_rows = int((audit["status"] == "failed").sum())
    summary: dict[str, object] = {
        "raw_rows": len(raw),
        "standardization_failed_rows": failed_rows,
        "standardized_rows": len(standardized),
        "duplicate_rows_removed": deduplicated.summary["duplicate_rows_removed"],
        "duplicate_groups": deduplicated.summary["duplicate_groups"],
        "exact_duplicate_rows": deduplicated.summary["exact_duplicate_rows"],
        "complementary_label_merges": deduplicated.summary[
            "complementary_label_merges"
        ],
        "endpoint_conflicts": deduplicated.summary["endpoint_conflicts"],
        "molecules_with_conflicts": deduplicated.summary[
            "molecules_with_conflicts"
        ],
        "final_molecules": len(molecules),
    }

    for path in (processed_path, audit_path, summary_path, fingerprint_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    molecules.to_csv(processed_path, index=False, lineterminator="\n")
    audit.to_csv(audit_path, index=False, lineterminator="\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.save(fingerprint_path, fingerprints, allow_pickle=False)
    return summary


def _validate_raw_frame(frame: pd.DataFrame, config: DataConfig) -> None:
    if len(frame) != config.expected_raw_rows:
        raise ValueError(
            f"expected {config.expected_raw_rows} rows, found {len(frame)}"
        )
    required = {config.smiles_column, *config.tasks}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"raw Tox21 data is missing columns: {', '.join(missing)}")
