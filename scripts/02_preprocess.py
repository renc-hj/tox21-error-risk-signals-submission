"""Standardize, deduplicate, and fingerprint the downloaded Tox21 data."""

from __future__ import annotations

import argparse
from pathlib import Path

from tox21_risk.config import load_data_config
from tox21_risk.workflow import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/tox21.csv.gz"))
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/tox21_molecules.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/processed/preprocessing_audit.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/preprocessing_audit_summary.json"),
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=Path("data/processed/morgan_r2_1024.npy"),
    )
    args = parser.parse_args()

    summary = run_preprocessing(
        raw_path=args.raw,
        processed_path=args.processed,
        audit_path=args.audit,
        summary_path=args.summary,
        fingerprint_path=args.fingerprints,
        config=load_data_config(args.config),
    )
    print(
        f"Preprocessed {summary['raw_rows']} rows into "
        f"{summary['final_molecules']} unique molecules"
    )


if __name__ == "__main__":
    main()
