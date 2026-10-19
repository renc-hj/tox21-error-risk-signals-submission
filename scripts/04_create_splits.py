"""Create fixed random, scaffold, and similarity-cluster Tox21 splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from tox21_risk.config import load_data_config
from tox21_risk.split_workflow import run_split_workflow
from tox21_risk.splitting import load_split_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--split-config", type=Path, default=Path("configs/splits.yaml")
    )
    parser.add_argument(
        "--molecules",
        type=Path,
        default=Path("data/processed/tox21_molecules.csv"),
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=Path("data/processed/morgan_r2_1024.npy"),
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/processed/split_assignments.csv"),
    )
    parser.add_argument("--tables", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    split_config = load_split_config(args.split_config)
    metadata = run_split_workflow(
        molecule_path=args.molecules,
        fingerprint_path=args.fingerprints,
        assignments_path=args.assignments,
        summary_path=args.tables / "split_summary.csv",
        prevalence_path=args.tables / "split_endpoint_prevalence.csv",
        similarity_path=args.tables / "split_similarity_summary.csv",
        metadata_path=args.tables / "split_metadata.json",
        tasks=data_config.tasks,
        config=split_config,
    )
    print(
        "Created three fixed splits for "
        f"{metadata['n_molecules']:,} molecules: {args.assignments}"
    )


if __name__ == "__main__":
    main()
