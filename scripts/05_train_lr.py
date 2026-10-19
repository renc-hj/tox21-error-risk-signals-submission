"""Train endpoint-specific Logistic Regression models on all fixed splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from tox21_risk.config import load_data_config
from tox21_risk.lr_baseline import load_lr_config
from tox21_risk.lr_workflow import run_lr_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--lr-config", type=Path, default=Path("configs/lr.yaml"))
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
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/lr_predictions.csv"),
    )
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    lr_config = load_lr_config(args.lr_config)
    metadata = run_lr_workflow(
        molecule_path=args.molecules,
        fingerprint_path=args.fingerprints,
        assignment_path=args.assignments,
        metrics_path=args.tables / "lr_endpoint_metrics.csv",
        summary_path=args.tables / "lr_split_summary.csv",
        predictions_path=args.predictions,
        metadata_path=args.tables / "lr_metadata.json",
        tasks=data_config.tasks,
        config=lr_config,
    )
    print(
        "Completed Logistic Regression for "
        f"{metadata['n_endpoints']} endpoints across 3 fixed splits"
    )


if __name__ == "__main__":
    main()
