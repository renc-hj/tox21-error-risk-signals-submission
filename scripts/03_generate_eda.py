"""Generate aggregate Tox21 EDA tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tox21_risk.config import load_data_config
from tox21_risk.eda import generate_eda_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
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
    parser.add_argument("--tables", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    config = load_data_config(args.config)
    outputs = generate_eda_outputs(
        frame=pd.read_csv(args.processed),
        audit=pd.read_csv(args.audit),
        tasks=config.tasks,
        table_dir=args.tables,
        figure_dir=args.figures,
        seed=args.seed,
    )
    print(f"Generated {len(outputs)} EDA artifacts")


if __name__ == "__main__":
    main()
