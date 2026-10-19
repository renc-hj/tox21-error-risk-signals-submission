"""Build the local PyTorch Geometric cache for the processed Tox21 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tox21_risk.config import load_data_config
from tox21_risk.graph_data import ATOM_FEATURE_DIM, build_graphs, save_graph_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--molecules",
        type=Path,
        default=Path("data/processed/tox21_molecules.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tox21_graphs.pt"),
    )
    args = parser.parse_args()

    config = load_data_config(args.data_config)
    molecules = pd.read_csv(args.molecules)
    graphs = build_graphs(molecules, config.tasks)
    save_graph_cache(graphs, args.output, args.molecules, config.tasks)
    print(
        f"Built {len(graphs):,} molecular graphs with "
        f"{ATOM_FEATURE_DIM} atom features"
    )


if __name__ == "__main__":
    main()
