"""Train the nine fixed Tox21 GCN runs and write aggregate results."""

from __future__ import annotations

import argparse
from pathlib import Path

from tox21_risk.config import load_data_config
from tox21_risk.gcn_model import load_gcn_config
from tox21_risk.gcn_workflow import run_gcn_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--gcn-config", type=Path, default=Path("configs/gcn.yaml"))
    parser.add_argument(
        "--molecules",
        type=Path,
        default=Path("data/processed/tox21_molecules.csv"),
    )
    parser.add_argument(
        "--graphs", type=Path, default=Path("data/processed/tox21_graphs.pt")
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
        default=Path("data/processed/gcn_predictions.csv"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/processed/gcn_training_history.csv"),
    )
    parser.add_argument(
        "--checkpoints", type=Path, default=Path("artifacts/gcn")
    )
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    gcn_config = load_gcn_config(args.gcn_config)
    metadata = run_gcn_workflow(
        molecule_path=args.molecules,
        graph_cache_path=args.graphs,
        assignment_path=args.assignments,
        endpoint_metrics_path=args.tables / "gcn_endpoint_metrics.csv",
        split_summary_path=args.tables / "gcn_split_summary.csv",
        training_summary_path=args.tables / "gcn_training_summary.csv",
        metadata_path=args.tables / "gcn_metadata.json",
        predictions_path=args.predictions,
        history_path=args.history,
        checkpoint_dir=args.checkpoints,
        tasks=data_config.tasks,
        config=gcn_config,
    )
    print(f"Completed {metadata['n_fits']} GCN fits", flush=True)


if __name__ == "__main__":
    main()
