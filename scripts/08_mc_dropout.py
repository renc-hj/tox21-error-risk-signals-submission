"""Run 50-pass MC Dropout and compare three endpoint error-risk signals."""

from __future__ import annotations

import argparse
from pathlib import Path

from tox21_risk.config import load_data_config
from tox21_risk.gcn_model import load_gcn_config
from tox21_risk.uncertainty_workflow import run_uncertainty_workflow


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
        "--fingerprints",
        type=Path,
        default=Path("data/processed/morgan_r2_1024.npy"),
    )
    parser.add_argument(
        "--graphs", type=Path, default=Path("data/processed/tox21_graphs.pt")
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/processed/split_assignments.csv"),
    )
    parser.add_argument("--checkpoints", type=Path, default=Path("artifacts/gcn"))
    parser.add_argument("--tables", type=Path, default=Path("reports/tables"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/mc_dropout_predictions.csv"),
    )
    parser.add_argument("--passes", type=int, default=50)
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    gcn_config = load_gcn_config(args.gcn_config)
    metadata = run_uncertainty_workflow(
        molecule_path=args.molecules,
        fingerprint_path=args.fingerprints,
        graph_cache_path=args.graphs,
        assignment_path=args.assignments,
        checkpoint_dir=args.checkpoints,
        predictions_path=args.predictions,
        endpoint_metrics_path=args.tables / "risk_signal_endpoint_metrics.csv",
        summary_path=args.tables / "risk_signal_summary.csv",
        similarity_bins_path=args.tables / "risk_signal_similarity_bins.csv",
        metadata_path=args.tables / "mc_dropout_metadata.json",
        tasks=data_config.tasks,
        config=gcn_config,
        n_passes=args.passes,
    )
    print(
        f"Completed {metadata['n_fits']} checkpoints with "
        f"{metadata['n_passes']} MC passes",
        flush=True,
    )


if __name__ == "__main__":
    main()
