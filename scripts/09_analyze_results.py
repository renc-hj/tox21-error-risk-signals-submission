"""Create final report tables, figures, and a concise results summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from tox21_risk.results_analysis import run_results_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/mc_dropout_predictions.csv"),
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/results_summary.md")
    )
    args = parser.parse_args()

    metadata = run_results_analysis(
        lr_summary_path=args.tables / "lr_split_summary.csv",
        gcn_summary_path=args.tables / "gcn_split_summary.csv",
        predictions_path=args.predictions,
        risk_endpoint_metrics_path=args.tables
        / "risk_signal_endpoint_metrics.csv",
        similarity_bins_path=args.tables / "risk_signal_similarity_bins.csv",
        table_dir=args.tables,
        figure_dir=args.figures,
        summary_path=args.summary,
    )
    print(
        f"Created {len(metadata['tables'])} tables and "
        f"{len(metadata['figures'])} figures",
        flush=True,
    )


if __name__ == "__main__":
    main()
