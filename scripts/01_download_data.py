"""Download and verify the configured Tox21 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tox21_risk.acquisition import download_tox21
from tox21_risk.config import load_data_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--destination", type=Path, default=Path("data/raw/tox21.csv.gz")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/tox21_source.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_data_config(args.config)
    manifest = download_tox21(
        args.destination, args.manifest, config, force=args.force
    )
    print(
        f"Verified {manifest.row_count} rows; "
        f"SHA-256 {manifest.sha256}; manifest {args.manifest}"
    )


if __name__ == "__main__":
    main()
