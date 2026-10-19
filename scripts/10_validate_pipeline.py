"""Validate saved Tox21 artifacts and write a machine-readable audit report."""

from __future__ import annotations

import argparse
from pathlib import Path

from tox21_risk.pipeline_audit import run_pipeline_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/pipeline_audit.json"),
    )
    args = parser.parse_args()
    output = args.output
    if not output.is_absolute():
        output = args.repo_root / output
    report = run_pipeline_audit(args.repo_root, output)
    print(
        f"Pipeline audit {report['status']}: "
        f"{report['n_checks']} checks, "
        f"{len(report['failed_checks'])} failed",
        flush=True,
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
