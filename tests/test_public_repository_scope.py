from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_public_repository_contains_no_report_files() -> None:
    tracked_files = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()

    assert not any(path == "report" or path.startswith("report/") for path in tracked_files)
    assert not any(path.lower().endswith(".pdf") for path in tracked_files)
