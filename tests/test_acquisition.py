from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from tox21_risk import acquisition
from tox21_risk.acquisition import download_tox21, inspect_tox21_archive, sha256_file
from tox21_risk.config import load_data_config


def _small_config():
    return replace(load_data_config(Path("configs/data.yaml")), expected_raw_rows=2)


def _write_archive(path: Path, *, omit_task: str | None = None) -> None:
    config = _small_config()
    data = {task: [0.0, 1.0] for task in config.tasks if task != omit_task}
    data[config.smiles_column] = ["CCO", "CCN"]
    pd.DataFrame(data).to_csv(path, index=False, compression="gzip")


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"tox21\n")

    assert sha256_file(path) == sha256_file(path)
    assert sha256_file(path) == "caddca9c64ca69f27c96f26c50d8a9f13f9d4074fdff24125d122c81a5d04c07"


def test_inspect_tox21_archive_validates_schema_and_rows(tmp_path: Path) -> None:
    archive = tmp_path / "tox21.csv.gz"
    _write_archive(archive)

    manifest = inspect_tox21_archive(archive, _small_config())

    assert manifest.row_count == 2
    assert manifest.byte_count == archive.stat().st_size
    assert manifest.columns[-1] == "smiles"
    assert manifest.task_names == _small_config().tasks
    assert manifest.sha256 == sha256_file(archive)


def test_inspect_tox21_archive_rejects_missing_task_column(tmp_path: Path) -> None:
    archive = tmp_path / "tox21.csv.gz"
    missing_task = _small_config().tasks[-1]
    _write_archive(archive, omit_task=missing_task)

    with pytest.raises(ValueError, match=missing_task):
        inspect_tox21_archive(archive, _small_config())


def test_inspect_tox21_archive_rejects_unexpected_row_count(tmp_path: Path) -> None:
    archive = tmp_path / "tox21.csv.gz"
    _write_archive(archive)
    wrong_count = replace(_small_config(), expected_raw_rows=3)

    with pytest.raises(ValueError, match="expected 3 rows, found 2"):
        inspect_tox21_archive(archive, wrong_count)


def test_download_cleans_partial_file_when_retrieval_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "tox21.csv.gz"
    manifest_path = tmp_path / "manifest.json"

    def broken_retrieval(source_url: str, output_path: Path) -> None:
        output_path.write_bytes(b"partial")
        raise OSError("connection dropped")

    monkeypatch.setattr(acquisition, "_retrieve_url", broken_retrieval)

    with pytest.raises(OSError, match="connection dropped"):
        download_tox21(destination, manifest_path, _small_config())

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()
    assert not manifest_path.exists()


def test_download_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "tox21.csv.gz"
    destination.write_bytes(b"existing")
    manifest_path = tmp_path / "manifest.json"

    def unexpected_retrieval(source_url: str, output_path: Path) -> None:
        raise AssertionError("retrieval should not start")

    monkeypatch.setattr(acquisition, "_retrieve_url", unexpected_retrieval)

    with pytest.raises(FileExistsError, match="force=True"):
        download_tox21(destination, manifest_path, _small_config())


def test_download_force_replaces_atomically_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "tox21.csv.gz"
    destination.write_bytes(b"old")
    manifest_path = tmp_path / "metadata" / "source.json"
    source_archive = tmp_path / "source.csv.gz"
    _write_archive(source_archive)

    def local_retrieval(source_url: str, output_path: Path) -> None:
        output_path.write_bytes(source_archive.read_bytes())

    monkeypatch.setattr(acquisition, "_retrieve_url", local_retrieval)

    manifest = download_tox21(
        destination, manifest_path, _small_config(), force=True
    )

    assert sha256_file(destination) == sha256_file(source_archive)
    assert manifest_path.exists()
    assert manifest.row_count == 2
    assert not destination.with_name(destination.name + ".part").exists()
