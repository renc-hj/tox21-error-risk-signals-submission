"""Atomic acquisition and provenance recording for the Tox21 source data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

import pandas as pd

from tox21_risk.config import DataConfig


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Provenance and schema information for one downloaded source archive."""

    source_url: str
    retrieved_at_utc: str
    sha256: str
    byte_count: int
    row_count: int
    columns: tuple[str, ...]
    task_names: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_tox21_archive(path: Path, config: DataConfig) -> SourceManifest:
    """Validate a gzip CSV against the configured Tox21 schema."""

    try:
        frame = pd.read_csv(path, compression="gzip")
    except Exception as exc:
        raise ValueError(f"unable to read Tox21 gzip CSV: {path}") from exc

    if len(frame) != config.expected_raw_rows:
        raise ValueError(
            f"expected {config.expected_raw_rows} rows, found {len(frame)}"
        )

    required_columns = (config.smiles_column, *config.tasks)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Tox21 archive is missing required columns: {missing}")

    return SourceManifest(
        source_url=config.source_url,
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        sha256=sha256_file(path),
        byte_count=path.stat().st_size,
        row_count=len(frame),
        columns=tuple(str(column) for column in frame.columns),
        task_names=config.tasks,
    )


def download_tox21(
    destination: Path,
    manifest_path: Path,
    config: DataConfig,
    force: bool = False,
) -> SourceManifest:
    """Download, validate, and atomically publish the Tox21 archive."""

    if destination.exists() and not force:
        raise FileExistsError(
            f"destination already exists: {destination}; pass force=True to replace it"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + ".part")
    manifest_partial_path = manifest_path.with_name(manifest_path.name + ".part")

    partial_path.unlink(missing_ok=True)
    manifest_partial_path.unlink(missing_ok=True)
    try:
        _retrieve_url(config.source_url, partial_path)
        manifest = inspect_tox21_archive(partial_path, config)
        partial_path.replace(destination)
        _write_manifest(manifest, manifest_partial_path)
        manifest_partial_path.replace(manifest_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        manifest_partial_path.unlink(missing_ok=True)
        raise

    return manifest


def _retrieve_url(source_url: str, output_path: Path) -> None:
    with urlopen(source_url, timeout=120) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def _write_manifest(manifest: SourceManifest, path: Path) -> None:
    path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
