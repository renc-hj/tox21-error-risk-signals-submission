"""Load and validate the locked Tox21 data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


LOCKED_TASKS = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Immutable configuration shared by all data-pipeline stages."""

    source_url: str
    expected_raw_rows: int
    smiles_column: str
    tasks: tuple[str, ...]
    fingerprint_radius: int
    fingerprint_bits: int


def load_data_config(path: Path) -> DataConfig:
    """Load ``path`` and reject deviations from the study's data contract."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")

    config = DataConfig(
        source_url=_require_string(raw, "source_url"),
        expected_raw_rows=_require_integer(raw, "expected_raw_rows"),
        smiles_column=_require_string(raw, "smiles_column"),
        tasks=_require_tasks(raw.get("tasks")),
        fingerprint_radius=_require_integer(raw, "fingerprint_radius"),
        fingerprint_bits=_require_integer(raw, "fingerprint_bits"),
    )
    _validate_locked_contract(config)
    return config


def _require_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_integer(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_tasks(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(task, str) for task in value):
        raise ValueError("tasks must be a list of endpoint names")
    return tuple(value)


def _validate_locked_contract(config: DataConfig) -> None:
    if config.expected_raw_rows != 7_831:
        raise ValueError("expected_raw_rows must be 7831")
    if config.smiles_column != "smiles":
        raise ValueError("smiles_column must be 'smiles'")
    if config.tasks != LOCKED_TASKS:
        raise ValueError("tasks must match the 12 locked Tox21 endpoints in order")
    if config.fingerprint_radius != 2:
        raise ValueError("fingerprint_radius must be 2")
    if config.fingerprint_bits != 1_024:
        raise ValueError("fingerprint_bits must be 1024")
