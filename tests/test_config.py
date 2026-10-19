from pathlib import Path

import pytest
import yaml

from tox21_risk.config import load_data_config


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


def test_data_config_matches_locked_tox21_contract() -> None:
    config = load_data_config(Path("configs/data.yaml"))

    assert config.expected_raw_rows == 7_831
    assert config.smiles_column == "smiles"
    assert config.tasks == LOCKED_TASKS
    assert config.fingerprint_radius == 2
    assert config.fingerprint_bits == 1_024


def test_data_config_rejects_duplicate_endpoint_names(tmp_path: Path) -> None:
    config_data = yaml.safe_load(Path("configs/data.yaml").read_text())
    config_data["tasks"][-1] = config_data["tasks"][0]
    path = tmp_path / "duplicate_tasks.yaml"
    path.write_text(yaml.safe_dump(config_data))

    with pytest.raises(ValueError, match="tasks must match"):
        load_data_config(path)


def test_data_config_rejects_nonstandard_fingerprint_size(tmp_path: Path) -> None:
    config_data = yaml.safe_load(Path("configs/data.yaml").read_text())
    config_data["fingerprint_bits"] = 2_048
    path = tmp_path / "wrong_fingerprint_bits.yaml"
    path.write_text(yaml.safe_dump(config_data))

    with pytest.raises(ValueError, match="fingerprint_bits must be 1024"):
        load_data_config(path)
