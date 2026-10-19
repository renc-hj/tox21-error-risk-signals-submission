"""Configuration and standard multi-task GCN for Tox21."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, global_mean_pool


@dataclass(frozen=True, slots=True)
class GCNConfig:
    """Training and architecture values used by the locked experiment."""

    seeds: tuple[int, ...] = (2026, 2027, 2028)
    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    max_epochs: int = 200
    patience: int = 20
    device: str = "cpu"


def load_gcn_config(path: Path) -> GCNConfig:
    """Load YAML and reject changes to the predeclared experiment."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("GCN configuration must be a YAML mapping")
    config = GCNConfig(
        seeds=_integer_tuple(raw, "seeds"),
        hidden_channels=_integer(raw, "hidden_channels"),
        num_layers=_integer(raw, "num_layers"),
        dropout=_number(raw, "dropout"),
        batch_size=_integer(raw, "batch_size"),
        learning_rate=_number(raw, "learning_rate"),
        weight_decay=_number(raw, "weight_decay"),
        max_epochs=_integer(raw, "max_epochs"),
        patience=_integer(raw, "patience"),
        device=_string(raw, "device"),
    )
    _validate_locked_config(config)
    return config


class Tox21GCN(nn.Module):
    """Three-layer GCN with mean pooling and a multi-task logit head."""

    def __init__(
        self,
        in_channels: int,
        num_tasks: int,
        *,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_tasks <= 0 or hidden_channels <= 0:
            raise ValueError("model dimensions must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        widths = [in_channels, *([hidden_channels] * num_layers)]
        self.convolutions = nn.ModuleList(
            GCNConv(widths[index], widths[index + 1])
            for index in range(num_layers)
        )
        self.output = nn.Linear(hidden_channels, num_tasks)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        for convolution in self.convolutions:
            x = convolution(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, batch)
        return self.output(pooled)


def _validate_locked_config(config: GCNConfig) -> None:
    expected = GCNConfig()
    for field_name in expected.__dataclass_fields__:
        actual_value = getattr(config, field_name)
        expected_value = getattr(expected, field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"{field_name} must be {expected_value!r}, got {actual_value!r}"
            )


def _integer(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _integer_tuple(values: dict[str, Any], key: str) -> tuple[int, ...]:
    value = values.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key} must be a list of integers")
    return tuple(value)


def _string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value
