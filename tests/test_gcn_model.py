from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from tox21_risk.gcn_model import GCNConfig, Tox21GCN, load_gcn_config
from tox21_risk.graph_data import ATOM_FEATURE_DIM, smiles_to_graph


def test_load_gcn_config_reads_locked_submission_settings(tmp_path) -> None:
    path = tmp_path / "gcn.yaml"
    path.write_text(
        """seeds: [2026, 2027, 2028]
hidden_channels: 128
num_layers: 3
dropout: 0.2
batch_size: 64
learning_rate: 0.001
weight_decay: 0.00001
max_epochs: 200
patience: 20
device: cpu
""",
        encoding="utf-8",
    )

    config = load_gcn_config(path)

    assert config == GCNConfig()


def test_load_gcn_config_rejects_changed_locked_architecture(tmp_path) -> None:
    path = tmp_path / "gcn.yaml"
    path.write_text(
        """seeds: [2026, 2027, 2028]
hidden_channels: 64
num_layers: 3
dropout: 0.2
batch_size: 64
learning_rate: 0.001
weight_decay: 0.00001
max_epochs: 200
patience: 20
device: cpu
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hidden_channels must be 128"):
        load_gcn_config(path)


def test_gcn_returns_twelve_logits_for_two_graph_batch() -> None:
    graphs = [
        smiles_to_graph("mol_1", "CC", [0.0] * 12),
        smiles_to_graph("mol_2", "CO", [1.0] * 12),
    ]
    batch = Batch.from_data_list(graphs)
    model = Tox21GCN(in_channels=ATOM_FEATURE_DIM, num_tasks=12)

    logits = model(batch.x, batch.edge_index, batch.batch)

    assert len(model.convolutions) == 3
    assert model.dropout == 0.2
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_gcn_processes_bondless_graph() -> None:
    graph = smiles_to_graph("mol_br", "[Br-]", [0.0] * 12)
    batch = Batch.from_data_list([graph])
    model = Tox21GCN(in_channels=ATOM_FEATURE_DIM, num_tasks=12)

    logits = model(batch.x, batch.edge_index, batch.batch)

    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()
