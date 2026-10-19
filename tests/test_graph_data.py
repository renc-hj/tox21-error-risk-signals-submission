from __future__ import annotations

import numpy as np
import pytest
import torch
from rdkit import Chem

from tox21_risk.graph_data import (
    ATOM_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    atom_feature_vector,
    build_graphs,
    load_graph_cache,
    save_graph_cache,
    smiles_to_graph,
)


TASKS = ("task_a", "task_b")


def test_atom_feature_vector_uses_locked_166_feature_layout() -> None:
    atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)

    features = atom_feature_vector(atom)

    assert ATOM_FEATURE_DIM == 166
    assert features.shape == (166,)
    assert features.dtype == np.float32
    assert np.flatnonzero(features).tolist() == [5, 119, 136, 147, 156, 162]


def test_atom_feature_vector_encodes_aromaticity_and_charge_fallback() -> None:
    aromatic_atom = Chem.MolFromSmiles("c1ccccc1").GetAtomWithIdx(0)
    uncommon_charge = Chem.MolFromSmiles("[Fe+6]").GetAtomWithIdx(0)

    aromatic_features = atom_feature_vector(aromatic_atom)
    charge_features = atom_feature_vector(uncommon_charge)

    assert aromatic_features[151] == 1.0
    assert charge_features[142] == 1.0
    assert np.array_equal(
        atom_feature_vector(aromatic_atom), atom_feature_vector(aromatic_atom)
    )


def test_smiles_to_graph_has_bidirectional_edges_and_preserves_missing_labels() -> None:
    graph = smiles_to_graph("mol_1", "CCO", [1.0, np.nan])

    assert graph.mol_id == "mol_1"
    assert graph.standardized_smiles == "CCO"
    assert graph.x.shape == (3, ATOM_FEATURE_DIM)
    assert graph.edge_index.shape == (2, 4)
    assert set(map(tuple, graph.edge_index.t().tolist())) == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }
    assert graph.y.shape == (1, 2)
    assert graph.y[0, 0].item() == 1.0
    assert torch.isnan(graph.y[0, 1])


def test_smiles_to_graph_accepts_bondless_molecule() -> None:
    graph = smiles_to_graph("mol_br", "[Br-]", [0.0, 1.0])

    assert graph.edge_index.shape == (2, 0)
    assert graph.x.shape == (1, ATOM_FEATURE_DIM)


def test_smiles_to_graph_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="invalid standardized SMILES"):
        smiles_to_graph("bad", "not-a-smiles", [0.0, 1.0])


def test_graph_cache_round_trip_includes_provenance(tmp_path) -> None:
    molecule_path = tmp_path / "molecules.csv"
    molecule_path.write_text(
        "mol_id,standardized_smiles,task_a,task_b\n"
        "mol_1,CC,1,\n"
        "mol_2,[Br-],0,1\n",
        encoding="utf-8",
    )
    import pandas as pd

    frame = pd.read_csv(molecule_path)
    graphs = build_graphs(frame, TASKS)
    cache_path = tmp_path / "graphs.pt"

    save_graph_cache(graphs, cache_path, molecule_path, TASKS)
    cache = load_graph_cache(cache_path)

    assert cache["feature_schema_version"] == FEATURE_SCHEMA_VERSION == 1
    assert cache["feature_dim"] == ATOM_FEATURE_DIM
    assert cache["endpoint_order"] == list(TASKS)
    assert cache["molecule_count"] == 2
    assert cache["mol_ids"] == ["mol_1", "mol_2"]
    assert len(cache["graphs"]) == 2
    assert len(cache["molecule_file_sha256"]) == 64
    assert cache["graphs"][1].edge_index.shape == (2, 0)


def test_load_graph_cache_rejects_unknown_format_version(tmp_path) -> None:
    cache_path = tmp_path / "graphs.pt"
    torch.save(
        {
            "format_version": 999,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_dim": ATOM_FEATURE_DIM,
            "endpoint_order": list(TASKS),
            "molecule_count": 0,
            "mol_ids": [],
            "molecule_file_sha256": "0" * 64,
            "graphs": [],
        },
        cache_path,
    )

    with pytest.raises(ValueError, match="format version"):
        load_graph_cache(cache_path)


def test_build_graphs_rejects_nonbinary_observed_label() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "mol_id": ["mol_1"],
            "standardized_smiles": ["CC"],
            "task_a": [2.0],
            "task_b": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="binary or missing"):
        build_graphs(frame, TASKS)
