"""Convert standardized Tox21 molecules into reproducible PyG graphs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, ChiralType, HybridizationType
from torch_geometric.data import Data


FEATURE_SCHEMA_VERSION = 1
ATOM_FEATURE_DIM = 166

_HYBRIDIZATIONS = (
    HybridizationType.UNSPECIFIED,
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.OTHER,
)
_CHIRALITIES = (
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
)


def atom_feature_vector(atom: Atom) -> np.ndarray:
    """Return the locked 166-value feature vector for one RDKit atom."""

    atomic_number = _indexed_one_hot(atom.GetAtomicNum(), tuple(range(1, 119)))
    degree = _indexed_one_hot(atom.GetDegree(), tuple(range(0, 11)))
    formal_charge = _indexed_one_hot(atom.GetFormalCharge(), tuple(range(-5, 6)))
    hybridization = _indexed_one_hot(
        atom.GetHybridization(), _HYBRIDIZATIONS[:-1]
    )
    aromaticity = np.asarray([float(atom.GetIsAromatic())], dtype=np.float32)
    total_hydrogens = _indexed_one_hot(atom.GetTotalNumHs(), tuple(range(0, 9)))
    chirality = _indexed_one_hot(atom.GetChiralTag(), _CHIRALITIES)
    features = np.concatenate(
        (
            atomic_number,
            degree,
            formal_charge,
            hybridization,
            aromaticity,
            total_hydrogens,
            chirality,
        )
    )
    if features.shape != (ATOM_FEATURE_DIM,):
        raise RuntimeError(
            f"atom feature schema produced {len(features)} values, expected "
            f"{ATOM_FEATURE_DIM}"
        )
    return features


def smiles_to_graph(
    mol_id: str, standardized_smiles: str, labels: Sequence[float]
) -> Data:
    """Convert one standardized molecule and its endpoint labels to a graph."""

    molecule = Chem.MolFromSmiles(standardized_smiles)
    if molecule is None:
        raise ValueError(
            f"{mol_id}: invalid standardized SMILES {standardized_smiles!r}"
        )
    if molecule.GetNumAtoms() == 0:
        raise ValueError(f"{mol_id}: standardized SMILES contains no atoms")

    node_features = np.stack(
        [atom_feature_vector(atom) for atom in molecule.GetAtoms()]
    )
    directed_edges: list[tuple[int, int]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        directed_edges.extend(((begin, end), (end, begin)))
    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(
        x=torch.from_numpy(node_features),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float32).reshape(1, -1),
    )
    graph.mol_id = str(mol_id)
    graph.standardized_smiles = standardized_smiles
    return graph


def build_graphs(frame: pd.DataFrame, tasks: Sequence[str]) -> list[Data]:
    """Build graphs in the exact row order of a validated molecule table."""

    required = {"mol_id", "standardized_smiles", *tasks}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("molecule table is missing columns: " + ", ".join(missing))
    if frame["mol_id"].isna().any() or not frame["mol_id"].is_unique:
        raise ValueError("molecule mol_id values must be present and unique")

    label_matrix = frame[list(tasks)].to_numpy(dtype=float)
    observed = label_matrix[~np.isnan(label_matrix)]
    if not np.isin(observed, (0.0, 1.0)).all():
        raise ValueError("endpoint labels must be binary or missing")

    return [
        smiles_to_graph(row.mol_id, row.standardized_smiles, labels)
        for row, labels in zip(
            frame[["mol_id", "standardized_smiles"]].itertuples(index=False),
            label_matrix,
            strict=True,
        )
    ]


def save_graph_cache(
    graphs: Sequence[Data],
    cache_path: Path,
    molecule_path: Path,
    tasks: Sequence[str],
) -> None:
    """Save local graphs together with the complete provenance contract."""

    payload: dict[str, Any] = {
        "format_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": ATOM_FEATURE_DIM,
        "endpoint_order": list(tasks),
        "molecule_count": len(graphs),
        "mol_ids": [str(graph.mol_id) for graph in graphs],
        "molecule_file_sha256": _sha256(molecule_path),
        "graphs": list(graphs),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def load_graph_cache(cache_path: Path) -> dict[str, Any]:
    """Load a graph cache created locally by :func:`save_graph_cache`."""

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = {
        "format_version",
        "feature_schema_version",
        "feature_dim",
        "endpoint_order",
        "molecule_count",
        "mol_ids",
        "molecule_file_sha256",
        "graphs",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("graph cache does not match the required format")
    if payload["format_version"] != 1:
        raise ValueError("graph cache format version is incompatible")
    return payload


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local artifact."""

    return _sha256(path)


def _indexed_one_hot(value: object, known_values: Sequence[object]) -> np.ndarray:
    encoded = np.zeros(len(known_values) + 1, dtype=np.float32)
    try:
        index = known_values.index(value)
    except ValueError:
        index = len(known_values)
    encoded[index] = 1.0
    return encoded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
