"""Deterministic molecule splitting and structural-similarity helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Small, immutable record of every split choice used in the study."""

    seed: int = 2026
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    include_scaffold_chirality: bool = False
    cluster_algorithm: str = "butina"
    similarity_threshold: float = 0.6
    distance_cutoff: float = 0.4

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(value <= 0 or value >= 1 for value in fractions) or not np.isclose(
            sum(fractions), 1.0
        ):
            raise ValueError("split fractions must be positive and sum to one")
        if self.cluster_algorithm != "butina":
            raise ValueError("cluster algorithm must be 'butina'")
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("similarity cutoff must be in (0, 1]")
        if not 0 <= self.distance_cutoff < 1 or not np.isclose(
            self.distance_cutoff, 1.0 - self.similarity_threshold
        ):
            raise ValueError(
                "distance cutoff must equal 1 - similarity cutoff"
            )


def load_split_config(path: Path) -> SplitConfig:
    """Load a human-readable YAML file and validate all split choices."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("split configuration must be a YAML mapping")
    fractions = _mapping(raw, "fractions")
    scaffold = _mapping(raw, "scaffold")
    cluster = _mapping(raw, "similarity_cluster")
    return SplitConfig(
        seed=_integer(raw, "seed"),
        train_fraction=_number(fractions, "train"),
        validation_fraction=_number(fractions, "validation"),
        test_fraction=_number(fractions, "test"),
        include_scaffold_chirality=_boolean(scaffold, "include_chirality"),
        cluster_algorithm=_string(cluster, "algorithm"),
        similarity_threshold=_number(cluster, "similarity_threshold"),
        distance_cutoff=_number(cluster, "distance_cutoff"),
    )


def random_assignments(n_molecules: int, config: SplitConfig) -> np.ndarray:
    """Return one seeded random 80/10/10-style assignment array."""

    if n_molecules <= 0:
        raise ValueError("n_molecules must be positive")
    order = np.random.default_rng(config.seed).permutation(n_molecules)
    n_train = int(config.train_fraction * n_molecules)
    n_validation = int(config.validation_fraction * n_molecules)
    assignments = np.full(n_molecules, "test", dtype=object)
    assignments[order[:n_train]] = "train"
    assignments[order[n_train : n_train + n_validation]] = "validation"
    return assignments.astype(str)


def bemis_murcko_scaffolds(
    smiles: Sequence[str], include_chirality: bool = False
) -> list[str]:
    """Return canonical Bemis-Murcko scaffold SMILES for each molecule."""

    scaffolds: list[str] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value, sanitize=True)
        if molecule is None:
            raise ValueError(f"invalid SMILES at index {index}: {value!r}")
        scaffolds.append(
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule, includeChirality=include_chirality
            )
        )
    return scaffolds


def group_assignments(
    groups: Sequence[Hashable], config: SplitConfig
) -> np.ndarray:
    """Assign whole groups using deterministic largest-first cutoffs."""

    if len(groups) == 0:
        raise ValueError("groups must not be empty")
    positions_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for position, group in enumerate(groups):
        positions_by_group[group].append(position)

    ordered_groups = sorted(
        positions_by_group.values(),
        key=lambda positions: (len(positions), positions[0]),
        reverse=True,
    )
    n_molecules = len(groups)
    train_cutoff = config.train_fraction * n_molecules
    validation_cutoff = (
        config.train_fraction + config.validation_fraction
    ) * n_molecules
    partition_indices = {name: [] for name in PARTITIONS}

    for positions in ordered_groups:
        if len(partition_indices["train"]) + len(positions) <= train_cutoff:
            partition = "train"
        elif (
            len(partition_indices["train"])
            + len(partition_indices["validation"])
            + len(positions)
            <= validation_cutoff
        ):
            partition = "validation"
        else:
            partition = "test"
        partition_indices[partition].extend(positions)

    assignments = np.empty(n_molecules, dtype=object)
    for partition, positions in partition_indices.items():
        assignments[positions] = partition
    return assignments.astype(str)


def butina_cluster_ids(
    fingerprints: np.ndarray, config: SplitConfig
) -> np.ndarray:
    """Cluster binary fingerprints using Tanimoto distance and Butina."""

    bit_vectors = _bit_vectors(fingerprints)
    n_fingerprints = len(bit_vectors)
    distances = np.empty(
        n_fingerprints * (n_fingerprints - 1) // 2, dtype=np.float64
    )
    offset = 0
    for index in range(1, n_fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            bit_vectors[index], bit_vectors[:index]
        )
        distances[offset : offset + index] = 1.0 - np.asarray(similarities)
        offset += index

    clusters = Butina.ClusterData(
        distances,
        n_fingerprints,
        config.distance_cutoff,
        isDistData=True,
    )
    cluster_ids = np.empty(n_fingerprints, dtype=np.int64)
    for cluster_id, members in enumerate(clusters):
        cluster_ids[list(members)] = cluster_id
    return cluster_ids


def maximum_training_similarity(
    fingerprints: np.ndarray,
    assignments: Sequence[str],
    target_partitions: Sequence[str] = ("test",),
) -> np.ndarray:
    """Return target molecules' maximum Tanimoto similarity to training."""

    bit_vectors = _bit_vectors(fingerprints)
    split_array = np.asarray(assignments)
    if len(split_array) != len(bit_vectors):
        raise ValueError("assignments and fingerprints must have the same row count")
    train_positions = np.flatnonzero(split_array == "train")
    if len(train_positions) == 0:
        raise ValueError("at least one training molecule is required")
    if not target_partitions or "train" in target_partitions:
        raise ValueError("target partitions must be non-empty and exclude train")
    training = [bit_vectors[position] for position in train_positions]
    output = np.full(len(bit_vectors), np.nan, dtype=float)
    target_mask = np.isin(split_array, tuple(target_partitions))
    for position in np.flatnonzero(target_mask):
        output[position] = max(
            DataStructs.BulkTanimotoSimilarity(bit_vectors[position], training)
        )
    return output


def _bit_vectors(fingerprints: np.ndarray) -> list[DataStructs.ExplicitBitVect]:
    matrix = np.asarray(fingerprints)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("fingerprints must be a non-empty two-dimensional matrix")
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("fingerprints must contain only binary values")
    return [
        DataStructs.CreateFromBitString("".join(row.astype(str))) for row in matrix
    ]


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _integer(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _boolean(values: dict[str, Any], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value
