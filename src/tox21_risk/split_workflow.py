"""Create fixed Tox21 splits and compact report-ready audit tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tox21_risk.fingerprints import morgan_fingerprint_matrix
from tox21_risk.splitting import (
    PARTITIONS,
    SplitConfig,
    bemis_murcko_scaffolds,
    butina_cluster_ids,
    group_assignments,
    maximum_training_similarity,
    random_assignments,
)


SPLIT_COLUMNS = {
    "random": "random_split",
    "scaffold": "scaffold_split",
    "similarity_cluster": "similarity_cluster_split",
}
FINGERPRINT_RADIUS = 2


def run_split_workflow(
    molecule_path: Path,
    fingerprint_path: Path,
    assignments_path: Path,
    summary_path: Path,
    prevalence_path: Path,
    similarity_path: Path,
    metadata_path: Path,
    tasks: Sequence[str],
    config: SplitConfig,
) -> dict[str, object]:
    """Generate fixed assignments and all tables needed for Data section 3.3."""

    molecules = pd.read_csv(molecule_path)
    fingerprints = np.load(fingerprint_path, allow_pickle=False)
    _validate_inputs(molecules, fingerprints, tasks)

    scaffolds = bemis_murcko_scaffolds(
        molecules["standardized_smiles"].tolist(),
        include_chirality=config.include_scaffold_chirality,
    )
    cluster_ids = butina_cluster_ids(fingerprints, config)
    assignments = pd.DataFrame(
        {
            "mol_id": molecules["mol_id"],
            "scaffold": scaffolds,
            "cluster_id": cluster_ids,
            "random_split": random_assignments(len(molecules), config),
            "scaffold_split": group_assignments(scaffolds, config),
            "similarity_cluster_split": group_assignments(cluster_ids, config),
        }
    )

    summary_rows: list[dict[str, object]] = []
    prevalence_rows: list[dict[str, object]] = []
    similarity_rows: list[dict[str, object]] = []
    for strategy, split_column in SPLIT_COLUMNS.items():
        split_values = assignments[split_column].to_numpy()
        _validate_assignments(split_values, len(molecules))
        max_similarity = maximum_training_similarity(fingerprints, split_values)
        test_similarities = max_similarity[split_values == "test"]

        summary_rows.append(
            {
                "strategy": strategy,
                "n_total": len(molecules),
                "n_train": int((split_values == "train").sum()),
                "n_validation": int((split_values == "validation").sum()),
                "n_test": int((split_values == "test").sum()),
                "molecule_overlap_count": _partition_overlap_count(
                    molecules["mol_id"], split_values
                ),
                "scaffold_overlap_count": (
                    _group_overlap_count(scaffolds, split_values)
                    if strategy == "scaffold"
                    else pd.NA
                ),
                "cluster_overlap_count": (
                    _group_overlap_count(cluster_ids, split_values)
                    if strategy == "similarity_cluster"
                    else pd.NA
                ),
            }
        )
        similarity_rows.append(
            {
                "strategy": strategy,
                "n_test": len(test_similarities),
                "test_max_tanimoto_mean": float(np.mean(test_similarities)),
                "test_max_tanimoto_median": float(np.median(test_similarities)),
            }
        )
        prevalence_rows.extend(
            _endpoint_prevalence_rows(molecules, tasks, strategy, split_values)
        )

    summary = pd.DataFrame(summary_rows)
    prevalence = pd.DataFrame(prevalence_rows)
    similarity = pd.DataFrame(similarity_rows)
    metadata: dict[str, object] = {
        "n_molecules": len(molecules),
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_bits": int(fingerprints.shape[1]),
        "seed": config.seed,
        "fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
            "test": config.test_fraction,
        },
        "scaffold": {
            "method": "Bemis-Murcko",
            "include_chirality": config.include_scaffold_chirality,
        },
        "cluster_algorithm": config.cluster_algorithm,
        "similarity_metric": "Tanimoto",
        "similarity_threshold": config.similarity_threshold,
        "distance_cutoff": config.distance_cutoff,
        "molecule_file_sha256": _sha256(molecule_path),
        "fingerprint_file_sha256": _sha256(fingerprint_path),
    }

    for path in (
        assignments_path,
        summary_path,
        prevalence_path,
        similarity_path,
        metadata_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    prevalence.to_csv(prevalence_path, index=False, lineterminator="\n")
    similarity.to_csv(similarity_path, index=False, lineterminator="\n")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _validate_inputs(
    molecules: pd.DataFrame, fingerprints: np.ndarray, tasks: Sequence[str]
) -> None:
    required = {"mol_id", "standardized_smiles", *tasks}
    missing = sorted(required.difference(molecules.columns))
    if missing:
        raise ValueError(f"molecule table is missing columns: {', '.join(missing)}")
    if molecules["mol_id"].isna().any() or not molecules["mol_id"].is_unique:
        raise ValueError("mol_id values must be present and unique")
    if len(molecules) != len(fingerprints):
        raise ValueError("molecule and fingerprint row count must match")
    expected_fingerprints = morgan_fingerprint_matrix(
        molecules["standardized_smiles"].tolist(),
        radius=FINGERPRINT_RADIUS,
        n_bits=fingerprints.shape[1],
    )
    if not np.array_equal(fingerprints, expected_fingerprints):
        raise ValueError(
            "fingerprint rows are not aligned with the molecule table"
        )
    for task in tasks:
        observed = molecules[task].dropna()
        if not observed.isin([0, 1]).all():
            raise ValueError(f"{task} contains labels other than 0, 1, or missing")


def _validate_assignments(assignments: np.ndarray, n_molecules: int) -> None:
    if len(assignments) != n_molecules:
        raise ValueError("every molecule must receive one split assignment")
    observed = set(assignments)
    if observed != set(PARTITIONS):
        raise ValueError(f"expected all three partitions, found {sorted(observed)}")


def _partition_overlap_count(
    molecule_ids: Sequence[str], assignments: np.ndarray
) -> int:
    split_sets = {
        partition: set(np.asarray(molecule_ids)[assignments == partition])
        for partition in PARTITIONS
    }
    return sum(
        len(split_sets[first] & split_sets[second])
        for index, first in enumerate(PARTITIONS)
        for second in PARTITIONS[index + 1 :]
    )


def _group_overlap_count(
    groups: Sequence[object], assignments: np.ndarray
) -> int:
    frame = pd.DataFrame({"group": list(groups), "split": assignments})
    return int((frame.groupby("group", dropna=False)["split"].nunique() > 1).sum())


def _endpoint_prevalence_rows(
    molecules: pd.DataFrame,
    tasks: Sequence[str],
    strategy: str,
    assignments: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition in PARTITIONS:
        subset = molecules.loc[assignments == partition]
        for task in tasks:
            labels = subset[task]
            positive = int((labels == 1).sum())
            negative = int((labels == 0).sum())
            observed = positive + negative
            rows.append(
                {
                    "strategy": strategy,
                    "split": partition,
                    "endpoint": task,
                    "positive": positive,
                    "negative": negative,
                    "missing": int(labels.isna().sum()),
                    "observed": observed,
                    "positive_prevalence": (
                        positive / observed if observed else np.nan
                    ),
                }
            )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
