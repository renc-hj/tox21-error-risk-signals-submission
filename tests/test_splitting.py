from pathlib import Path

import numpy as np
import pytest

from tox21_risk.splitting import (
    SplitConfig,
    bemis_murcko_scaffolds,
    butina_cluster_ids,
    group_assignments,
    load_split_config,
    maximum_training_similarity,
    random_assignments,
)


def test_load_split_config_reads_locked_study_choices(tmp_path: Path) -> None:
    path = tmp_path / "splits.yaml"
    path.write_text(
        """
seed: 2026
fractions: {train: 0.8, validation: 0.1, test: 0.1}
scaffold: {include_chirality: false}
similarity_cluster:
  algorithm: butina
  similarity_threshold: 0.60
  distance_cutoff: 0.40
""".strip(),
        encoding="utf-8",
    )

    config = load_split_config(path)

    assert config == SplitConfig(
        seed=2026,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
        include_scaffold_chirality=False,
        cluster_algorithm="butina",
        similarity_threshold=0.6,
        distance_cutoff=0.4,
    )


@pytest.mark.parametrize(
    "similarity,distance",
    [(0.6, 0.5), (1.1, -0.1)],
)
def test_load_split_config_rejects_inconsistent_cutoffs(
    tmp_path: Path, similarity: float, distance: float
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"""
seed: 2026
fractions: {{train: 0.8, validation: 0.1, test: 0.1}}
scaffold: {{include_chirality: false}}
similarity_cluster:
  algorithm: butina
  similarity_threshold: {similarity}
  distance_cutoff: {distance}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cutoff"):
        load_split_config(path)


def test_split_config_rejects_fractions_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="fractions"):
        SplitConfig(train_fraction=0.8, validation_fraction=0.2, test_fraction=0.2)


def test_random_assignments_are_deterministic_and_have_target_sizes() -> None:
    config = SplitConfig()

    first = random_assignments(10, config)
    second = random_assignments(10, config)

    np.testing.assert_array_equal(first, second)
    assert {name: int((first == name).sum()) for name in np.unique(first)} == {
        "test": 1,
        "train": 8,
        "validation": 1,
    }


def test_group_assignments_never_split_a_group() -> None:
    groups = ["a", "a", "a", "b", "b", "c", "d", "e", "f", "g"]

    assignments = group_assignments(groups, SplitConfig())

    for group in set(groups):
        positions = [index for index, value in enumerate(groups) if value == group]
        assert len(set(assignments[positions])) == 1
    assert {name: int((assignments == name).sum()) for name in np.unique(assignments)} == {
        "test": 1,
        "train": 8,
        "validation": 1,
    }


def test_group_assignments_accept_numpy_cluster_ids() -> None:
    groups = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])

    assignments = group_assignments(groups, SplitConfig())

    assert len(assignments) == len(groups)
    assert assignments[0] == assignments[1]


def test_bemis_murcko_scaffolds_keep_acyclic_molecules_in_empty_group() -> None:
    scaffolds = bemis_murcko_scaffolds(["CCO", "CCN", "c1ccccc1"])

    assert scaffolds == ["", "", "c1ccccc1"]


def test_butina_clusters_only_fingerprints_above_similarity_threshold() -> None:
    fingerprints = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )

    cluster_ids = butina_cluster_ids(fingerprints, SplitConfig())

    assert cluster_ids[0] == cluster_ids[1]
    assert cluster_ids[0] != cluster_ids[2]
    assert cluster_ids[0] != cluster_ids[3]
    assert cluster_ids[2] != cluster_ids[3]


def test_maximum_training_similarity_is_reported_only_for_test_molecules() -> None:
    fingerprints = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assignments = np.array(["train", "train", "test", "test"])

    similarities = maximum_training_similarity(fingerprints, assignments)

    assert np.isnan(similarities[:2]).all()
    np.testing.assert_allclose(similarities[2:], [1.0, 1.0 / 3.0])


def test_maximum_training_similarity_accepts_multiple_target_partitions() -> None:
    fingerprints = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assignments = np.array(["train", "train", "validation", "test"])

    similarities = maximum_training_similarity(
        fingerprints,
        assignments,
        target_partitions=("validation", "test"),
    )

    assert np.isnan(similarities[:2]).all()
    np.testing.assert_allclose(similarities[2:], [1.0, 1.0 / 3.0])
