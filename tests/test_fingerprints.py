import numpy as np
import pytest

from tox21_risk.fingerprints import morgan_fingerprint_matrix


def test_morgan_fingerprints_have_locked_shape_type_and_values() -> None:
    matrix = morgan_fingerprint_matrix(["CCO", "CCN"])

    assert matrix.shape == (2, 1_024)
    assert matrix.dtype == np.uint8
    assert set(np.unique(matrix)).issubset({0, 1})
    assert not np.array_equal(matrix[0], matrix[1])


def test_morgan_fingerprints_are_deterministic() -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]

    first = morgan_fingerprint_matrix(smiles)
    second = morgan_fingerprint_matrix(smiles)

    np.testing.assert_array_equal(first, second)


def test_invalid_smiles_raises_instead_of_returning_zero_vector() -> None:
    with pytest.raises(ValueError, match="index 1"):
        morgan_fingerprint_matrix(["CCO", "not-a-smiles"])
