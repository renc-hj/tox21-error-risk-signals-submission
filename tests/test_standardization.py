import pytest

from tox21_risk.standardization import standardize_smiles


def test_valid_smiles_is_canonical_and_deterministic() -> None:
    first = standardize_smiles("OCC")
    second = standardize_smiles("OCC")

    assert first == second
    assert first.status == "success"
    assert first.standardized_smiles == "CCO"
    assert first.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert first.failure_reason is None
    assert first.transformation_log


def test_salt_selects_largest_organic_fragment() -> None:
    result = standardize_smiles("CC(=O)[O-].[Na+]")

    assert result.status == "success"
    assert result.standardized_smiles == "CC(=O)O"
    assert "selected_largest_organic_fragment" in result.transformation_log


@pytest.mark.parametrize("smiles", [None, "", "   "])
def test_empty_smiles_returns_explicit_failure(smiles: str | None) -> None:
    result = standardize_smiles(smiles)

    assert result.status == "failed"
    assert result.standardized_smiles is None
    assert result.inchikey is None
    assert result.failure_reason == "empty_smiles"


def test_invalid_smiles_returns_explicit_failure() -> None:
    result = standardize_smiles("not-a-smiles")

    assert result.status == "failed"
    assert result.failure_reason == "parse_or_sanitize_failed"


def test_inorganic_only_smiles_uses_valid_fallback() -> None:
    result = standardize_smiles("[Na+].[Cl-]")

    assert result.status == "success"
    assert result.standardized_smiles is not None
    assert "selected_largest_inorganic_fragment" in result.transformation_log


def test_stereoisomers_remain_distinct() -> None:
    clockwise = standardize_smiles("F[C@H](Cl)Br")
    counterclockwise = standardize_smiles("F[C@@H](Cl)Br")

    assert clockwise.status == counterclockwise.status == "success"
    assert clockwise.standardized_smiles != counterclockwise.standardized_smiles
