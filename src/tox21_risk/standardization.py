"""Deterministic, audited molecular standardization with RDKit."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True, slots=True)
class StandardizationResult:
    """Outcome and audit information for one source SMILES value."""

    original_smiles: str | None
    standardized_smiles: str | None
    inchikey: str | None
    status: str
    failure_reason: str | None
    transformation_log: tuple[str, ...]


def standardize_smiles(smiles: str | None) -> StandardizationResult:
    """Standardize one SMILES while preserving stereochemical identity."""

    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return _failure(smiles, "empty_smiles")

    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        return _failure(smiles, "parse_or_sanitize_failed")

    log: list[str] = ["parsed_and_sanitized"]
    try:
        molecule, selection_step = _select_fragment(molecule)
        log.append(selection_step)

        molecule = rdMolStandardize.Cleanup(molecule)
        log.append("cleanup")
        molecule = rdMolStandardize.Normalizer().normalize(molecule)
        log.append("normalize")
        molecule = rdMolStandardize.Reionizer().reionize(molecule)
        log.append("reionize")
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        log.append("uncharge")
        Chem.SanitizeMol(molecule)
        log.append("final_sanitize")

        standardized = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        if not standardized:
            return _failure(smiles, "empty_standardized_smiles", tuple(log))

        inchikey: str | None
        try:
            inchikey = inchi.MolToInchiKey(molecule) or None
        except Exception:
            inchikey = None

        return StandardizationResult(
            original_smiles=smiles,
            standardized_smiles=standardized,
            inchikey=inchikey,
            status="success",
            failure_reason=None,
            transformation_log=tuple(log),
        )
    except Exception as exc:
        return _failure(
            smiles,
            f"standardization_failed:{type(exc).__name__}",
            tuple(log),
        )


def _select_fragment(molecule: Chem.Mol) -> tuple[Chem.Mol, str]:
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if len(fragments) == 1:
        return fragments[0], "single_fragment"

    organic = [fragment for fragment in fragments if _contains_carbon(fragment)]
    candidates = organic if organic else list(fragments)
    step = (
        "selected_largest_organic_fragment"
        if organic
        else "selected_largest_inorganic_fragment"
    )
    selected = max(candidates, key=_fragment_score)
    return selected, step


def _contains_carbon(molecule: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())


def _fragment_score(molecule: Chem.Mol) -> tuple[int, float, str]:
    return (
        molecule.GetNumHeavyAtoms(),
        Descriptors.MolWt(molecule),
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
    )


def _failure(
    original_smiles: str | None,
    reason: str,
    transformation_log: tuple[str, ...] = (),
) -> StandardizationResult:
    return StandardizationResult(
        original_smiles=original_smiles,
        standardized_smiles=None,
        inchikey=None,
        status="failed",
        failure_reason=reason,
        transformation_log=transformation_log,
    )
