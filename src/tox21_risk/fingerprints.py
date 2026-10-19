"""Morgan fingerprint generation for the shared molecular representation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def morgan_fingerprint_matrix(
    smiles: Sequence[str], radius: int = 2, n_bits: int = 1_024
) -> np.ndarray:
    """Return a deterministic binary Morgan fingerprint matrix."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value, sanitize=True)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"invalid SMILES at index {index}: {value!r}")
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    return matrix
