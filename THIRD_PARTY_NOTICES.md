# Third-party notices

This project uses third-party data, software, and published workflows. No source code was copied from the comparison repositories listed below.

## Tox21 data

The pipeline downloads the MoleculeNet/DeepChem Tox21 mirror from the URL recorded in `configs/data.yaml`. Raw and molecule-level processed data are not redistributed in this repository. Users are responsible for checking the original providers' current terms before obtaining or redistributing the data.

The dataset and benchmark context should be cited as:

> Wu, Z., Ramsundar, B., Feinberg, E. N., et al. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9, 513–530. https://doi.org/10.1039/C7SC02664A

## Software and workflow references

- **RDKit** — BSD 3-Clause License. Used for molecular parsing, standardization, fingerprints, InChIKeys, and structure drawings. https://github.com/rdkit/rdkit
- **DeepChem** — MIT License. The public Tox21 mirror and MoleculeNet schema are used; DeepChem source code is not bundled. https://github.com/deepchem/deepchem
- **Reliability-Aware Molecular Toxicity Prediction** — BSD 3-Clause License. Consulted as the closest reliability-aware workflow; source code is not bundled. https://github.com/ybenachour/reliability-aware-molecular-toxicity

Python package versions used in the verified run are recorded in `uv.lock`. Each installed dependency remains subject to its own license.
