# Tox21 Error-Risk Signals — Submission Repository

This repository contains the reproducible working pipeline for the MD6329
project, *Comparing Error-Risk Signals for GCN Toxicity Prediction Across
Molecular Data Splits*. The numbered project scripts and their implementation
under `src/tox21_risk/` are the primary deliverable.

## Environment

The locked environment uses Python 3.11 and `uv`:

```bash
uv sync --frozen --group dev --group presentation
```

## Project pipeline

Run the numbered scripts from the repository root in order:

```bash
uv run python scripts/01_download_data.py
uv run python scripts/02_preprocess.py
uv run python scripts/03_generate_eda.py
uv run python scripts/04_create_splits.py
uv run python scripts/05_train_lr.py
uv run python scripts/06_build_graphs.py
uv run python scripts/07_train_gcn.py
uv run python scripts/08_mc_dropout.py
uv run python scripts/09_analyze_results.py
uv run python scripts/10_validate_pipeline.py
```

| Step | Script | Purpose | Main output |
|---:|---|---|---|
| 01 | `01_download_data.py` | Download and verify the public Tox21 source | `data/raw/tox21.csv.gz` |
| 02 | `02_preprocess.py` | Standardize, deduplicate, and fingerprint molecules | processed molecules and Morgan fingerprints |
| 03 | `03_generate_eda.py` | Generate aggregate dataset summaries | EDA tables and figures |
| 04 | `04_create_splits.py` | Create fixed random, scaffold, and similarity-cluster splits | split assignments and diagnostics |
| 05 | `05_train_lr.py` | Train endpoint-specific Logistic Regression baselines | LR predictions and metrics |
| 06 | `06_build_graphs.py` | Convert molecules to PyTorch Geometric graphs | graph cache |
| 07 | `07_train_gcn.py` | Train three seeds for each of the three split strategies | nine GCN checkpoints, predictions, and metrics |
| 08 | `08_mc_dropout.py` | Run 50-pass MC Dropout and calculate error-risk signals | uncertainty predictions and summary tables |
| 09 | `09_analyze_results.py` | Produce the final comparisons used in the report | report-ready tables, figures, and results summary |
| 10 | `10_validate_pipeline.py` | Verify saved artifacts, hashes, split integrity, and locked settings | machine-readable pipeline audit |

Configuration is locked under `configs/`. Generated raw data, processed data,
model checkpoints, and molecule-level predictions are intentionally ignored by
Git. The repository retains the small aggregate tables and figures required by
the Lectrace presentation.

## Code organization

- `src/tox21_risk/`: reusable acquisition, preprocessing, splitting, model,
  uncertainty, analysis, and audit modules
- `scripts/`: the ten executable pipeline entry points listed above
- `configs/`: data, split, Logistic Regression, and GCN settings
- `tests/`: unit, workflow, and Lectrace tests
- `data/metadata/tox21_source.json`: source URL, checksum, and dataset manifest

## Run the tests

```bash
uv run pytest
```

## Run the Lectrace presentation

The presentation uses compact executable examples plus the tracked aggregate
results:

```bash
cd presentation
uv run --project .. lectrace serve 01_tox21_error_risk.py
```

Then open `http://localhost:7000`.

## Included submission materials

- `report/Choi_Heeju_MD6329_Project_Report.pdf`: final report with the two-page
  course cover form removed
- `presentation/`: the 10-minute Lectrace presentation
- `reports/`: the three figures and three result tables required by Lectrace

The project website, website build scripts, academic poster, poster source, raw
data, processed molecule-level data, model checkpoints, and the course cover
form are intentionally excluded.

## Privacy check before publication

The two-page course cover form is not included. The report body still identifies
the author by name and student ID on its first page. Confirm that those two
fields are appropriate for public release before changing this repository from
private to public.
