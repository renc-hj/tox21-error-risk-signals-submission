# Tox21 Error-Risk Signals — Submission Repository

This repository contains the code and presentation materials for the MD6329
project, *Comparing Error-Risk Signals for GCN Toxicity Prediction Across
Molecular Data Splits*.

## Included deliverables

- `report/Choi_Heeju_MD6329_Project_Report.pdf`: final report with the two-page
  course cover form removed
- `presentation/`: the 10-minute Lectrace presentation
- `src/tox21_risk/`: reusable analysis and model code
- `scripts/01_download_data.py` through `scripts/10_validate_pipeline.py`: the
  reproducible workflow
- `configs/`: locked experiment settings
- `reports/`: the three figures and three result tables required by Lectrace
- `tests/`: code and Lectrace tests

The project website, website build scripts, academic poster, poster source, raw
data, processed molecule-level data, model checkpoints, and the course cover
form are intentionally excluded.

## Environment

The locked environment uses Python 3.11 and `uv`:

```bash
uv sync --frozen --group dev --group presentation
```

## Run the Lectrace presentation

```bash
cd presentation
uv run --project .. lectrace serve 01_tox21_error_risk.py
```

Then open `http://localhost:7000`.

## Run the tests

```bash
uv run pytest
```

## Reproduce the analysis

Run the numbered scripts from the repository root in order. The first script
downloads the public Tox21 source data. Generated raw data, processed data,
checkpoints, and other large artifacts remain ignored by Git.

## Privacy check before publication

The two-page course cover form is not included. The report body still identifies
the author by name and student ID on its first page. Confirm that those two
fields are appropriate for public release before changing this repository from
private to public.
