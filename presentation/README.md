# Lectrace presentation

This directory contains one coherent 10-minute Lectrace presentation with four
selected execution states. It focuses on the computational steps that explain
how the model produces and evaluates its results rather than replaying the full
training pipeline.

## Run the presentation

From the repository root:

```bash
uv sync --group presentation
cd presentation
uv run --project .. lectrace serve 01_tox21_error_risk.py
```

Open `http://localhost:7000` and use the left and right Arrow keys. Values marked
with `@inspect` appear in the variable panel.

## Selected execution states

1. Group-preserving molecular splits and the realised Figure 1 comparison.
2. The three-layer GCN's graph-to-12-endpoint shape transition.
3. Single-pass uncertainty versus 50-pass MC Dropout mutual information.
4. Full error-risk ranking and the paired within-similarity-bin comparison.

Each state identifies what is happening, what changes, and why the computation
matters. Compact examples expose intermediate values but are clearly separated
from the reported experiment. All reported numerical results are loaded from the
tracked result tables.
