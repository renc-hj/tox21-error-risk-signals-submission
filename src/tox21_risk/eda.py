"""Aggregate, report-ready exploratory analysis for processed Tox21 data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def build_endpoint_summary(
    frame: pd.DataFrame, tasks: Sequence[str]
) -> pd.DataFrame:
    """Count endpoint labels and calculate prevalence among observed labels."""

    records: list[dict[str, object]] = []
    for task in tasks:
        if task not in frame.columns:
            raise ValueError(f"processed data is missing endpoint column: {task}")
        values = pd.to_numeric(frame[task], errors="raise")
        observed_values = set(values.dropna().unique().tolist())
        if not observed_values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"{task} contains values other than 0, 1, or missing")
        positive = int((values == 1).sum())
        negative = int((values == 0).sum())
        missing = int(values.isna().sum())
        observed = positive + negative
        prevalence = positive / observed if observed else math.nan
        records.append(
            {
                "endpoint": task,
                "positive": positive,
                "negative": negative,
                "missing": missing,
                "observed": observed,
                "positive_prevalence": prevalence,
            }
        )
    return pd.DataFrame.from_records(records)


def build_dataset_summary(
    frame: pd.DataFrame, audit: pd.DataFrame, tasks: Sequence[str]
) -> dict[str, object]:
    """Build an aggregate dataset and preprocessing audit summary."""

    if frame.empty:
        raise ValueError("processed molecule table must not be empty")
    required_audit = {"status", "standardized_smiles"}
    missing_audit = sorted(required_audit.difference(audit.columns))
    if missing_audit:
        raise ValueError(f"audit is missing columns: {', '.join(missing_audit)}")

    observed_per_molecule = frame[list(tasks)].notna().sum(axis=1)
    atom_counts = _atom_counts(frame["standardized_smiles"])
    successful_audit = audit.loc[audit["status"] == "success"]
    identity_counts = successful_audit["standardized_smiles"].value_counts()

    endpoint_conflicts = 0
    if "conflict_count" in successful_audit.columns:
        endpoint_conflicts = int(
            successful_audit.groupby("standardized_smiles", sort=True)[
                "conflict_count"
            ]
            .max()
            .fillna(0)
            .sum()
        )

    return {
        "raw_rows": len(audit),
        "standardization_failed_rows": int((audit["status"] == "failed").sum()),
        "standardized_rows": len(successful_audit),
        "duplicate_rows_removed": int((identity_counts - 1).clip(lower=0).sum()),
        "duplicate_groups": int((identity_counts > 1).sum()),
        "endpoint_conflicts": endpoint_conflicts,
        "molecule_count": len(frame),
        "endpoint_count": len(tuple(tasks)),
        "observed_labels": int(frame[list(tasks)].notna().sum().sum()),
        "missing_labels": int(frame[list(tasks)].isna().sum().sum()),
        "mean_observed_endpoints_per_molecule": float(observed_per_molecule.mean()),
        "atom_count_min": int(atom_counts.min()),
        "atom_count_median": float(atom_counts.median()),
        "atom_count_mean": float(atom_counts.mean()),
        "atom_count_max": int(atom_counts.max()),
    }


def generate_eda_outputs(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    tasks: Sequence[str],
    table_dir: Path,
    figure_dir: Path,
    seed: int = 2026,
) -> list[Path]:
    """Write two aggregate tables and five deterministic EDA figures."""

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    endpoint_summary = build_endpoint_summary(frame, tasks)
    dataset_summary = build_dataset_summary(frame, audit, tasks)

    endpoint_path = table_dir / "endpoint_summary.csv"
    dataset_path = table_dir / "dataset_summary.json"
    endpoint_summary.to_csv(endpoint_path, index=False, lineterminator="\n")
    dataset_path.write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure_paths = [
        _plot_label_availability(endpoint_summary, figure_dir),
        _plot_prevalence(endpoint_summary, figure_dir),
        _plot_observed_endpoints(frame, tasks, figure_dir),
        _plot_atom_counts(frame, figure_dir),
        _plot_example_structures(frame, figure_dir, seed),
    ]
    return [endpoint_path, dataset_path, *figure_paths]


def _plot_label_availability(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(summary))
    negative = summary["negative"].to_numpy()
    positive = summary["positive"].to_numpy()
    missing = summary["missing"].to_numpy()
    ax.bar(x, negative, label="Negative", color="#4C78A8")
    ax.bar(x, positive, bottom=negative, label="Positive", color="#F58518")
    ax.bar(
        x,
        missing,
        bottom=negative + positive,
        label="Missing",
        color="#BAB0AC",
    )
    ax.set(title="Tox21 label availability", ylabel="Molecules")
    ax.set_xticks(x, summary["endpoint"], rotation=45, ha="right")
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir / "label_availability.png")


def _plot_prevalence(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(summary))
    prevalence = summary["positive_prevalence"]
    ax.bar(x, prevalence.fillna(0), color="#E45756")
    for index, value in enumerate(prevalence):
        if pd.isna(value):
            ax.text(index, 0.01, "NA", ha="center", va="bottom", fontsize=8)
    ax.set(
        title="Positive prevalence among observed labels",
        ylabel="Positive prevalence",
        ylim=(0, 1),
    )
    ax.set_xticks(x, summary["endpoint"], rotation=45, ha="right")
    return _save_figure(fig, output_dir / "positive_prevalence.png")


def _plot_observed_endpoints(
    frame: pd.DataFrame, tasks: Sequence[str], output_dir: Path
) -> Path:
    counts = frame[list(tasks)].notna().sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(-0.5, len(tuple(tasks)) + 1.5, 1)
    ax.hist(counts, bins=bins, color="#72B7B2", edgecolor="white")
    ax.set(
        title="Observed endpoints per molecule",
        xlabel="Observed endpoints",
        ylabel="Molecules",
    )
    return _save_figure(fig, output_dir / "observed_endpoints_per_molecule.png")


def _plot_atom_counts(frame: pd.DataFrame, output_dir: Path) -> Path:
    counts = _atom_counts(frame["standardized_smiles"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(counts, bins="auto", color="#54A24B", edgecolor="white")
    ax.set(title="Molecular atom counts", xlabel="Atoms", ylabel="Molecules")
    return _save_figure(fig, output_dir / "atom_count_distribution.png")


def _plot_example_structures(
    frame: pd.DataFrame, output_dir: Path, seed: int
) -> Path:
    sample_size = min(6, len(frame))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(frame), size=sample_size, replace=False))
    columns = 3
    rows = math.ceil(sample_size / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(9, 3 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indices, strict=False):
        record = frame.iloc[int(index)]
        molecule = Chem.MolFromSmiles(str(record["standardized_smiles"]))
        if molecule is None:
            raise ValueError(f"invalid processed SMILES at row {index}")
        axis.imshow(Draw.MolToImage(molecule, size=(320, 240)))
        axis.set_title(str(record.get("mol_id", f"row_{index}")), fontsize=9)
    fig.suptitle("Example standardized structures")
    return _save_figure(fig, output_dir / "example_structures.png")


def _atom_counts(smiles: pd.Series) -> pd.Series:
    counts: list[int] = []
    for index, value in smiles.items():
        molecule = Chem.MolFromSmiles(str(value), sanitize=True)
        if molecule is None:
            raise ValueError(f"invalid processed SMILES at row {index}")
        counts.append(molecule.GetNumAtoms())
    return pd.Series(counts, dtype="int64")


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
