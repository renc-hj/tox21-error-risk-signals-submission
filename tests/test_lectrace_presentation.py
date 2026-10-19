from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "presentation" / "_project_data.py"
LECTURE_PATH = ROOT / "presentation" / "01_tox21_error_risk.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("project_data", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load presentation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_demo_uses_project_split_contracts() -> None:
    helper = _load_helper()

    state = helper.build_split_demo()

    assert len(state.smiles) == 12
    assert state.fingerprints.shape == (12, 1_024)
    assert state.config.train_fraction == pytest.approx(0.8)
    assert state.config.validation_fraction == pytest.approx(0.1)
    assert state.config.test_fraction == pytest.approx(0.1)
    for groups, assignments in (
        (state.scaffolds, state.scaffold_assignments),
        (state.cluster_ids, state.cluster_assignments),
    ):
        grouped_partitions: dict[object, set[str]] = {}
        for group, partition in zip(groups, assignments, strict=True):
            grouped_partitions.setdefault(group, set()).add(str(partition))
        assert all(len(partitions) == 1 for partitions in grouped_partitions.values())

    presentation_state = helper.build_split_presentation_state()
    assert presentation_state["fingerprint_shape"] == (12, 1_024)
    assert "fingerprints" not in presentation_state


def test_uncertainty_demo_uses_project_signal_contracts() -> None:
    helper = _load_helper()

    state = helper.build_uncertainty_demo()

    assert state.labels.shape == (4, 2)
    assert state.single_probabilities.shape == (4, 2)
    assert state.mc_probabilities.shape == (50, 4, 2)
    assert state.single_pass_uncertainty[0, 0] == pytest.approx(0.1)
    assert state.is_error.sum() == pytest.approx(2.0)
    assert np.isfinite(state.mutual_information).all()

    presentation_state = helper.build_uncertainty_presentation_state()
    assert presentation_state["mc_dropout_passes_shape"] == (50, 4, 2)
    assert "mc_probabilities" not in presentation_state
    example = presentation_state["illustrative_endpoint"]
    assert example["true_label"] == 1
    assert example["deterministic_probability"] == pytest.approx(0.48)
    assert example["predicted_label_at_0.5"] == 0
    assert example["single_pass_uncertainty"] == pytest.approx(0.48)
    assert example["mc_dropout_mutual_information"] >= 0.0


def test_gcn_demo_exposes_the_reported_architecture_shape_transition() -> None:
    helper = _load_helper()

    state = helper.build_gcn_forward_demo()

    assert state["atom_feature_shape"] == (3, 166)
    assert state["gcn_layers"] == 3
    assert state["layer_output_shapes"] == [(3, 128), (3, 128), (3, 128)]
    assert state["pooled_molecule_shape"] == (1, 128)
    assert state["endpoint_probability_shape"] == (1, 12)
    assert len(state["endpoint_probabilities"]) == 12


def test_within_similarity_demo_recomputes_reported_test_summary() -> None:
    helper = _load_helper()

    rows = helper.load_within_similarity_summary()

    assert [row["split"] for row in rows] == [
        "random",
        "scaffold",
        "similarity-cluster",
    ]
    assert [row["single_pass_better_fraction"] for row in rows] == pytest.approx(
        [0.951, 0.895, 0.906]
    )


def test_headline_results_match_tracked_tables() -> None:
    helper = _load_helper()

    results = helper.load_headline_results()

    assert results["random"] == pytest.approx(
        {"single_pass": 0.778, "mc_dropout_mi": 0.669}, abs=0.0005
    )
    assert results["scaffold"] == pytest.approx(
        {"single_pass": 0.708, "mc_dropout_mi": 0.630}, abs=0.0005
    )
    assert results["similarity_cluster"] == pytest.approx(
        {"single_pass": 0.787, "mc_dropout_mi": 0.697}, abs=0.0005
    )


def test_lectrace_builds_one_coherent_presentation(tmp_path: Path) -> None:
    output = tmp_path / "lectrace"
    subprocess.run(
        [
            "uv",
            "run",
            "lectrace",
            "build",
            str(LECTURE_PATH),
            "--output",
            str(output),
            "--no-incremental",
        ],
        cwd=ROOT,
        check=True,
    )

    index = json.loads((output / "traces" / "index.json").read_text())
    traces = index["traces"]
    assert [trace["id"] for trace in traces] == ["01_tox21_error_risk"]
    assert 20 <= traces[0]["step_count"] <= 45
    for trace in traces:
        payload = json.loads((output / trace["path"]).read_text())
        assert payload["metadata"]["error"] is None


def test_built_traces_preserve_report_results_and_scope(tmp_path: Path) -> None:
    output = tmp_path / "lectrace"
    subprocess.run(
        [
            "uv",
            "run",
            "lectrace",
            "build",
            str(LECTURE_PATH),
            "--output",
            str(output),
            "--no-incremental",
        ],
        cwd=ROOT,
        check=True,
    )
    rendered_text: list[str] = []
    payload = json.loads(
        (output / "traces" / f"{LECTURE_PATH.stem}.json").read_text()
    )
    for step in payload["steps"]:
        rendered_text.extend(
            rendering["data"]
            for rendering in step["renderings"]
            if rendering["type"] in {"markdown", "note"}
        )
    narrative = " ".join(rendered_text)

    for phrase in (
        "Lecture 11 recording on responsible AI",
        "**Healthcare problem.**",
        "**Motivation.**",
        "**Approach.**",
        "prediction errors can be detected before deployment",
        "Report Figure 1",
        "Report Figure 2",
        "Report Figure 3",
        "Report Table 5",
        "threshold-aligned",
        "does not establish clinical toxicity or drug safety",
        "one fixed partition",
        "STATE 1",
        "STATE 2",
        "STATE 3",
        "STATE 4",
        "labels for 9.95 endpoints on average",
        "three $3\\times128$ hidden states",
        "true label 1",
        "not a selected test case",
        "would not replace either one",
    ):
        assert phrase in narrative
    for stale_reference in ("Report Figure 4", "Report Table 6"):
        assert stale_reference not in narrative
    for value in (
        "0.789",
        "0.724",
        "0.802",
        "0.778",
        "0.669",
        "0.708",
        "0.630",
        "0.787",
        "0.697",
        "95.1%",
        "89.5%",
        "90.6%",
    ):
        assert value in narrative
