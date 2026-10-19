from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-lectrace.yml"


def test_github_pages_workflow_deploys_only_lectrace() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    required = (
        "pages: write",
        "id-token: write",
        "uv sync --frozen --group presentation",
        "lectrace build presentation/01_tox21_error_risk.py --output _site",
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v3",
        "path: _site",
        "actions/deploy-pages@v4",
    )
    for phrase in required:
        assert phrase in workflow

    assert "website" not in workflow
    assert "poster" not in workflow
    assert "scripts/11_build_showcase.py" not in workflow
