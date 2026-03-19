from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_artifacts_exist() -> None:
    required_paths = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "requirements.txt",
        ROOT / "docs" / "submission_notes.md",
        ROOT / "docs" / "gsoc26_evaluation_checklist.md",
        ROOT / "output" / "jupyter-notebook" / "common-test-i-multiclass.ipynb",
        ROOT / "output" / "jupyter-notebook" / "deeplense-test-v-baseline.ipynb",
        ROOT / "output" / "jupyter-notebook" / "lsst-mock-pipeline.ipynb",
        ROOT / "reports" / "best_current_run.json",
        ROOT / "reports" / "common_test_i_fft.json",
        ROOT / "reports" / "lsst_mock_pipeline_run.json",
    ]
    for path in required_paths:
        assert path.exists(), f"Missing required artifact: {path}"


def test_notebooks_are_valid_json() -> None:
    notebook_dir = ROOT / "output" / "jupyter-notebook"
    for notebook_path in notebook_dir.glob("*.ipynb"):
        json.loads(notebook_path.read_text())


def test_best_run_contains_expected_metrics() -> None:
    report = json.loads((ROOT / "reports" / "best_current_run.json").read_text())
    assert "best_validation" in report
    assert "test" in report
    assert report["test"]["roc_auc"] > 0.8
