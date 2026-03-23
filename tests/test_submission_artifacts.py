from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_core_artifacts_exist() -> None:
    required_paths = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "requirements.txt",
        ROOT / "docs" / "submission_notes.md",
        ROOT / "docs" / "gsoc26_evaluation_checklist.md",
        ROOT / "docs" / "source_synthesis.md",
        ROOT / "output" / "jupyter-notebook" / "common-test-i-multiclass.ipynb",
        ROOT / "output" / "jupyter-notebook" / "deeplense-test-v-baseline.ipynb",
        ROOT / "output" / "jupyter-notebook" / "lsst-mock-pipeline.ipynb",
        ROOT / "output" / "jupyter-notebook" / "rubin-dp02-access.ipynb",
        ROOT / "output" / "jupyter-notebook" / "test-iv-neural-operator.ipynb",
        ROOT / "reports" / "best_current_run.json",
        ROOT / "reports" / "common_test_i_fft.json",
        ROOT / "reports" / "common_test_i_polar_9010_noaug_long.json",
        ROOT / "reports" / "lsst_mock_pipeline_run.json",
        ROOT / "reports" / "test_iv_spectral.json",
    ]
    for path in required_paths:
        assert path.exists(), f"Missing required artifact: {path}"


def test_notebooks_are_valid_json() -> None:
    notebook_dir = ROOT / "output" / "jupyter-notebook"
    for notebook_path in notebook_dir.glob("*.ipynb"):
        json.loads(notebook_path.read_text())
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        assert all("id" in cell for cell in notebook.cells)


def test_best_run_contains_expected_metrics() -> None:
    report = json.loads((ROOT / "reports" / "best_current_run.json").read_text())
    assert "best_validation" in report
    assert "test" in report
    assert report["test"]["roc_auc"] > 0.8


def _save_array(path: Path, value: float, channels: int) -> None:
    array = np.full((channels, 64, 64), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _build_binary_dataset(root: Path) -> None:
    for split, offset in {"train": 0.0, "test": 0.2}.items():
        for class_name, value in {"lenses": 0.8 + offset, "nonlenses": 0.2 + offset}.items():
            for idx in range(4):
                _save_array(root / f"{split}_{class_name}" / f"sample_{idx}.npy", value + idx * 0.01, 3)


def _build_multiclass_dataset(root: Path) -> None:
    class_values = {"no": 0.15, "sphere": 0.5, "vort": 0.85}
    for split, offset in {"train": 0.0, "val": 0.05}.items():
        for class_name, value in class_values.items():
            for idx in range(4):
                _save_array(root / split / class_name / f"sample_{idx}.npy", value + offset + idx * 0.01, 1)


def _smoke_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    mpl_dir = tmp_path / "mplconfig"
    cache_dir = tmp_path / "xdg-cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLBACKEND"] = "Agg"
    return env


def test_binary_training_cli_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "lens-finding-test"
    _build_binary_dataset(data_root)
    report_path = tmp_path / "reports" / "binary_smoke.json"
    model_path = tmp_path / "models" / "binary_smoke.pt"
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--data-root",
            str(data_root),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--validation-fraction",
            "0.5",
            "--report-path",
            str(report_path),
            "--model-path",
            str(model_path),
        ],
        cwd=ROOT,
        env=_smoke_env(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "test_auc=" in result.stdout
    payload = json.loads(report_path.read_text())
    assert payload["config"]["data_root"] == str(data_root)
    assert "best_validation" in payload
    assert "test" in payload


def test_common_test_i_training_cli_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "common-test-i"
    _build_multiclass_dataset(data_root)
    report_path = tmp_path / "reports" / "common_test_i_smoke.json"
    model_path = tmp_path / "models" / "common_test_i_smoke.pt"
    result = subprocess.run(
        [
            sys.executable,
            "train_common_test_i.py",
            "--data-root",
            str(data_root),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--model-type",
            "conv",
            "--report-path",
            str(report_path),
            "--model-path",
            str(model_path),
        ],
        cwd=ROOT,
        env=_smoke_env(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "val_auc=" in result.stdout
    payload = json.loads(report_path.read_text())
    assert payload["config"]["model_type"] == "conv"
    assert payload["best_validation"]["macro_roc_auc"] >= 0.0


def test_mock_pipeline_cli_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "lens-finding-test"
    _build_binary_dataset(data_root)
    output_root = tmp_path / "mock_pipeline"
    report_path = tmp_path / "reports" / "lsst_pipeline_smoke.json"
    subprocess.run(
        [
            sys.executable,
            "run_lsst_mock_pipeline.py",
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--max-per-folder",
            "2",
            "--report-path",
            str(report_path),
        ],
        cwd=ROOT,
        env=_smoke_env(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(report_path.read_text())
    assert payload["num_records"] == 8
    assert (output_root / "deeplense_dataset" / "train_lenses").exists()
    assert (output_root / "manifests" / "manifest.json").exists()
