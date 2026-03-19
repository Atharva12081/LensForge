# LensForge GSoC 2026 Evaluation Checklist

This checklist maps the explicit GSoC 2026 DeepLense evaluation requirements to completed LensForge artifacts.

## Common Test I

Requirement:
- complete the common multi-class classification test
- provide a notebook
- report ROC/AUC on validation data

Covered by:
- `train_common_test_i.py`
- `src/common_test_i.py`
- `train_common_test_i_radial.py`
- `train_common_test_i_fft.py`
- `train_common_test_i_hog.py`
- `output/jupyter-notebook/common-test-i-multiclass.ipynb`
- `reports/common_test_i_experiments.md`
- `reports/common_test_i_experiments_compact.md`

## Test V: Lens Finding & Data Pipelines

Requirement:
- train on `train_lenses` and `train_nonlenses`
- evaluate on `test_lenses` and `test_nonlenses`
- address class imbalance
- report ROC/AUC on validation data
- provide a notebook

Covered by:
- `train.py`
- `src/lens_finding_baseline.py`
- `output/jupyter-notebook/deeplense-test-v-baseline.ipynb`
- `reports/best_current_run.json`
- `reports/strategy_summary.md`
- `reports/focal_loss_summary.md`
- `reports/error_summary.md`

## Jupyter Notebooks

Requirement:
- submit Jupyter notebooks for each evaluation deliverable

Covered by:
- `output/jupyter-notebook/common-test-i-multiclass.ipynb`
- `output/jupyter-notebook/deeplense-test-v-baseline.ipynb`
- `output/jupyter-notebook/lsst-mock-pipeline.ipynb`

## Validation Metrics

Requirement:
- calculate and present required evaluation metrics for validation data with a 90:10 split

Covered by:
- `train.py` with `--validation-fraction 0.1`
- `train_common_test_i.py` using the provided `train/val` structure
- saved validation metrics in `reports/*.json`
- notebook visualizations in both task notebooks

## Trained Weights

Requirement:
- include trained weights if any

Covered by:
- `models/*.pt`

## Repository Submission

Requirement:
- provide a GitHub repository containing the solution

Covered by:
- self-contained LensForge folder structure
- repo-local `data/` placeholders
- repo-local paths in README, notebooks, and human-facing summaries

## LSST Data Pipeline Project Brief

Requirement:
- functional pipeline capable of interfacing LSST-style data with DeepLense applications
- test the workflow on mock surveys

Covered by:
- `src/lsst_pipeline/`
- `run_lsst_mock_pipeline.py`
- `output/jupyter-notebook/lsst-mock-pipeline.ipynb`
- `reports/lsst_mock_pipeline_run.json`
- `reports/lsst_mock_pipeline_summary.md`
- `docs/lsst_pipeline_design.md`

## Completion Status

LensForge covers every explicit GSoC 2026 DeepLense evaluation deliverable addressed in this submission:

- Common Test I
- Test V
- notebooks
- validation metrics
- trained weights
- repository packaging
- LSST/mock-survey workflow artifact
