# Submission Notes

This repository is organized as a mentor-facing GSoC 2026 DeepLense evaluation submission.

## Start Here

1. Open `README.md` for the high-level overview.
2. Open `docs/gsoc26_evaluation_checklist.md` for a requirement-to-artifact map.
3. Review the notebooks in `output/jupyter-notebook/`.

## Primary Artifacts

- Common Test I:
  - `output/jupyter-notebook/common-test-i-multiclass.ipynb`
  - `train_common_test_i.py`
  - `reports/common_test_i_experiments_compact.md`
- Test V:
  - `output/jupyter-notebook/deeplense-test-v-baseline.ipynb`
  - `train.py`
  - `reports/best_current_run.json`
  - `reports/error_summary.md`
- LSST/data pipeline:
  - `output/jupyter-notebook/lsst-mock-pipeline.ipynb`
  - `run_lsst_mock_pipeline.py`
  - `docs/lsst_pipeline_design.md`
  - `reports/lsst_mock_pipeline_summary.md`

## Current Best Recorded Results

- Test V:
  - validation ROC-AUC: `0.8864`
  - validation PR-AUC: `0.3121`
  - test ROC-AUC: `0.8828`
  - test PR-AUC: `0.0969`
  - test recall: `0.85`
- Common Test I:
  - best recorded validation accuracy: `0.3867`
  - best recorded validation macro ROC-AUC: `0.5587`

## Scope Note

The repository is organized around three reviewer-visible outcomes:

- a strong, fully documented Test V lens-finding baseline
- a complete Common Test I baseline with retained reference results
- a runnable mock-survey LSST pipeline that connects into the downstream DeepLense workflow
