# Submission Notes

This repository is organized as a mentor-facing GSoC 2026 DeepLense evaluation submission.

## Start Here

1. Open `README.md` for the high-level overview.
2. Open `docs/gsoc26_evaluation_checklist.md` for a requirement-to-artifact map.
3. Open `reports/LENSFORGE_REPORT.md` for the consolidated technical summary.
4. Review the notebooks in `output/jupyter-notebook/`.

## Primary Artifacts

- Common Test I:
  - `output/jupyter-notebook/common-test-i-multiclass.ipynb`
  - `train_common_test_i.py`
  - `reports/common_test_i_experiments_compact.md`
- Optional Test IV extension:
  - `output/jupyter-notebook/test-iv-neural-operator.ipynb`
  - `train_test_iv_neural_operator.py`
  - `reports/test_iv_spectral.json`
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
- Consolidated summary:
  - `reports/LENSFORGE_REPORT.md`

## Current Best Recorded Results

- Test V:
  - validation ROC-AUC: `0.9753`
  - validation PR-AUC: `0.7043`
  - test ROC-AUC: `0.9659`
  - test PR-AUC: `0.3795`
  - test precision: `0.4082`
  - test recall: `0.5128`
  - selected baseline uses focal loss with PR-AUC-based model selection
- Common Test I:
  - best recorded validation accuracy: `0.6144`
  - best recorded validation macro ROC-AUC: `0.8333`
  - strongest retained run uses an explicit stratified `90:10` validation split with the polar-view CNN and no augmentation
- Optional Test IV spectral baseline:
  - validation accuracy: `0.3333`
  - validation macro ROC-AUC: `0.5245`

## Scope Note

The repository is organized around three reviewer-visible outcomes:

- a strong, fully documented Test V lens-finding baseline
- a complete Common Test I baseline with retained reference results
- a runnable mock-survey LSST pipeline that connects into the downstream DeepLense workflow
