# LensForge

LensForge is a self-contained GitHub submission for the GSoC 2026 DeepLense "Lens Finding & Data Pipelines" evaluation.

This repository includes:

- the required Common Test I multi-class classification deliverable
- the project-specific Test V lens-finding deliverable
- a runnable mock LSST-style data pipeline that feeds downstream DeepLense workflows

## Evaluation Scope

Train a binary classifier that separates lensed galaxies from non-lensed galaxies using:

- `data/lens-finding-test/train_lenses`
- `data/lens-finding-test/train_nonlenses`
- `data/lens-finding-test/test_lenses`
- `data/lens-finding-test/test_nonlenses`

Each sample is a normalized NumPy array with shape `(3, 64, 64)`.

## What is included

- `src/lens_finding_baseline.py`: dataset, model, training loop, evaluation helpers
- `train.py`: CLI entry point for training and evaluation
- `output/jupyter-notebook/deeplense-test-v-baseline.ipynb`: notebook scaffold for the submission workflow
- `output/jupyter-notebook/common-test-i-multiclass.ipynb`: scaffold for the required Common Test I deliverable
- `src/lsst_pipeline/`: runnable mock Rubin/LSST-style query, fetch, cutout, preprocess, and package stages
- `run_lsst_mock_pipeline.py`: entry point for packaging mock-survey inputs into DeepLense-ready folders
- `output/jupyter-notebook/lsst-mock-pipeline.ipynb`: notebook showing the LSST-style packaging workflow
- `docs/lsst_pipeline_design.md`: implementation note for the data-pipeline side of the project brief
- `docs/gsoc26_evaluation_checklist.md`: requirement-to-artifact checklist for the GSoC 2026 evaluation
- `docs/submission_notes.md`: quick guide for mentor review
- `LICENSE`: MIT license
- `requirements.txt`: Python dependency list
- `reports/`: curated experiment summaries and key JSON metrics
- `reports/lsst_mock_pipeline_summary.md`: compact end-to-end summary of the pipeline handoff into Test V
- `reports/focal_loss_summary.md`: direct BCE vs focal-loss comparison notes
- `train_common_test_i.py`: multiclass baseline trainer for Common Test I
- `train_common_test_i_radial.py`: handcrafted radial-feature baseline for Common Test I
- `train_common_test_i_fft.py`: FFT radial-feature baseline for Common Test I
- `train_common_test_i_hog.py`: HOG-feature baseline for Common Test I

## Quick start

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Then run the Test V baseline with:

```bash
python3 train.py \
  --data-root "data/lens-finding-test" \
  --epochs 5 \
  --batch-size 128 \
  --train-fraction 0.25 \
  --balance-strategy both \
  --test-fraction 0.25
```

This command uses a stratified 90:10 split on the training folders for validation and also reports metrics on the provided test folders.

Use `--test-fraction < 1.0` for quicker iteration during development, then switch back to `1.0` for a final report.

## Results snapshot

- Test V best recorded run:
  - validation ROC-AUC: `0.8864`
  - validation PR-AUC: `0.3121`
  - test ROC-AUC: `0.8828`
  - test PR-AUC: `0.0969`
  - test recall: `0.85`
- Common Test I best recorded run:
  - validation accuracy: `0.3867`
  - validation macro ROC-AUC: `0.5587`

## Mock LSST pipeline

Run the pipeline layer with:

```bash
python3 run_lsst_mock_pipeline.py \
  --data-root "data/lens-finding-test" \
  --output-root "tmp/lsst_mock_pipeline" \
  --max-per-folder 16 \
  --report-path "reports/lsst_mock_pipeline_run.json"
```

This packages a small mock-survey subset into:

```text
tmp/lsst_mock_pipeline/deeplense_dataset/
```

with the same `train_lenses`, `train_nonlenses`, `test_lenses`, and `test_nonlenses` layout consumed by `train.py`.

## Data layout

Keep the dataset inside the repository at:

```text
data/lens-finding-test/
```

with the four class folders under it.

Keep the Common Test I dataset inside the repository at:

```text
data/common-test-i/
  train/no
  train/sphere
  train/vort
  val/no
  val/sphere
  val/vort
```

## Baseline strategy

- Use a compact PyTorch CNN tailored to `(3, 64, 64)` inputs.
- Handle severe class imbalance with both `WeightedRandomSampler` and `BCEWithLogitsLoss(pos_weight=...)`.
- Tune the classification threshold on the validation split instead of assuming `0.5`.
- Report ROC-AUC as the main metric, plus PR-AUC, confusion matrix values, and loss curves.
- Save validation and test ROC/PR curve data into the run report for notebook visualization.
- Save the main Test V checkpoint to `models/best_current_run.pt`.

## Current status

- Common Test I is implemented with multiple baseline families and a dedicated notebook.
- Test V is implemented with imbalance handling, threshold tuning, error analysis, and saved reports.
- The LSST/data-pipeline side is represented by a runnable mock-survey packaging workflow plus a downstream smoke test.
- The explicit GSoC 2026 evaluation requirements are mapped in `docs/gsoc26_evaluation_checklist.md`.

## Review order

If you are evaluating the repository quickly, the recommended order is:

1. `docs/submission_notes.md`
2. `docs/gsoc26_evaluation_checklist.md`
3. `output/jupyter-notebook/deeplense-test-v-baseline.ipynb`
4. `output/jupyter-notebook/common-test-i-multiclass.ipynb`
5. `output/jupyter-notebook/lsst-mock-pipeline.ipynb`
