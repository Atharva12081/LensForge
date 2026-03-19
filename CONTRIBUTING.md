# Contributing

Thank you for your interest in improving LensForge.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic Checks

```bash
python -m py_compile train.py train_common_test_i.py train_common_test_i_fft.py train_common_test_i_hog.py train_common_test_i_radial.py run_lsst_mock_pipeline.py
pytest -q
```

## Pull Requests

- keep changes focused and well documented
- update notebooks or docs when behavior changes
- avoid committing local caches, temporary files, or private datasets
- preserve reproducibility of saved reports and reviewer-facing artifacts
