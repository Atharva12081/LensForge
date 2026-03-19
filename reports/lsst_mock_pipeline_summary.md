# LSST Mock Pipeline Summary

LensForge now includes a runnable mock Rubin/LSST-style pipeline that packages survey-like inputs into the same folder structure consumed by the Test V lens-finding trainer.

## Pipeline run

- Runner: `run_lsst_mock_pipeline.py`
- Stages: `query -> fetch -> cutout -> preprocess -> package`
- Records packaged: `64`
- Split counts: `32 train`, `32 test`
- Class counts: `16` per folder for `train_lenses`, `train_nonlenses`, `test_lenses`, and `test_nonlenses`

## Downstream smoke validation

The packaged output was passed into `train.py` for a one-epoch end-to-end smoke run.

- Test ROC-AUC: `0.7188`
- Test PR-AUC: `0.6138`
- Test recall: `1.0000`

This is not a final performance benchmark. It is a structural validation that the pipeline output is immediately consumable by the existing DeepLense-style lens-finding workflow.
