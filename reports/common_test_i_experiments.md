# Common Test I Experiments

This repository keeps the strongest Common Test I comparison artifacts rather than the full scratch log.

## Retained baselines

| report | family | model | val acc | val macro ROC-AUC |
|---|---|---|---:|---:|
| common_test_i_fft.json | handcrafted | fft_radial_logreg | 0.3867 | 0.5587 |
| common_test_i_hog.json | handcrafted | hog_logreg | 0.3680 | 0.5317 |

Selected reference result:
- `common_test_i_fft.json`

Reason:
- it is the strongest recorded Common Test I result kept in LensForge
- it gives a compact, reproducible baseline for the required multi-class task
