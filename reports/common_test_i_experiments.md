# Common Test I Experiments

This repository keeps the strongest Common Test I comparison artifacts rather than the full scratch log.

## Retained baselines

| report | family | model | val acc | val macro ROC-AUC |
|---|---|---|---:|---:|
| common_test_i_polar_9010_noaug_long.json | neural | polar_cnn_9010_noaug_long | 0.6144 | 0.8333 |
| common_test_i_polar_9010.json | neural | polar_cnn_9010 | 0.5143 | 0.7560 |
| common_test_i_polar_9010_noaug.json | neural | polar_cnn_9010_noaug | 0.5013 | 0.7522 |
| common_test_i_polar.json | neural | polar_cnn | 0.4150 | 0.7209 |
| common_test_i_fft.json | handcrafted | fft_radial_logreg | 0.3867 | 0.5587 |
| common_test_i_hog.json | handcrafted | hog_logreg | 0.3680 | 0.5317 |

Selected reference result:
- `common_test_i_polar_9010_noaug_long.json`

Reason:
- it is the strongest recorded Common Test I result kept in LensForge
- it aligns directly with the written `90:10` validation guidance from the task folder
- it materially improves the retained multiclass baseline over the earlier polar and handcrafted floors
- it gives a compact, reproducible baseline for the required multi-class task
