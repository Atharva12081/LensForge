# Common Test I Compact Comparison

Selected Common Test I baselines retained in the repository:

| rank | report | family | model | val acc | val macro ROC-AUC |
|---:|---|---|---|---:|---:|
| 1 | common_test_i_polar_9010_noaug_w32.json | neural | polar_cnn_9010_noaug_w32 | 0.9141 | 0.9849 |
| 2 | common_test_i_polar_9010_noaug_long.json | neural | polar_cnn_9010_noaug_long | 0.6144 | 0.8333 |
| 3 | common_test_i_polar_9010.json | neural | polar_cnn_9010 | 0.5143 | 0.7560 |
| 4 | common_test_i_polar_9010_noaug.json | neural | polar_cnn_9010_noaug | 0.5013 | 0.7522 |
| 5 | common_test_i_polar.json | neural | polar_cnn | 0.4150 | 0.7209 |

The selected Common Test I reference result in LensForge is now the wider stratified `90:10` polar-view CNN without augmentation, trained on the full combined train+val pool.
