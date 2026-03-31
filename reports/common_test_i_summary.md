# Common Test I Summary

LensForge includes several Common Test I baselines, but the selected retained result is the stratified `90:10` polar-view CNN run.

Selected result:

- validation accuracy: `0.6144`
- validation macro ROC-AUC: `0.8333`
- report: `reports/common_test_i_polar_9010_noaug_long.json`
- model weight: `models/common_test_i_polar_9010_noaug_long.pt`

Why this baseline is retained:

- the polar transform is a geometry-aware representation for ring-like and arc-like lensing structure
- it outperformed the earlier small image-space CNN runs kept in the repository
- it also outperformed the retained handcrafted FFT, radial, and HOG reference floors

Interpretation:

- Common Test I is complete and reproducible in LensForge
- its current result is weaker than the repository's Test V result
- the most promising direction found so far is morphology-aware preprocessing rather than larger image-space CNNs alone
