# Common Test I Summary

LensForge includes several Common Test I baselines, but the selected retained result is now the wider stratified `90:10` polar-view CNN run trained on the full combined train+val pool.

Selected result:

- validation accuracy: `0.9141`
- validation macro ROC-AUC: `0.9849`
- report: `reports/common_test_i_polar_9010_noaug_w32.json`
- model weight: `models/common_test_i_polar_9010_noaug_w32.pt`

Why this baseline is retained:

- the polar transform is a geometry-aware representation for ring-like and arc-like lensing structure
- a wider `width=32` polar backbone with cosine decay and full-data `90:10` validation splitting substantially outperformed the earlier retained runs
- it also outperformed the retained handcrafted FFT, radial, HOG, and earlier smaller polar reference baselines

Interpretation:

- Common Test I is now both complete and genuinely competitive in LensForge
- the most promising direction found so far is morphology-aware preprocessing paired with a stronger polar classifier trained on the full available pool
- this closes much of the earlier gap between the Common Test I and Test V portions of the repository
