# BCE vs Focal Loss

| loss | threshold | val ROC-AUC | val PR-AUC | test ROC-AUC | test PR-AUC | precision | recall | fp | fn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bce | 0.95 | 0.8864 | 0.3121 | 0.8828 | 0.0969 | 0.0283 | 0.8500 | 584 | 3 |
| focal | 0.90 | 0.9753 | 0.7043 | 0.9659 | 0.3795 | 0.4082 | 0.5128 | 145 | 95 |

Conclusion: on the final full-data Test V run, focal loss becomes the strongest ranking setup in LensForge and is now the selected baseline.
