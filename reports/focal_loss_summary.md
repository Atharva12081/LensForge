# BCE vs Focal Loss

| loss | threshold | val ROC-AUC | val PR-AUC | test ROC-AUC | test PR-AUC | precision | recall | fp | fn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bce | 0.95 | 0.8864 | 0.3121 | 0.8828 | 0.0969 | 0.0283 | 0.8500 | 584 | 3 |
| focal | 0.75 | 0.8741 | 0.2189 | 0.8641 | 0.0457 | 0.0474 | 0.6000 | 241 | 8 |

Conclusion: focal loss reduces false positives sharply, but the current setting hurts recall and both ROC-AUC / PR-AUC relative to BCE.
