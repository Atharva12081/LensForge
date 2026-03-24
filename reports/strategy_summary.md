# Strategy Comparison

| strategy | best threshold | val ROC-AUC | val PR-AUC | test ROC-AUC | test PR-AUC | test recall |
|---|---:|---:|---:|---:|---:|---:|
| sampler | 0.75 | 0.8039 | 0.1416 | 0.8253 | 0.0355 | 0.9500 |
| loss | 0.90 | 0.8103 | 0.1421 | 0.8231 | 0.0343 | 0.7000 |
| both | 0.95 | 0.8864 | 0.3121 | 0.8828 | 0.0969 | 0.8500 |

This table is the earlier small-slice strategy comparison retained for context.
The current full-data Test V baseline used in LensForge is the focal-loss run in `reports/best_current_run.json`, which reaches test PR-AUC `0.3795`.
