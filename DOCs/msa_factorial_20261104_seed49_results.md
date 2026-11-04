# Seed49 direct-MSA screen: completed validation results

All six arms completed50 epochs. MSA Transformer jobs799171_12 and799171_13 exited COMPLETED/0 after30m49s and33m23s. Profile jobs799165_6 and799165_7 also completed. No test was evaluated.

Same frozen union-v2/OF3 contract:4,868 train and1,075 validation exact sequences;404,267 validation residues. Primary selection is validation chain-macro AP. Pooled AP, AUROC and pooled F1/MCC below come from that same checkpoint; F1 and MCC gates are individually validation-selected, not test-selected. Pooled AP and trapezoidal AUPRC differ only at about the sixth decimal here.

| Run | Epoch | Macro AP | Pooled AP | AUROC | Best pooled F1@gate | Best pooled MCC@gate | Ratio MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| D_none | 27 | 0.7282 | 0.7605 | 0.8717 | 0.6844@0.34 | 0.5687@0.41 | 0.0552 |
| E_none | 19 | 0.7286 | 0.7548 | 0.8695 | 0.6819@0.53 | 0.5613@0.66 | 0.0915 |
| D_profile | 38 | 0.7295 | 0.7520 | 0.8667 | 0.6778@0.31 | 0.5621@0.52 | 0.0596 |
| E_profile | 19 | 0.7274 | 0.7562 | 0.8708 | 0.6808@0.28 | 0.5629@0.53 | 0.0585 |
| D_encoder | 13 | 0.7343 | 0.7608 | 0.8752 | 0.6838@0.50 | 0.5625@0.68 | 0.0746 |
| E_encoder | 18 | 0.7331 | 0.7585 | 0.8733 | 0.6842@0.48 | 0.5642@0.69 | 0.0757 |

## Interpretation

- D + MSA Transformer versus D without MSA: macro AP+0.0061102; pooled AP+0.0002548 at the macro-selected checkpoint. This is a modest ranking signal, with pooled AP essentially unchanged, not a proven cross-seed gain.
- E + MSA Transformer versus E without MSA: macro AP+0.0045338; pooled AP+0.0036930 at the macro-selected checkpoint.
- The25D MSA profile does not improve both AP objectives consistently. D profile loses pooled AP despite a small macro increase; E profile loses macro AP despite a small pooled increase.
- Transformer ratio MAE is worse than D no-MSA; MAE remains diagnostic and must not replace the AP-based selection policy.
- Separate pooled-AP selection gives E encoder epoch28: pooled AP0.7635955, macro AP0.7301978. Its macro value must not be mixed with the epoch18 value0.7331230. D encoder pooled-best epoch25 gives pooled AP0.7613738 but macro AP0.7286444.
- All arms have zero skipped/nonfinite validation batches. Dynamic AMP skipped1--3 optimizer updates per run; these are recorded, not silently treated as failed runs.

This is a single-seed augmented-system comparison. New adapters add parameters and consume constructor RNG; same seed is not a parameter-matched initialization control. No statistical superiority or final model replacement is claimed.

Raw run directories, config and metrics SHA-256, both selection points and full compact metrics are in msa_factorial_20261104_seed49_results.json. Existing seed50/51 acd_u jobs remain pending; no extra seed jobs were submitted for this analysis.

