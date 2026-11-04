# Completed Dropout Controls

K19 and K20 both completed50epochs, Slurm800313_0/1 COMPLETED exit0:0,
in41m09s/41m08s respectively. Every epoch covered all1,075 validation queries
and404,267 residues without validation skips. No test was evaluated.

All other hyperparameters matched K3. Main/head dropout was0.15/0.30 in
K19 and0.35/0.55 in K20, versus the original0.25/0.45. Checkpoints were
selected by validation pooled AP with the existing same-point macro guardrail.
Fixed ensembles used the predeclared equal thirds, without weight search.

| System | Epoch(s) | Macro AP | Pooled AP |
|---|---:|---:|---:|
| K19 lower dropout | 25 | 0.756893 | 0.786648 |
| K20 higher dropout | 13 | 0.758941 | 0.785894 |
| G2+MSA_D+K19 | 27+13+25 | 0.768562 | 0.799197 |
| G2+MSA_D+K20 | 27+13+13 | 0.766231 | 0.795878 |
| Current leader: G2+MSA_D+K18 | 27+13+31 | 0.768986 | 0.799705 |

Neither dropout change improves the current completed validation leader or
reaches0.80. K20's macro-best epoch12 has macro AP0.762960 but pooled AP
0.784242; do not combine it with epoch13's pooled AP. Training recorded
1/2AMP-skipped updates and4/3sanitized gradient elements for K19/K20.

These are selected-validation single-seed comparisons, not statistical proof
about dropout generally. The four other regularization arms are still running.
See the [terminal/source selection receipt](regularization_screen_20261105_K19_K20_resolved.json)
and [full validation scores and provenance](regularization_screen_20261105_K19_K20_results.json).
