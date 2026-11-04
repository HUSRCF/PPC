# Single-Chain Model Results

Updated: 2026-11-04. The previous repository integration omitted the intermediate
result summaries, although the source code and latest union experiments were
included. This index closes that gap. M7c is not the latest completed model.

## Where to Look

| Results | Evidence and location |
|---|---|
| M0-M6, original M18 and M21C test progression | [Frozen progression table](results_history_20261104/sources/progression/model_progression_validation_locked.md) and [CSV](results_history_20261104/sources/progression/model_progression_validation_locked.csv) |
| M7a versus M7c, seeds 42/43/44 | [Completed confirmation](results_history_20261104/sources/hpc2/m7_confirmation/m7_confirmation_summary.md) |
| M8a-M8e validation screen | [Completed screen](results_history_20261104/sources/hpc2/m8_screen/m8_screen_summary.md) |
| M9-M21 training and eval exports | [62-record catalog](results_history_20261104/STAR_RUNS.md), [CSV](results_history_20261104/star_m9_m21_validation.csv), [source hashes/configs](results_history_20261104/star_m9_m21_runs.json) |
| M14 conditional-gain/stacking study | [Frozen analysis](results_history_20261104/sources/m14/M14_CONDITIONAL_GAIN_SUMMARY.md) |
| Original M18 five-seed final result | [Frozen test summary](results_history_20261104/sources/m18/locked_five_seed_test_summary.json), [validation selection](results_history_20261104/sources/m18/validation_selection_summary.json), [checkpoint paths/hashes](results_history_20261104/sources/m18/CHECKPOINTS.tsv) |
| M21C three-seed ratio-head variant | [Frozen report](results_history_20261104/sources/m21c/M21C_LOCKED_ENSEMBLE_TEST_20260718.md), [metrics](results_history_20261104/sources/m21c/metrics.json), [paired uncertainty](results_history_20261104/sources/m21c/m21c_minus_m18_ratio_mae_component_bootstrap.json) |
| August matched-seed feature ablation A-E | [Metrics](results_history_20261104/sources/controlled_ablation/controlled_ablation_metrics.csv), [supported conclusions](results_history_20261104/sources/controlled_ablation/findings.md) |
| November exact-sequence union + OpenFold3 A-E and R0-R2 | [Methods and results](m18_union_of3_20261104/README.md), [checkpoint/feature index](m18_union_of3_20261104/asset_index.json) |
| Active D/E x direct-MSA, three-seed follow-up (no final results yet) | [Protocol](MSA_FACTORIAL_20261104.md), [submitted jobs and acceptance](msa_factorial_20261104_launch.json) |

## Original Frozen Test Results

The following are **historical context-specific labels with ESMFold**, not the
new exact-sequence union labels. Validation has 3,796 chain contexts; test has
2,392 chain contexts / 1,036,256 residues / 43 SI30 components.
Gates and ensemble rules were fixed from validation for the cited evaluation.

| Model | Test AUPRC | Test AUROC | Test chain-macro AP | Test F1@val gate | Test MCC@same gate | Residue-mean ratio MAE |
|---|---:|---:|---:|---:|---:|---:|
| M6 validation-selected logit blend | 0.741125 | 0.866190 | 0.699985 | 0.666093@0.52 | 0.544013 | - |
| M18 hierarchical five-seed | **0.759486** | **0.877876** | **0.714871** | **0.674172@0.61** | **0.561137** | 0.070287 |
| M21C fixed equal-three | 0.753285 | 0.871132 | 0.706841 | 0.669602@0.45 | 0.554581 | 0.061444 |

- Original M18 remains the primary frozen residue-prediction reference; it is not
  replaced by the latest validation-only experiment merely because it ran later.
- M18 validation chain-macro AP is 0.764811; validation AUPRC is 0.787666.
- M21C has a lower ratio-MAE point estimate, but its paired 43-component,
  10,000-replicate difference is -0.008843, CI95 [-0.022093, +0.006092].
  This does **not** establish a statistically resolved improvement.
- These are model-progression comparisons, not controlled architecture ablations.
  Seed counts, ensembling and some feature/model choices differ.
- This test set has been inspected repeatedly during the wider project. A
  validation-locked operating point does not retroactively make it a sealed holdout.
- Raw archived summaries retain test-oracle diagnostics for audit only. Those
  gates are not used in this table, model selection or promotion.

## Latest Experiments Are a Separate Contract

November union/OF3 runs use 6,667 exact sequences, split into 4,868 train / 1,075
validation / 724 test sequences. Labels are OR-merged across retained assessed
contexts; predicted structures are OpenFold3 rather than ESMFold.

Completed loss-matrix A-E and auxiliary-head R0-R2 results are **validation-only**.
Their macroAP values around 0.727-0.731 cannot be compared directly to the old
3,796-context validation or 2,392-context test numbers above. No new test evaluation
or training was performed to assemble this index.

The August controlled ablation also calls its variants A-E, but those are
feature-removal experiments. They are not the November loss-matrix A-E.

See [history scope and provenance](results_history_20261104/README.md).
