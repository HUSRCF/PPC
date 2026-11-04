# Single-Chain Model Results

Updated: 2026-11-05. The previous repository integration omitted the intermediate
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
| D/E x direct-MSA: seed49 completed, seed50/51 still queued | [Completed seed49 results](msa_factorial_20261104_seed49_results.md), [protocol](MSA_FACTORIAL_20261104.md), [submitted jobs](msa_factorial_20261104_launch.json) |
| Fixed validation MSA blends and per-sequence/component analysis | [Results and interpretation](msa_complementarity_20261104.md), [metrics/bootstrap](msa_complementarity_20261104/summary.json), [per-sequence table](msa_complementarity_20261104/per_chain_deltas.csv) |
| Completed OF3 geometry/global-context screen | [Final matrix and C1 numerical caveat](geometry_screen_20261104_results.md), [full metrics and hashes](geometry_screen_20261104_results.json), [protocol](geometry_screen_20261104.md) |
| Completed post-MSA BiGRU / fusion-position / capacity controls and train AP | [Final results](msa_context_screen_20261104_results.md), [metrics and hashes](msa_context_screen_20261104_results.json), [protocol](msa_context_screen_20261104.md) |
| Gated DeltaNet feasibility; no installation or training yet | [Proposed bidirectional adaptation and dependency audit](deltanet_feasibility_20261104.md) |
| Geometry x MSA combinations toward pooled AP0.80, new seed49 screen | [Nine-arm plan](geometry_msa_screen_20261104_round2.md), [launch/provenance](geometry_msa_screen_20261104_round2_launch.json) |
| Fixed G2 plus MSA validation ensembles | [Completed results](geometry_msa_screen_20261104_fixed_blends_results.md), [full metrics and hashes](geometry_msa_screen_20261104_fixed_blends_results.json) |
| Evolution-conditioned geometry attention, K9/K10 | [Two-arm design](evolutionary_geometry_edges_20261105.md), [launch/provenance](evolutionary_geometry_edges_20261105_launch.json) |
| Completed K1 versus K9 one-layer edge-bias control | [Result and checkpoint-selection caveat](evolutionary_geometry_edges_20261105_K1_K9_results.md), [frozen metrics and hashes](evolutionary_geometry_edges_20261105_K1_K9_results.json) |
| Geometry depth x ratio loss, K11-K13 | [Design](geometry_depth_ratio_20261105.md), [initial failures and strict retry mapping](geometry_depth_ratio_20261105_launch.json) |
| DeltaNet dependency dry run; installation approval pending | [Plan and constraints](deltanet_20261105/README.md), [exact19-package proposal](deltanet_20261105/dry_run_summary.json) |
| Current pooled-AP0.80 campaign snapshot | [Observed statuses and same-checkpoint metrics](pooled_AP080_status_20261105.json), [K5 host-memory retry](geometry_msa_K5_host_memory_retry_20261105.json), [two planned fixed follow-up blends](geometry_msa_followup_blends_20261105.json) |
| Completed-model ranking diagnostics: 11 points, 55 SI components | [Interpretation and subgroup results](validation_diagnostics_20261105/README.md), [conditional bootstrap](validation_diagnostics_20261105/summary.json), [per-chain table](validation_diagnostics_20261105/per_chain.csv) |
| Completed K2/K10-K13 follow-ups and depth x loss matrix | [Same-checkpoint table and limits](geometry_completed_followups_20261105.md), [metrics, source commits and hashes](geometry_completed_followups_20261105.json) |
| Completed fixed K3 ensembles: pooled AP0.798613, macro AP0.766068 | [Results and follow-up hypotheses](geometry_msa_followup_blends_20261105_results.md), [metrics/source hashes](geometry_msa_followup_blends_20261105_results.json), [resolved policy](geometry_msa_followup_blends_20261105_resolved.json) |
| MSA bottleneck width versus matched capacity, K14-K17 | [Four-arm design](msa_bottleneck_width_20261105.md), [job800186 and startup receipts](msa_bottleneck_width_20261105_launch.json), [prospective fixed-blend policy](msa_bottleneck_width_20261105_fixed_blends.json) |
| Fusion-position x geometry-depth missing cell, K18 | [Controlled design](geometry_position_depth_20261105.md), [job800231 and startup receipts](geometry_position_depth_20261105_launch.json), [fixed-blend policy](geometry_position_depth_20261105_fixed_blend.json) |
| Fixed K3 ensembles: exact raw-score reproduction and 55-component uncertainty | [Interpretation](validation_ensemble_diagnostics_20261105/README.md), [effects and intervals](validation_ensemble_diagnostics_20261105/summary.json), [integrity receipts](validation_ensemble_diagnostics_20261105/reproduction_receipt.json) |
| AP0.80 milestone: validation lock, test evaluation, then whole-RCSB | [User contract and leakage safeguards](pooled_AP080_milestones_20261105.md), [machine-readable policy](pooled_AP080_milestones_20261105_policy.json) |
| Completed K15/K16 and first fixed width blends | [Readout and uncompleted matched-pair caveat](msa_bottleneck_width_20261105_K15_K16_results.md), [full metrics](msa_bottleneck_width_20261105_K15_K16_results.json) |

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

The current fixed seed49 validation blend D_none e27 + D_encoder e13 reaches
macro AP 0.741355 and pooled AP 0.772186 on the 1,075-sequence union validation
set. It is an exploratory validation candidate, not a new frozen test result.
Its component bootstrap is conditional on already-selected validation checkpoints
and does not include checkpoint-selection correction or seed uncertainty.

See [history scope and provenance](results_history_20261104/README.md).
