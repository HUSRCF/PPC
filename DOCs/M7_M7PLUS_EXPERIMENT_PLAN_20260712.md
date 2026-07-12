# M7 / M7+ Validation-Locked Experiment Plan (2026-07-12)

## Objective

Improve single-chain residue effect-site ranking without changing the global-SI30
chain-filtered data universe or the exact-sequence ESMFold structure policy. Historical
test metrics are development context only. New checkpoint, architecture, blend, and
threshold decisions use validation data exclusively.

## Metric and Selection Protocol

1. Rank checkpoints and single-model variants by validation chain-macro AP.
2. Use validation pooled AP (`pr_auc` in the training log) as a guardrail and tie-breaker.
3. Select the operating threshold by pooled validation F1 only after the checkpoint is
   locked. Report MCC at that same frozen threshold.
4. Track validation raw ratio MAE as a task-specific diagnostic, but do not use it to
   change the residue-ranking model unless a separate ratio-calibration experiment is
   predeclared.
5. Keep `eval_test_each_epoch=false`; export and evaluate test scores only after all
   validation decisions are frozen.

F1 is retained as an operating-point metric rather than the checkpoint-ranking metric
because epoch-wise threshold maximization adds variance and is sensitive to calibration.
Chain-macro AP weights every chain equally and therefore matches the unseen-chain claim;
pooled AP prevents a gain driven only by short/easy chains.

## Historical Selection Audit

The existing seed-42 trajectories show that changing from validation F1 to validation
pooled AP selects a different epoch for M0, M1, M4, and M5, but not M2 or M3. Historical
test effects are mixed, so they cannot justify choosing either rule post hoc. This is why
the M7 rule is frozen from task semantics rather than from the observed test deltas.

## M6 Seed Robustness

M6 remains the original F1-selected protocol for comparability. Seeds 43 and 44 train
M0 and M2 once each. Every run additionally saves the best validation-F1, pooled-AP,
and chain-macro-AP checkpoints, so metric sensitivity can be audited without duplicate
training trajectories.

| Component | Seed | Slurm job | Test during training |
|---|---:|---:|---|
| M0 | 43 | 9967945 | hidden |
| M0 | 44 | 9967946 | hidden |
| M2 | 43 | 9967948 | hidden |
| M2 | 44 | 9967949 | hidden |

For each seed, select M0/M2 checkpoints by the original M6 validation-F1 rule, fit the
two-member logit blend and threshold on validation, and evaluate test once. Report the
three seed-specific M6 results as mean +/- standard deviation. Also form one seed-averaged
M6 by averaging family logits before validation-only blend/threshold selection.

The seed-specific runs and dependency-locked exports completed successfully. The
historical M6 protocol gives the following test mean +/- sample SD over seeds 42/43/44:

| Metric | Mean | SD |
|---|---:|---:|
| Frozen-threshold F1 | 0.668064 | 0.003192 |
| Frozen-threshold MCC | 0.546171 | 0.002318 |
| Pooled AP | 0.739351 | 0.003141 |
| AUROC | 0.867349 | 0.001058 |
| Chain-macro AP | 0.698436 | 0.001677 |
| Raw effect-site-ratio MAE | 0.102598 | 0.012451 |

F1/MCC and ranking metrics are reasonably stable, while raw score calibration is more
seed-sensitive. The seed-averaged-logit M6 remains a separate pending aggregate; the
table above is the mean of three independently validation-selected M6 models.

## M7 Single-Model Screen

| ID | Change | Question | Priority | Slurm job |
|---|---|---|---|---:|
| M7a | M2 clean; macro-AP checkpoint selection | Protocol control | Must run | 9968013 |
| M7b | M7a + per-chain mean weighted CE | Does chain-balanced training improve unseen-chain ranking? | Must run | 9968015 |
| M7c | M7b + two gated depthwise TCN blocks | Does cheap local sequence context improve motifs? | Must run | 9968055 |
| M7d | M5 gated ESM2/ESM-C fusion + per-chain mean CE | Does dual-representation fusion survive a fair loss/selection protocol? | Must run | 9968056 |
| M7e | M7b with one sparse graph-transformer layer replacing the simple contact block | Does richer nonlocal message passing help? | Conditional | not submitted |

All primary variants use seed 42 for screening, 30 epochs, the optimized loader profile,
chain-macro-AP checkpoint selection, secondary F1/AP checkpoints, and no per-epoch test.
M7e is launched only if M7b improves validation chain-macro AP by at least 0.005 without
a material pooled-AP regression.

The primary jobs were submitted on 2026-07-12 with a twelve-hour Slurm time limit. This
changes only the scheduler reservation and leaves every training parameter unchanged.

Promotion gate: advance a candidate only if validation chain-macro AP improves by at
least 0.005 over M7a and pooled AP does not decline by more than 0.002. Confirm the top
two candidates with seeds 43 and 44. Choose M7 by seed-mean validation chain-macro AP,
not by the best seed.

### Screening Result

All four seed-42 jobs completed without OOM, runtime exceptions, or non-finite skips.

| ID | Best epoch | Validation macro AP | Delta vs M7a | Validation pooled AP | Decision |
|---|---:|---:|---:|---:|---|
| M7a | 11 | 0.723554 | +0.000000 | 0.750322 | Control |
| M7b | 13 | 0.724197 | +0.000643 | 0.748572 | Reject |
| M7c | 20 | 0.739431 | +0.015877 | 0.753759 | Advance |
| M7d | 6 | 0.725383 | +0.001829 | 0.753857 | Reject |

Only M7c clears both gates. M7b shows that chain-balanced loss alone is insufficient;
the useful signal comes from the added gated local-context blocks. M7e is not triggered
because its prerequisite M7b gain was not met. Confirmation therefore compares M7c
against the M7a protocol control at seeds 43 and 44, without evaluating test.

| Confirmation | Seed | Slurm job | Test during training |
|---|---:|---:|---|
| M7a control | 43 | 9969262 | hidden |
| M7a control | 44 | 9969266 | hidden |
| M7c candidate | 43 | 9969267 | hidden |
| M7c candidate | 44 | 9969268 | hidden |

## M7+ Ensemble

M7+ is deliberately restricted to two simple candidates:

1. `M7+u`: unweighted logit mean of the locked M0 family and locked M7 family.
2. `M7+l`: one-parameter M0/M7 logit blend; choose alpha on validation chain-macro AP,
   then choose the threshold on validation F1.

Keep the learned blend only if it improves both validation chain-macro AP and pooled AP
over M7 and the unweighted blend, while frozen-threshold F1/MCC do not regress. Otherwise
publish the single model or unweighted blend. Do not use a stacking MLP, three-member
ensemble, per-chain threshold, or test-fitted calibration.

Implementation: `scripts/analysis/build_validation_logit_ensemble.py`. It evaluates the
full alpha grid on validation chain-macro AP, uses pooled validation AP only as a
tie-breaker, freezes alpha, then selects the operating threshold on validation F1. It
also emits the unweighted logit mean as a mandatory control and records an explicit
validation-only promotion decision before reporting test metrics.

## Seed and Uncertainty Protocol

- M6 robustness: seeds 42, 43, 44.
- M7 screening: seed 42.
- M7 confirmation: top two variants on seeds 42, 43, 44.
- Final expansion, if needed: add seeds 45 and 46 to the locked finalist only.
- Report mean +/- standard deviation across seeds on validation.
- On the final locked test pass, report 95% chain-bootstrap confidence intervals for
  F1, MCC, pooled AUPRC/AP, chain-macro AP, and ratio MAE.

## Compute Budget

Observed A800 wall times are approximately 0.7 GPU-hours for M0/M2 and 1.25 GPU-hours
for M5. The four new M6 seed jobs cost about 3 GPU-hours total. M7a-M7d screening should
cost approximately 4-5 GPU-hours; M7e adds roughly 1 GPU-hour only if promoted.

## Explicit Non-Goals

- No test-driven checkpoint, architecture, seed, blend, or threshold choice.
- No additional MSA or structure source; all structure-derived features remain ESMFold.
- No broad reopening of M1-M5 architecture search.
- No learned stacking model or ensemble larger than two model families.
- No claim that a single development split is a sealed holdout.
