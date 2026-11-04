# M21c Locked Three-Seed Ensemble Test

Date: 2026-07-18

## Protocol

- Task: single-chain effect-site residue prediction and chain effect-site ratio.
- Split: global pairwise DeepTMInter SI30, chain-filtered, no length limit.
- Structure source: exact-sequence ESMFold structures only; no UniProt, TmAlphaFold,
  AFDB, or PDBTM bridge.
- Training universe: 14,052 chains; validation: 3,796 chains; test: 2,392 chains.
- Residue checkpoints were selected by validation chain-macro AP in M21a.
- The detached chain-ratio heads were then calibrated with the residue models frozen.
  M21c checkpoints were selected only by validation negative ratio-head MAE.
- Test data were absent from all training and checkpoint-selection runs (`n_test=0`).
- Before test evaluation, the ensemble rule was fixed as an arithmetic mean of three
  residue probabilities and an arithmetic mean of three ratio-head probabilities.
- The classification threshold was selected on ensemble validation predictions only.
  No ensemble weights or thresholds were fitted on test.

## Implementation Checks

- Ratio-head-only implementation commit: `eae9d33`.
- Eval-only prediction-export commit: `6ad6f95`.
- The trainer and ensemble changes, plus all four execution configs, were narrowly
  reviewed with `gpt-5.3-codex-spark` at xhigh reasoning before execution.
- Pre-test smoke used 16 validation chains and one batch, with `n_test=0`.
- Smoke checkpoint loading reported zero missing and zero unexpected keys.
- Smoke produced only `predictions/val.pt`; `weights_only=True` loading succeeded.
- All formal evaluations reported zero parameter gradients and zero parameter-version
  changes.

## Validation Stability

| Seed | Val chain AP | Val AUPRC | Val AUROC | Val best F1 | Val ratio-head MAE | Val ratio-head bias |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.737981 | 0.766320 | 0.867293 | 0.696072 | 0.047686 | -0.010739 |
| 43 | 0.732968 | 0.757247 | 0.859102 | 0.682891 | 0.049370 | -0.008681 |
| 44 | 0.745334 | 0.762175 | 0.868864 | 0.688769 | 0.051618 | -0.008653 |
| Equal-weight ensemble | 0.751646 | 0.776927 | 0.874767 | 0.698305 | 0.048098 | -0.009358 |

The three independently selected heads therefore looked stable on validation. Their
mean single-seed validation MAE was about 0.04956.

## Locked Test Results

### Per-Seed Results

| Seed | Test chain AP | Test AUPRC | Test AUROC | F1 at val gate | MCC at val gate | Raw ratio MAE | Ratio-head MAE | Ratio-head bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.701097 | 0.749318 | 0.864950 | 0.661093 @ 0.58 | 0.550468 @ 0.58 | 0.062438 | 0.083438 | -0.063582 |
| 43 | 0.692480 | 0.731329 | 0.853728 | 0.648612 @ 0.42 | 0.530782 @ 0.42 | 0.065218 | 0.081246 | -0.061328 |
| 44 | 0.691962 | 0.737445 | 0.860673 | 0.663154 @ 0.39 | 0.535911 @ 0.39 | 0.064541 | 0.084475 | -0.063476 |

### Fixed Equal-Weight Ensemble

| Metric | Validation | Locked test |
|---|---:|---:|
| Chain-macro AP | 0.751646 | 0.706841 |
| Pooled AUPRC | 0.776927 | 0.753285 |
| AUROC | 0.874767 | 0.871132 |
| F1 at validation-selected gate | 0.698305 @ 0.45 | 0.669602 @ 0.45 |
| MCC at validation-selected gate | 0.573905 @ 0.45 | 0.554581 @ 0.45 |
| F1 at default 0.5 | 0.697933 | 0.660611 |
| MCC at default 0.5 | 0.578023 | 0.550745 |
| Raw residue-mean ratio MAE | 0.064271 | 0.061444 |
| Raw residue-mean ratio RMSE | 0.079919 | 0.091016 |
| Raw residue-mean ratio bias | +0.042921 | -0.002913 |
| Dedicated ratio-head MAE | 0.048098 | 0.082478 |
| Dedicated ratio-head RMSE | 0.066734 | 0.111902 |
| Dedicated ratio-head bias | -0.009358 | -0.062795 |
| Top-L/20 precision, chain macro | 0.883028 | 0.839007 |
| Top-L/10 precision, chain macro | 0.835064 | 0.790553 |

The test-oracle residue thresholds were 0.26 for maximum F1 and 0.28 for maximum
MCC. They are diagnostic only and must not be used as reportable operating points.

## Comparison With Existing Locked References

| Metric | M19a reference | M20a ratio-selected | M21c equal-3 | Delta vs M19a |
|---|---:|---:|---:|---:|
| Raw chain-ratio MAE | 0.073043 | 0.072577 | 0.061444 | -0.011599 |
| Raw chain-ratio RMSE | 0.105404 | 0.110874 | 0.091016 | -0.014388 |
| Chain-macro AP | 0.696629 | 0.685577 | 0.706841 | +0.010212 |
| Pooled AUPRC | 0.728543 | 0.721941 | 0.753285 | +0.024742 |
| AUROC | 0.854090 | 0.839948 | 0.871132 | +0.017042 |
| F1 at validation-selected gate | 0.651954 | 0.645265 | 0.669602 | +0.017648 |
| MCC at validation-selected gate | 0.530901 | 0.519377 | 0.554581 | +0.023680 |

M21c equal-3 improves raw test ratio MAE by about 15.9% relative to M19a, while also
improving all listed residue-ranking and validation-gated classification metrics. This
comparison is partly an ensemble comparison: M19a and M20a are single checkpoints,
whereas M21c equal-3 averages three independently trained residue models.

The current primary residue model is M18, a validation-locked hierarchical five-seed
ensemble, so that is the decisive comparison:

| Metric | M18 five-seed | M21c equal-3 | Delta (M21c - M18) |
|---|---:|---:|---:|
| Raw chain-ratio MAE | 0.070287 | 0.061444 | -0.008843 |
| Raw chain-ratio RMSE | 0.098664 | 0.091016 | -0.007648 |
| Raw chain-ratio bias | +0.036127 | -0.002913 | -0.039040 |
| Chain-macro AP | 0.714871 | 0.706841 | -0.008031 |
| Pooled AUPRC | 0.759486 | 0.753285 | -0.006201 |
| AUROC | 0.877876 | 0.871132 | -0.006744 |
| F1 at validation-selected gate | 0.674172 | 0.669602 | -0.004570 |
| MCC at validation-selected gate | 0.561137 | 0.554581 | -0.006555 |
| Top-L/20 precision, chain macro | 0.844631 | 0.839007 | -0.005624 |
| Top-L/10 precision, chain macro | 0.795269 | 0.790553 | -0.004716 |

M21c equal-3 improves raw ratio MAE by about 12.6% relative to M18, but is consistently
lower on residue ranking and classification. It therefore does not replace M18 as the
primary residue predictor.

A paired 43-component, 10,000-replicate bootstrap gives M21c-minus-M18 ratio MAE
`-0.008843` with 95% CI `[-0.022093, +0.006092]`. The interval includes zero, so the
lower M21c ratio-MAE point estimate is not statistically significant under the current
component-level uncertainty analysis.

## Interpretation

The three-seed residue ensemble is competitive and improves over the older M19a/M20a
locked references, but it does not surpass the current M18 five-seed ensemble. M18
remains the primary residue predictor. M21c equal-3 is retained as a ratio-focused
Pareto ablation because its raw chain-ratio MAE and RMSE are better. These residue
scores are not caused by phase-2 ratio-head calibration: that phase freezes every
residue-model parameter, so its residue scores remain unchanged.

The dedicated ratio head does not generalize. Although its validation MAE is stable
near 0.048-0.052, every seed shifts to roughly 0.081-0.084 MAE on test with a common
negative bias near -0.063. The equal-weight head therefore performs worse than simply
averaging the residue probabilities (0.082478 versus 0.061444 MAE). It must be reported
as a failed calibration ablation, not as the production ratio estimator.

The reportable M21 ratio output is the mean residue probability per chain. No post-test
affine correction, head reweighting, or threshold retuning is permitted from this
result. Promotion over M18 should be restricted to the ratio-MAE objective; M18 remains
the default model when residue-level prediction is the primary objective.

## Independent Result-to-Claim Gate

An independent high-reasoning reviewer returned `claim_supported: partial` with high
confidence. It supports reporting the favorable raw-ratio point estimate and the small
residue-performance gap. It rejects the claim that the detached ratio head improves
generalization, and it rejects a statistically established M21-over-M18 win.

The revised claim is:

> On this SI30 chain-filtered exact-sequence ESMFold split, M21 equal-3 has a favorable
> but not statistically significant raw residue-mean ratio MAE point estimate relative
> to M18, with slightly lower residue-ranking and classification metrics. The detached
> calibrated ratio head improves validation MAE but does not generalize to locked test.

Evidence still needed for a stronger claim includes a matched ensemble-size/weighting
comparison and an additional component-held-out or nested-validation evaluation that
has not been exposed during architecture development.

## Artifacts

- Ensemble metrics and provenance:
  `/data/husrcf/ProtBind/PPC/benchmark/results/chain_filtered_global_si30_20260709/ours_m21c_locked_equal3_20260718/metrics.json`
- Ensemble predictions:
  `/data/husrcf/ProtBind/PPC/benchmark/results/chain_filtered_global_si30_20260709/ours_m21c_locked_equal3_20260718/predictions.pt`
- M21c per-chain raw ratio:
  `benchmark_pipeline_tmp/m21_locked_test_export_20260718/results/effect_site_ratio/m21c_effect_site_ratio_per_chain.csv`
- M21c-minus-M18 component bootstrap:
  `benchmark_pipeline_tmp/m21_locked_test_export_20260718/results/effect_site_ratio/m21c_minus_m18_ratio_mae_component_bootstrap.json`
- Seed runs:
  `/home/husrcf/Code/PPC/runs/contact_site_cfsi30_m21c_locked_test_seed{42,43,44}_20260718_star`
- Reviewed configs and ensemble script:
  `/home/husrcf/Code/PPC/tmp/m21_locked_test_export_20260718`
