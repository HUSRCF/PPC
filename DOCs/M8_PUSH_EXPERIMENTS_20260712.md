# M8 Validation-Only Push Experiments (2026-07-12)

## Locked Baseline

M8 starts from M7c seed 42, selected only by validation chain-macro AP:

- chain-macro AP: `0.739431`
- pooled AP: `0.753759`
- architecture: M2 ESM-C MLC concat + one contact block + two gated depthwise TCN blocks
- TCN: kernel `7`, dilations `[1, 2]`
- loss: chain-mean weighted CE
- test visibility: hidden

## Screen Matrix

| Variant | Single controlled change | Rationale | Slurm job |
|---|---|---|---:|
| M8a | chain-mean -> residue-mean CE | Determine whether M7c's gain is TCN context rather than chain weighting | `9969976` |
| M8b | TCN 2 -> 3 blocks; dilations `[1,2,4]` | Increase contiguous sequence context without quadratic attention | `9969977` |
| M8c | TCN kernel 7 -> 11 | Widen local context while retaining two blocks | `9969978` |
| M8d | add ratio-MSE weight `0.1` | Directly align training with the single-chain effect-site-ratio objective | `9969979` |
| M8e | classifier dropout `0.45 -> 0.35` | Test whether TCN regularization permits a less underfit classifier | `9969980` |

All jobs use seed 42, 30 epochs, the optimized batch-64/token-65,536 loader, one A800-class GPU, and a 12-hour Slurm limit. At submission they were pending in the HPC2 priority queue; the scheduler estimate was 2026-07-14 01:16, but this can move when the four running M7 confirmation jobs finish.

Star was checked as an alternative. Both A100s were at 100% utilization with ESMFold/ScanNet work, so no duplicate M8 jobs were launched there.

## Ratio Objective

For chain `i`, the auxiliary prediction is the mean positive-class probability over valid residues:

`r_hat_i = mean_j p(y_ij = 1)`

The target is the observed positive-residue fraction `r_i`. M8d adds `0.1 * mean_i (r_hat_i - r_i)^2` to the existing chain-mean CE. A zero weight is bitwise identical to the pre-M8 loss implementation.

Validation now reports raw chain-ratio MAE, RMSE, and bias for every variant, even when the auxiliary loss is disabled.

## Selection Policy

1. Select each checkpoint by validation chain-macro AP.
2. Advance only if macro AP improves over M7c by at least `0.005`.
3. Require pooled AP to regress by no more than `0.003`.
4. Confirm advancing variants with seeds 43 and 44.
5. Freeze one winner and evaluate test once.

The dependent CPU summary job is `9970008`. It uses `afterany` on all five M8 jobs and writes:

`benchmark/reports/summary/esmc_matrix_20260712/m8_screen/`

It reads validation metrics from `best.pt` only and rejects checkpoints whose config shows `eval_test_each_epoch=true`.

## Leakage Controls

- `eval_test_each_epoch=false`: test IDs are not read and no test loader is constructed.
- Evaluation runs under `model.eval()` and `torch.no_grad()`.
- Runtime assertions require zero parameter gradients, non-gradient logits, and no parameter version change across evaluation.
- The assertions passed in live M7c seed-43/44 HPC2 training.
- Local split/cache audit passed across all 20,240 chains.
- Fresh 110-thread DeepTMInter SI30 recomputation found zero SI>=0.30 violations on both candidate routes.
- HPC2-local path audit job: `9970021`.

## Reproducibility

- Configs: `configs/experiments/20260712_m8_screen_hpc2/`
- Generator: `scripts/experiments/generate_m8_configs.py`
- Training implementation: `scripts/training/train_esm_site.py`
- Summary script: `scripts/analysis/summarize_m8_screen.py`
- Leakage report: `DOCs/DATASET_LEAKAGE_AUDIT_20260712.md`
