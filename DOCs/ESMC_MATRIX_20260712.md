# ESM-C Representation Matrix (2026-07-12)

## Scope and status

This experiment compares ESM2 and ESM-C residue representations inside the same PPC
residue classifier. Formal M0-M5 runs completed on HPC2 A800 GPUs. Epoch and operating
threshold selection use validation F1 only. M6 is the predeclared validation-selected
M0/M2 logit ensemble; its score export and frozen test replay also completed.

Data universe: the chain-filtered DeepTMInter-style global-SI30 split, with 14,052 train,
3,796 validation, and 2,392 test chains. The test set has 1,036,256 residues. Structure
scalar features come only from exact-sequence ESMFold predictions; no PDBTM -> UniProt ->
AlphaFold/TmAlphaFold bridge is used. Every variant uses the same frozen ESM2 contact
graph policy (`top_k=16`, minimum score 0.05, minimum sequence separation 6,
bidirectional).

## Variants

| ID | Representation |
|---|---|
| M0 | ESM2 layers 11/22/33 concatenated (3,840D) |
| M1 | ESM-C final block 36 (1,152D) |
| M2 | ESM-C blocks 12/24/36 concatenated (3,456D) |
| M3 | Learned scalar mix of ESM-C blocks 12/24/36 |
| M4 | Parameter-matched projected concat: ESM2->128 + ESM-C->128 |
| M5 | Full ESM2/ESM-C projections with gated residual fusion |
| M6 | Validation-selected logit blend of M0 and validation-winning M2 |

## Formal results

The principal F1/MCC columns below use the validation-selected epoch and validation-
selected threshold. `Oracle F1` is retained only as a diagnostic and is not used for
model or threshold selection.

| Variant | Epoch | Val F1@thr | Test F1@val-thr | Test MCC@val-thr | Test AUPRC | Test AUROC | F1@0.5 | F1@0.6 | Oracle F1@thr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 17 | 0.6665@0.58 | 0.6372 | 0.5040 | 0.7003 | 0.8454 | 0.6495 | 0.6332 | 0.6555@0.35 |
| M1 | 9 | 0.6765@0.48 | 0.6510 | 0.5201 | 0.7092 | 0.8485 | 0.6475 | 0.6155 | 0.6578@0.39 |
| M2 | 17 | **0.6835@0.51** | 0.6425 | 0.5123 | 0.7136 | 0.8423 | 0.6438 | 0.6330 | 0.6557@0.29 |
| M3 | 24 | 0.6728@0.52 | **0.6585** | 0.5218 | **0.7292** | 0.8569 | 0.6599 | 0.6512 | 0.6614@0.41 |
| M4 | 15 | 0.6750@0.45 | 0.6414 | 0.5134 | 0.7146 | 0.8509 | 0.6330 | 0.6132 | 0.6627@0.23 |
| M5 | 6 | 0.6813@0.60 | 0.6536 | **0.5250** | 0.7245 | **0.8594** | **0.6639** | **0.6536** | **0.6675@0.42** |

M5 versus M0 at the frozen validation threshold is +0.0164 F1 and +0.0210 MCC;
threshold-independent deltas are +0.0242 AUPRC and +0.0140 AUROC. These are test point
estimates, not the selection rule. M2 has the highest validation F1 and is therefore the
ESM-C member used by M6. Selecting M3 or M5 after inspecting test would be test leakage.

## M6 validation-only ensemble

Validation selected `0.45 * logit(M0) + 0.55 * logit(M2)` and threshold 0.52. These
values were frozen before test replay.

| Metric | M0 | M2 | M6 |
|---|---:|---:|---:|
| Test F1 at validation threshold | 0.6372 | 0.6425 | **0.6661** |
| Test MCC at validation threshold | 0.5040 | 0.5123 | **0.5440** |
| Test AUPRC | 0.7003 | 0.7136 | **0.7411** |
| Test AUROC | 0.8454 | 0.8423 | **0.8662** |
| F1 / MCC at 0.5 | 0.6495 / 0.5081 | 0.6438 / 0.5130 | **0.6684 / 0.5443** |
| F1 / MCC at 0.6 | 0.6332 / 0.5025 | 0.6330 / 0.5113 | **0.6522 / 0.5383** |

M6 also reaches L/20, L/10, and L/5 chain-macro accuracies of 0.8304, 0.7813, and
0.6936. Its raw mean-score single-chain effect-site-ratio MAE is 0.0894; thresholded
ratio MAE at the frozen 0.52 operating point is 0.0803. Diagnostic test-oracle values
are F1 0.6816 at 0.3568 and MCC 0.5479 at 0.408; neither was used for selection.

For the project's direct raw single-chain ratio target, M6 has MAE/RMSE/bias
0.08941/0.11148/+0.04211, Pearson 0.81367, and Spearman 0.84423. Its MAE point estimate
is lower than ScanNet MSA (0.09157) and PeSTo (0.09996), but paired 43-component,
10,000-replicate bootstrap intervals for M6-minus-baseline MAE are respectively
[-0.02635, 0.01576] and [-0.03176, 0.01470]. Both include zero. The defensible claim is
best raw MAE point estimate on this development universe, not a statistically significant
MAE win. Full per-chain and bootstrap artifacts are under `m6/effect_site_ratio/`.

## Control audit

The earlier high-dropout baseline was reported as test F1 0.6596. That number is the
test-oracle value at threshold 0.40. Its validation-frozen test F1/MCC are 0.63750/0.50782
at validation threshold 0.62. Formal M0 gives 0.63723/0.50402 at validation threshold
0.58, with AUPRC 0.70033 versus 0.70186 previously. The frozen control is therefore
reproduced; the matrix is not using the known bad local AMP diagnostic as its baseline.

M0 changes only the loader/storage and fixed matrix context relative to that older run:
chain-level exact-sequence caches, batch 64/token budget 65,536, eight workers, and the
frozen contact graph. M0-M5 are internally matched on all of these settings.

## Performance audit

| Variant | A800 wall time | Best-epoch residues/s | Data-wait fraction |
|---|---:|---:|---:|
| M0 | 42:59 | 93,179 | 79.2% |
| M1 | 42:07 | 87,687 | 81.5% |
| M2 | 44:27 | 90,342 | 79.6% |
| M3 | 37:59 | 96,245 | 78.2% |
| M4 | 1:17:31 | 48,810 | 83.1% |
| M5 | 1:15:10 | 51,561 | 83.6% |

An isolated A40 profile improved from 71.4k to 91.7k residues/s (+28.3%) when moving
from workers4/prefetch2/cache4 to workers8/prefetch4/cache8; data-wait fell from 78.2%
to 71.1%. GPU activity remains bursty because this is a small classifier reading large
frozen float32 payloads. M4/M5 read both 3,840D ESM2 and 3,456D ESM-C streams and are
about 2x slower. The remaining bottleneck is storage bandwidth/small-file I/O, not VRAM.

## Reproducibility

- Formal Slurm jobs: `9965865` through `9965870`, all `COMPLETED (0:0)`.
- HPC2 runs: `/hpc2hdd/home/shuang886/jhspoolers/PPC/runs/contact_site_cfsi30_esmc_matrix_*_20260711_hpc2`.
- Local checkpoint archive: `benchmark/models/ours/esmc_matrix_20260712/`.
- Machine-readable table: `formal_m0_m5.tsv` and `formal_m0_m5.json` beside this report.
- Loader profile data: `profile_summary.json` beside this report.
- M6 jobs: M0 export `9967169`, M2 export `9967171`, dependent CPU merge `9967203`;
  all completed with exit code 0 and empty stderr.
- M6 machine-readable outputs: `m6/m6_summary.json`, `m6/m6_test_predictions.npz`,
  and `m6_scores/*.npz` beside this report.

Protocol caveat: `eval_test_each_epoch=true` exposed test metrics during development,
although checkpoints and thresholds were selected from validation only. These are
development-split results, not a sealed-holdout claim.
