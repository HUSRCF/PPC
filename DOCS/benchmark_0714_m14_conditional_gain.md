# M14 ESMC-ESM2 Conditional Gain Report (2026-07-14)

## 1. Question

This experiment tests whether ESM2 contains predictive information beyond the
current ESMC MLC model for single-chain effect-site residue prediction. ESMC is
kept as the primary representation; ESM2 is treated only as a residual expert.

The experiment does not test a symmetric dual-encoder architecture. It tests:

1. validation-only grouped OOF logit stacking;
2. a frozen-backbone, zero-initialized ESM2 logit residual (M14b);
3. a frozen-backbone, zero-initialized ESM2 hidden residual immediately before
   the final contact graph block (M14c).

## 2. Evaluation Contract

- Canonical split:
  `features/contact_labels_5p5A/splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708/chain_filtered_training`
- Chain manifest:
  `features/contact_labels_5p5A/splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708/pypropel_chain_filtered_no_len_limit/chain_manifest.csv`
- Validation: 3,796 chains, 55 SI components, 1,427,257 residues.
- Test: 2,392 chains, 43 SI components, 1,036,256 residues.
- Validation and test component sets are disjoint.
- Logit exports require exact chain and residue coverage.
- PDB IDs are normalized to lowercase, but chain suffixes remain case-sensitive.
  For example, `6gyb__C` and `6gyb__c` are distinct chains. The training loader
  already preserved this distinction; the new exporter was corrected and
  audited before the locked test evaluation.
- The stacker coefficients and all operating thresholds are fitted on
  validation only. The test set is evaluated once after locking them.
- Locked evaluator record: `test_used_for_fit_or_threshold = false`.

## 3. Frozen Base Models

Primary ESMC model:

- M10f, seed 44, checkpoint epoch 13.
- Checkpoint:
  `runs/contact_site_cfsi30_m10star_m10f_lr8e5_ratio075_seed44_e30_20260713_star/best_chain_ap_macro.pt`
- Input width: 3,456 (three 1,152-dimensional ESMC layers).

Secondary ESM2 model:

- M0 ESM2 MLC, seed 42, checkpoint epoch 30.
- Checkpoint:
  `runs/contact_site_cfsi30_esmc_matrix_m0_esm2_mlc_seed42_e30_20260713_star_insert/last.pt`
- Input width: 3,840 (three 1,280-dimensional ESM2 layers).

## 4. Grouped OOF Validation Result

Five-fold `StratifiedGroupKFold` uses SI component IDs as indivisible groups.
Fold-specific stackers and thresholds are fitted without their held-out
components.

| Method | Chain-macro AP | AUPRC | AUROC | Cross-fit F1 | Cross-fit MCC |
|---|---:|---:|---:|---:|---:|
| ESMC M10f | 0.752163 | 0.767180 | 0.869703 | 0.689189 | 0.552545 |
| ESM2 M0 | 0.703035 | 0.732066 | 0.850870 | - | - |
| Grouped OOF logit stack | 0.762989 | 0.782460 | 0.879260 | 0.697306 | 0.575099 |

OOF deltas versus ESMC M10f:

- Chain-macro AP: +0.010826.
- AUPRC: +0.015280.
- AUROC: +0.009557.

The ESM2 coefficient is positive in every fold. The five
`(ESMC, ESM2)` coefficient pairs are:

`(0.4999, 0.2540)`, `(0.4754, 0.2962)`, `(0.4602, 0.2652)`,
`(0.4604, 0.2887)`, and `(0.4720, 0.2733)`.

A chain-equal weighting sensitivity run gives effectively the same ranking
result (stack chain-macro AP 0.762988, AUPRC 0.782388, AUROC 0.879764).

## 5. Locked Test Result

The full-validation stacker is logistic regression with:

- intercept: -0.600319;
- ESMC coefficient: 0.472508;
- ESM2 coefficient: 0.276012.

| Method | Chain-macro AP | AUPRC | AUROC | F1 at val-F1 gate | MCC at val-F1 gate | Gate | F1@0.6 | MCC@0.6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESMC M10f | 0.689766 | 0.730239 | 0.850668 | 0.650898 | 0.528727 | 0.49 | 0.636901 | 0.525570 |
| ESM2 M0 | 0.661635 | 0.696176 | 0.834296 | 0.640621 | 0.501269 | 0.46 | 0.626402 | 0.500111 |
| Locked logit stack | 0.709986 | 0.753806 | 0.871844 | 0.674026 | 0.557381 | 0.33 | 0.580236 | 0.510872 |

Locked-stack ranking deltas versus ESMC M10f:

- Chain-macro AP: +0.020220.
- AUPRC: +0.023567.
- AUROC: +0.021176.
- F1 at validation-selected F1 gate: +0.023128.
- MCC at validation-selected F1 gate: +0.028654.

The fixed 0.6 gate is not appropriate for the calibrated stack output. Its
official comparison should use the validation-selected gate (0.33), while the
0.6 values are retained only for completeness.

## 6. Trainable Residual Experiments

Both residual variants load M10f strictly, freeze every non-ESM2-residual
parameter, force the frozen backbone into evaluation mode, stop gradients at
the ESMC context, and train only 274,946 residual parameters. The new residual
output projection is initialized to zero, so both models are bitwise identical
to M10f before the first update.

M14b:

- ESM2 three-layer MLC -> 64D residual -> residue logit correction.
- 20 epochs, seed 44, batch size 64, max batch tokens 65,536.
- Learning rate 3e-4, weight decay 1e-2, label smoothing 0.05.

M14c:

- ESM2 three-layer MLC -> 64D residual -> 256D hidden correction.
- Correction is inserted after TCN2 and before the final frozen contact graph.
- 20 epochs, seed 44, batch size 64, max batch tokens 65,536.
- Learning rate 3e-4, weight decay 1e-2, label smoothing 0.05.

Both jobs completed all 20 epochs and exited normally. Checkpoints are selected
by validation chain-macro AP.

| Model | Best epoch | Chain-macro AP | AUPRC | AUROC | Best F1 | MCC at best-F1 gate |
|---|---:|---:|---:|---:|---:|---:|
| Frozen parent M10f | 13 | 0.752163 | 0.767181 | 0.869703 | 0.690168 | 0.563921 |
| M14b logit residual | 1 | 0.742447 | 0.760290 | 0.871372 | 0.691343 | 0.564962 |
| M14c pre-contact residual | 1 | 0.738597 | 0.757287 | 0.867748 | 0.685229 | 0.553661 |

M14b versus parent:

- Chain-macro AP: -0.009716.
- AUPRC: -0.006891.
- AUROC: +0.001669.
- Best F1: +0.001175.

M14c versus parent:

- Chain-macro AP: -0.013566.
- AUPRC: -0.009894.
- AUROC: -0.001955.
- Best F1: -0.004939.

Both residual models peak at epoch 1 and then degrade. At epoch 20, M14b
chain-macro AP is 0.699260 and M14c is 0.691150. This is consistent with rapid
overfitting or destructive residual optimization, not with a useful hidden
fusion path under the tested hyperparameters.

## 7. Interpretation

The grouped OOF and locked test results provide direct evidence that ESM2 has
conditional predictive value after ESMC at the output-error level. Therefore,
the earlier failure of fixed-weight or ordinary hidden gating should not be
interpreted as evidence that ESM2 is information-free.

The current evidence supports a validation-fitted asymmetric logit ensemble.
It does not yet support replacing the ESMC main path, symmetric embedding
fusion, or unfreezing the downstream trunk. M14b and M14c did not exceed the
frozen parent on the predeclared validation selection metric, so the planned
partial-unfreeze stage is not started.

## 8. Limitations

- This is one ESMC checkpoint and one ESM2 checkpoint, not a multi-seed
  conditional-gain study.
- The base checkpoints were selected using the full validation set. Grouped OOF
  cross-fits the stacker, not the base models; this is not nested outer CV.
- No claim of statistical significance is made until SI-component block
  bootstrap and multi-seed replication are complete.
- The ESM2 model uses seed 42 while the ESMC model uses seed 44. This is a valid
  ensemble pair but not a matched-seed architectural ablation.

## 9. Provenance

Star repository commits:

- `7c756e6` Add conditional ESM2 residual experiments
- `29fc466` Preserve chain ID case in validation export
- `d53926e` Speed up grouped OOF threshold search
- `eeef02a` Lock validation-fitted logit stack evaluation
- `272cdcc` Preserve case-sensitive PDB chain identities

Artifacts:

- Validation logits SHA-256:
  `f1bc4f65f544fd484ca8515ed034ea23d5a5f5029dc7744a2b73739626063233`
- Test logits SHA-256:
  `bedcc019be8025fe4440f936db748c8de18e2a6c29804c415e836e0be95e7ac4`
- Locked metrics:
  `/data/husrcf/ProtBind/PPC/benchmark/runs/esmc_m14_star_20260714/oof_stack/locked_valfit_test_metrics.json`
- Complete Git bundle SHA-256:
  `56679b454b516c101a368dd1d64d6af529564f5184c39bd6c329cb6fdedc6457`
- Git bundle:
  `/data/husrcf/ProtBind/PPC/benchmark/runs/esmc_m14_star_20260714/provenance/PPC_m14_272cdcc.bundle`
