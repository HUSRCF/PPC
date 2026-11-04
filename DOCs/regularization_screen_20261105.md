# Controlled Regularization Screen

Requested by the user after architecture screening. The fixed reference is K3:
two OF3 geometric-attention blocks, post-TCN MSA residual, projection64D /
adapter64D, 3,693,034 trainable parameters. Its validation-pooled-selected e21
has macro AP0.757084 / pooled AP0.789054. The current strongest completed
fixed system is G2+MSA_D+K15 at0.768019 /0.798762; it remains below0.80.

## Matrix

| Arm | Main dropout | Classifier dropout | Weight decay | Label smoothing | MSA adapter dropout |
|---|---:|---:|---:|---:|---:|
| K3 completed reference | 0.25 | 0.45 | 0.10 | 0 | 0 |
| K19 lower dropout | 0.15 | 0.30 | 0.10 | 0 | 0 |
| K20 higher dropout | 0.35 | 0.55 | 0.10 | 0 | 0 |
| K21 lower weight decay | 0.25 | 0.45 | 0.03 | 0 | 0 |
| K22 higher weight decay | 0.25 | 0.45 | 0.20 | 0 | 0 |
| K23 light label smoothing | 0.25 | 0.45 | 0.10 | 0.02 | 0 |
| K24 MSA dropout | 0.25 | 0.45 | 0.10 | 0 | 0.10 |

K19/K20 change the grouped main+classifier dropout strength, not separately
isolated main-versus-head effects. K24 applies elementwise dropout at the
existing MSA projection output and adapter hidden layer; it does not drop the
entire MSA modality or change the input alignment.

All variants have identical initial model tensors, constructor RNG and parameter
count relative to K3. Training stochasticity intentionally changes when dropout
is active. No new model operator or package is needed.

## Unchanged Contract

- Same union-v2 labels, SI30 split and all input feature paths: 4,868 train /
  1,075 validation, 404,267 validation residues, no test evaluation.
- ESM-C3456D, corrected OF3 descriptors91D, frozen MSA Transformer768D.
- Seed49, from scratch50epochs; no resume or new seed controls.
- Chain-mean CE plus ratio-MSE weight0.75, except the explicitly stated CE
  label-smoothing setting in K23. Learning rate8e-5 and schedule unchanged.
- Batch64, tokens65536, workers8, existing pinning/prefetch/cache configuration.
- `acd_ue`, one GPU each, array0-5 with no percent throttle; 12 CPUs, 192GiB
  host memory, 12h. Exclude ACD1-28 as an operational precaution. Existing
  real-payload CUDA gates run before each formal job.

Select by validation pooled AP subject to same-checkpoint macro AP
>=0.7417948466002535; preserve macro-best separately. Reconstruct selected
prediction AP and provenance before promotion. Do not compare smoothed and
unsmoothed loss values as though they were the same objective; ranking metrics,
Brier/ratio diagnostics and fixed-validation gates remain directly reportable.

The six fixed readouts in [the policy](regularization_screen_20261105_fixed_blends.json)
retain G2e27 and MSA_De13 and substitute each newly completed arm with equal
logit weights. No alpha search, test threshold search or label changes.
This is an adaptive validation screen, not an independent significance claim.

## Milestone and Further Tricks

The [user's milestone contract](pooled_AP080_milestones_20261105.md) takes
precedence: once an eligible system truly reaches pooled AP>=0.80, freeze
validation-selected assets/gates, evaluate test without selecting on test, then
attempt whole-RCSB training under the same single-chain PPIS task.

If needed after this controlled screen, two distinct candidates are R-Drop
(agreement between two dropout forward passes) and EMA/SWA weight averaging.
These are not silently enabled here. R-Drop would require a separately reviewed
masked, FP32 consistency objective; EMA would require checkpoint/averaging-state
audits. Neither justifies upgrading the user's PyTorch environment or guarantees
a PPIS gain. Primary references: [R-Drop, NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)
and [PyTorch averaging documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html).
