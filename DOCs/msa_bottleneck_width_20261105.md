# MSA Bottleneck Versus Capacity Screen

## Question

The completed K3 two-layer geometry + MSA model reaches validation macro AP
0.757084 / pooled AP 0.789054. Its fixed equal-logit combination with G2 e27
and the original MSA_D e13 reaches 0.766068 / 0.798613. This motivates testing
whether reducing the MSA Transformer's 768D query representation to only 64D
loses useful information. It does not establish that compression is the cause
of the remaining gap.

The next screen changes only existing residual-branch dimensions. No new model
operator, package, feature extraction, seed control, label, split or test
evaluation is introduced.

## Matrix

| Arm | MSA projection | Adapter hidden | Geometry blocks | Trainable parameters | Role |
|---|---:|---:|---:|---:|---|
| K3, completed reference | 64 | 64 | 2 | 3,693,034 | Original bottleneck |
| K14 | 128 | 128 | 2 | 3,787,626 | Wider representation |
| K16 | 64 | 228 | 2 | 3,787,662 | Capacity control for K14, +36 parameters |
| K15 | 256 | 128 | 2 | 3,902,954 | Wider representation |
| K17 | 64 | 428 | 2 | 3,903,062 | Capacity control for K15, +108 parameters |

The narrow-input controls spend the extra capacity in the conditional adapter,
not in the frozen MSA encoder. All output residuals remain 256D and their last
projection remains zero-initialized. Parameter counts are from the actual
constructed models, not a theoretical estimate.

K14 versus K16 and K15 versus K17 help distinguish a wider MSA representation
from a larger adapter. They do not isolate all optimization effects: constructor
and dropout RNG streams change when dimensions change, despite a common seed.
No multi-seed or independent-test claim is supported by this screen alone.

## Unchanged Contract

- Union-v2 5.5A labels and original pairwise SI30 split: 4,868 train, 1,075
  validation, 724 test sequences. Test is not evaluated in this campaign.
- ESM-C MLC 3456D; corrected OF3 descriptors 91D; frozen unpaired MSA Transformer
  query-row 768D. The legacy `prottrans_*` option names here refer to that MSA
  branch, not ProtT5.
- Exactly K3's feature paths and model except `prottrans_residual_dim` and
  `prottrans_residual_hidden`. Training differs only in its fresh output path
  when constructed under the same experiment root.
- Seed49, 50 epochs, from scratch, no warm start/resume, D loss:
  chain-mean classification plus ratio-MSE weight 0.75.
- Batch64, max tokens65,536, workers8, pinned memory and the existing data
  loader configuration. No silent throughput tuning inside this comparison.
- Primary selection: validation pooled AP, same-checkpoint macro AP guardrail
  >=0.7417948466002535. Keep all epochs and macro-best separately; ties earliest.

GPU submission is to `acd_ue`, one GPU per task, array0-3 without a percent
throttle, 12 CPUs, 192GiB host memory and 12h per task. The larger host-memory
request follows the earlier K5 host OOM and changes resources, not the model.
ACD1-28 is excluded because previous pin-memory retries failed there; this is
an operational precaution, not proof of a defective node. Each task must pass
the existing real-payload CUDA two-update gate before formal training.

## Prospective Readout

1. Read single-model macro/pooled AP at the same selected epoch after all 50
   epochs complete. Do not restart a live task because an observation times out.
2. Compare each wider projection to both K3 and its capacity-matched control.
   Gains shared by wide and narrow controls favor a capacity/optimization
   explanation; gains specific to the wider projection support, but do not
   prove, the information-compression hypothesis.
3. Compute only the four equal-logit ensembles defined in
   [the fixed policy](msa_bottleneck_width_20261105_fixed_blends.json), after
   completion and full validation-provenance checks. No blend-weight search.
4. Remain on validation. A result above 0.80 must retain full coverage and the
   macro guardrail, be reproduced from saved scores, and be labeled as an
   exploratory selected-validation result rather than generalization proof.

The branch already reproduces identical old K0-K13 configurations against
commit `5e4d8e4`. The full local test suite passes 81 tests, with two optional
asset-dependent skips. New tests check actual constructed widths, parameter
matching, zero initialization, ragged masking, finite gradients, and task-driven
projection updates with weight decay zero. Independent review and launch
receipts are recorded in the companion launch JSON once accepted/submitted.
