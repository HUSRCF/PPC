# Fusion Position x Geometry Depth

Three completed cells motivate one missing controlled cell. Each value below
is from the single validation-pooled-AP-selected checkpoint, not independently
maximized metric columns.

| Geometry blocks | MSA post-TCN | MSA pre-TCN |
|---|---|---|
| 1 | K1 e22: macro0.749978 / pooled0.783074 | K2 e19: macro0.754082 / pooled0.787838 |
| 2 | K3 e21: macro0.757084 / pooled0.789054 | **K18: new controlled trial** |

K2 - K1 pooled AP is +0.004765; K3 - K1 is +0.005980. These selected-validation,
single-seed observations do not prove the effects will add. K18 tests their
interaction without changing parameter count, feature width or loss.

Relative to K3, the only model change is `msa_residual_position: pre_tcn`.
The model still has 3,693,034 trainable parameters and a 768 ->64D MSA
projection with a 64D zero-initialized residual adapter. At initialization,
the two placements have identical state tensors, RNG state and outputs in the
ragged-payload regression test. They intentionally differ after the residual
learns, allowing the TCN to process the MSA update before spatial propagation.

## Unchanged Training Contract

- Union-v2 5.5A labels, original pairwise SI30 split, 4,868 train / 1,075 val;
  test is not evaluated.
- ESM-C MLC3456D, corrected OF3 descriptors91D, frozen MSA Transformer768D.
- Two OF3 geometric-attention blocks, no new edges or structural source.
- D loss, chain-mean CE + ratio-MSE weight0.75; seed49, from scratch50epochs.
- Batch64, tokens65536, workers8; no package installation or throughput-setting
  change within this controlled comparison.
- `acd_ue`, one GPU, no array-percent throttle, 12 CPUs, 192GiB RAM, 12h;
  exclude ACD1-28 as an operational precaution, not a proven fault diagnosis.
- Existing real-payload CUDA gate must pass before training. Preserve failed
  attempts; do not restart a live task merely because observation timed out.

Primary selection is validation pooled AP with same-checkpoint macro AP
>=0.7417948466002535; ties earliest. All epochs and macro-best are retained.
After successful completion of50epochs/full1075queries/404267residues with no
validation skips or test evaluation, evaluate only the fixed equal-logit blend
in [the policy](geometry_position_depth_20261105_fixed_blend.json). No weight or
additional-combination search. This remains an adaptive exploratory validation
campaign, not independent confirmation or a new sealed holdout.

This trial is separate from K14-K17's projection-width/capacity screen. Their
frozen execution snapshot and active tasks are not modified.

## Launch

Submitted as **800231_0**, `acd_ue`, from frozen commit
`5af793878689343573a21bdb456e9aeeebd2434d`. Archive SHA-256:
`efc9cf642b4aed7167cb777721324a2cbda9eee607a36f3647aab28049b7d869`.
Actual K3/K18 config comparison confirmed identical data, only the intended
fusion-position model difference, and only fresh output/provenance training
paths. The full models have identical initial state and constructor RNG.

The task passed the real CUDA two-update gate and entered formal training with
4,868 train / 1,075 val / zero test samples and 3,693,034 parameters. The
2026-11-04 18:26 UTC health snapshot records its first full-coverage validation
epoch, no skipped validation batches and no AMP-skipped updates.

See [launch and preflight receipts](geometry_position_depth_20261105_launch.json)
and [actual configs/parent hashes](geometry_position_depth_20261105/).
