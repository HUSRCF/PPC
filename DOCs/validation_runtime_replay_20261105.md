# Validation-Only GPU Runtime Replay

Purpose: verify the dedicated strict inference producer on the complete
frozen validation cohort before the eventual test stage. This is not a new
model, checkpoint selection, weight search or permission to evaluate test.
The validation leader remains below0.80 at0.7997048805648456.

The input points are the existing K18 fixed-blend artifacts on HPC3:
`/data/user/shuang886/PPC_M18/sequence_union_v2_20261030/geometry_position_depth_20261105/fixed_K18_blend`

The driver `scripts/analysis/replay_union_validation_sources.py` replays:
G2epoch27,MSA_Depoch13,K18epoch31, then their already-declared equal-logit
thirds. It does not evaluate any newly proposed ensemble or change weights.

## Contract

- Require a full validation-only summary:1075queries/404267residues.
- Check original prediction/metrics hashes, checkpoint epochs and frozen
  validation IDs. Only `val_ids.txt` is opened;no shared train/test cohort table
  or test label/prediction files are needed.
- Reconstruct validation metadata from original saved validation predictions;
  validate labels, sequence hashes, offsets and original NPZ scores exactly.
- Check checkpoint configuration for4868train/1075val/0test and no per-epoch
  test evaluation;strictly load the checkpoint with no optimizer.
- Use the original evaluation batch/length-packing settings. Dataset path
  order and the original val ID file were verified sorted;no batching change
  is introduced in this replay.
- Reuse actual-input sequence/file audits, eval/no-grad and complete model
  state invariance from the reviewed producer. No skipped/truncated residues.
- Retain the predeclared score tolerances:maximum absolute error2e-5,
  RMSE2e-6,and absolute pooled/macro AP difference2e-6. Do not relax these to
  turn a failed replay into a pass. The earlier CPU diagnostic is not accepted
  as GPU reproduction.
- Recompute only the existing blend and compare it to its original cached
  NPZ. Record timing,peak VRAM,Torch/GPU and precision settings.
- Write original per-source replay predictions/integrity receipts and a final
  validation-only summary;fail closed and retain diagnostics on mismatch.

The final receipt always sets `test_inference_permitted=false`, even when
reproduction passes. The independent0.80 eligibility gate and validation
model lock remain mandatory before any test operation.

Six focused tests check full validation metadata/IDs, the exclusively
validation execution path, failure receipts without tolerance relaxation,
inference exceptions, manifest mutation and preservation of the first hash
for paths shared by multiple sources. The first manifest bytes are hashed
before parsing;all referenced files are checked unchanged after replay.
They complement the eight producer and six lock/scorer tests. The current
complete local suite has106passes/2optional skips. Narrow read-only review
used gpt-6-luna max and verified the audit-integrity fixes. Submission is a separate
single-GPU acd_ue job, with absolute Python and unbuffered output;no packages
or running training tasks are modified.
