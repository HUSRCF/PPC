# Post-MSA sequence-context screen

## Why this screen

The completed D_encoder seed49 checkpoint at epoch13 was reevaluated with model.eval(), no optimizer and gradient isolation. This was not the training-mode running metric.

| Split | Unique sequences | Residues | Macro AP | Pooled AP | AUROC | D loss |
|---|---:|---:|---:|---:|---:|---:|
| Train |4868|1754693|0.807470|0.863030|0.942460|0.330866|
| Validation |1075|404267|0.734337|0.760760|0.875200|0.460884|

Macro and pooled AP gaps are0.073133 and0.102270. Ranking is better on train, but train is not saturated; neither parameter count nor this single selected checkpoint diagnoses the cause. The aim is to test fusion placement and sequence inductive bias, not assert insufficient capacity. Ratio MAE is0.086107 train versus0.074625 validation and remains diagnostic.

Evaluation job799702 completed normally in2m31s. Both splits report grad mode false, zero parameter gradients and zero parameter version changes. No test was read. The compact evidence and raw output hash are in msa_context_screen_20261104_launch.json; raw output is retained on HPC3 under diagnostic/train_val_epoch13.json and locally under tmp/msa_context_diagnostic/.

## Controlled matrix

All arms use the original frozen90D descriptors, ESM-C MLC, frozen768D MSA Transformer features, ESM2 contact graph, union-v2 labels, D loss and seed49. This deliberately does not mix B0/B1 descriptor changes into the context comparison.

| Arm | MSA adapter location | New post-fusion processing | Total parameters | Execution |
|---|---|---|---:|---|
| A control |After TCN|None|2821250|Reuse completed D_encoder50ep; best macro epoch13|
| B_pre_tcn |Before TCN, after local MLPs|None|2821250|Fresh50ep|
| C_bigru |After TCN|Pre-LN, one bidirectional GRU256->128 per direction->256|3118210|Fresh50ep|
| D_capacity_mlp |After TCN|Pre-LN, MLP256->576->256|3117506|Fresh50ep|

C and D have explicit output dropout0.1 and fixed residual scale0.1. Added parameter counts including LayerNorm are296960 and296256, a704-parameter difference (~0.24%). No extra output projection or learned residual scale is introduced. The GRU's internal dropout is0 because there is only one recurrent layer.

Packed sequence lengths come from protein_mask, not the loss/label mask. The module rejects empty chains, non-prefix masks and multiple chains in a recurrent sample. Unpacking restores the original sample order and full padded length. Existing ignored-label residues remain sequence context. This does not alter the underlying observed-sequence contract or invent missing polymer residues.

New context modules initialize in a CPU RNG-preserving scope after the existing backbone. Real parent-config checks found identical data and training settings (except output directories), identical initial backbone weights and identical post-construction CPU RNG. Added module dropout still changes later random-number consumption, so trajectories are not bitwise-matched controls.

## Acceptance and selection

- Array799764_[0-2], order C/B/D, acd_ue. No percent concurrency throttle. One GPU,12 CPUs and96GiB per arm,12h each.
- Each arm first runs real-payload CUDA/AMP smoke: two updates, two validation chains, finite losses, updated MSA and context branches, prediction export, zero test samples and isolated validation.
- Formal runs start from scratch,50 epochs, unchanged batch64/token65536/worker8 contract. No resume or initialization checkpoint.
- Primary selection remains validation macro AP; pooled AP is co-primary, MAE diagnostic. All epochs and metric-specific checkpoints remain available.
- After training, best.pt is reevaluated read-only on train,val. A separate diagnostic config relaxes validation-only solely for this explicit standalone evaluator call; training configs remain validation-only and no test evaluation occurs.
- Independent narrow review passed model, config and launcher checks, including CPU BF16. Local suite60 tests:58 passed,2 optional skips. Legacy no-MSA and trained nonzero-MSA paths preserve parameters/RNG/logits exactly. All three real CUDA/AMP smokes subsequently passed; all three formal runs reached2 epochs concurrently with zero AMP skipped updates at verification.

## Concurrency amendment

The earlier two-task array throttle was an operational choice, not part of the training contract. User requested maximum parallelism. Existing geometry array was raised to its full8-task extent; the three ordinary MSA arrays now have no throttle and no artificial cross-training-stage dependencies. Full encoder/profile feature receipts were verified before clearing those dependencies. Scheduler/account quotas still apply; running tasks were not restarted or killed. New context array has no throttle.

Source execution commit03014ab605e3a52f73deb9228086bdad7f405e06. Code, configs, logs and runs are isolated at /data/user/shuang886/PPC_M18/sequence_union_v2_20261030/msa_context_screen_20261104. No packages or Torch components were installed or changed.

