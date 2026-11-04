# Completed Validation Arm Readouts

`scripts/analysis/read_completed_union_arm.py` consolidates the current
campaign's manual completion/selection checks. It does not train,infer on
test,select test gates or grant test permission. No new models or mixtures
are introduced by this tool.

## Required Evidence

- Bind the prospective fixed-blend policy to the copy in the original
  pre-training source snapshot,not a rewritten policy after seeing results.
- Identify exactly one jobs.json arm and its actual launched array task.
  Require sacct COMPLETED exit0 before parsing its metrics for selection.
- Check the submitted configuration hash and every explicitly declared
  data/training/model field against the real runtime configuration.
- Require seed49,scratch50epochs,4868train/1075val/0test,no cropping and no
  per-epoch test evaluation.
- Require exactly consecutive epochs1--50,full1075queries/404267residues,
  no skipped validation batches and explicit no-gradient/state-isolation
  fields for every validation epoch. Reject test events and nonfinite AP.
- Select maximum validation pooled AP subject to the unchanged same-point
  macro AP guardrail0.7417948466002535;earliest ties. Retain the distinct
  macro-best epoch in the completion receipt rather than mixing metrics.
- Resolve only the already-declared G2e27+MSA_De13+selected-arm equal-logit
  thirds,epsilon1e-7. Source validation arrays must reproduce logged AP via
  the existing `analyze_union_fixed_blends.py` helper. That helper selects
  validation rows from the shared sequence-group metadata;it does not open
  test labels,predictions or inference inputs.
- Bind input/config/policy/checkpoint/prediction hashes,the shared cohort
  metadata table and the resolved specification. Recheck them after analysis and recheck terminal status
  before writing completion_audit.json. Existing outputs/specifications are
  not overwritten;partial failed outputs remain unaudited.

This is not the strict checkpoint/model-state/actual-feature audit used for
the0.80 lock. A readout near or above0.80 still requires that separate
provenance/strict-state lock and GPU replay before any fixed test evaluation.
Every completion audit keeps`test_inference_permitted=false`.

## Invocation

Use absolute Python and a separately frozen analysis source directory. Do
not modify old training snapshots. For example,after K26 finishes:

```bash
/data/user/shuang886/.conda/envs/openfold/bin/python -u \
  /path/to/frozen/readout_code/scripts/analysis/read_completed_union_arm.py \
  --base /data/user/shuang886/PPC_M18/sequence_union_v2_20261030 \
  --experiment /data/user/shuang886/PPC_M18/sequence_union_v2_20261030/preTCN_interactions_20261105 \
  --policy /data/user/shuang886/PPC_M18/sequence_union_v2_20261030/preTCN_interactions_20261105/code/DOCs/preTCN_interactions_20261105_fixed_blends.json \
  --arm K26_G2_MSA_twohop_preTCN_pure_CE \
  --output /data/user/shuang886/PPC_M18/sequence_union_v2_20261030/preTCN_interactions_20261105/fixed_K26_blend
```

Only a readout with completion_audit.json and matching unchanged inputs is
accepted as this tool's completed output. A live task,observation timeout or
failed task is not a reason to restart training. Wait/poll the same task;
never substitute a partial epoch history for successful completion.

Local tests cover guardrail/ties,missing/duplicate epochs,test exposure,
coverage/skips/gradient isolation,invalid AP,exact scheduler identity,
live-task refusal,policy/runtime drift,input mutation and resolved-spec
mutation during analysis. Review and actual remote acceptance are recorded
separately once performed.

## Local Acceptance and Review

The complete local suite passed120tests with2optional skips. The10focused
readout tests passed. Narrow gpt-6-luna max review identified a missing hash
binding for the helper's shared cohort table;the table is now watched before
invocation and rechecked before audit publication. A dedicated mutation
regression fails closed. The reviewer rechecked this fix and returned PASS.
No model,training implementation,configuration or package was changed.
