# Geometry and MSA screen toward pooled AP0.80

## Contract and selection

This is a new validation-only screen, not a test result. Reuse the original pairwise-SI30 union-v2 split:4868 train,1075 validation,724 sealed test sequences. The fixed input is ESM-C MLC3456D, corrected91D OF3 descriptors and frozen768D MSA Transformer query-row features. OF3 inputs are predicted single-monomer structures, not experimental complexes. No partners, assembly coordinates or test labels enter the model.

All arms start from scratch at seed49 for50 epochs. Batch64, token budget65536, workers8 and existing optimized loaders remain unchanged. Do not resume earlier weights or add seed controls yet. Use acd_ue, one GPU per task and no artificial percent concurrency cap; scheduler/account quotas still apply. No dependencies or Torch components are installed or changed.

The user goal is validation pooled AP>=0.80. Set best.pt to pooled-AP selection and retain best_pooled_ap.pt, best_chain_ap_macro.pt and every epoch separately. This changes checkpoint aliases, not the fixed50-epoch training dynamics. The same-checkpoint macro guardrail is0.741795 (no more than0.005 below prior G2's0.746795). Also report the unconstrained pooled-best point, macro-best point and their exact epochs; never combine metrics across checkpoints. Ratio MAE remains diagnostic. This exploratory validation search is not independent evidence of a final test gain.

## Matrix

|Arm|Change from K1 G2+MSA|Question|
|---|---|---|
|K0_B1_MSA|Original ESM2 contact block instead of geometry attention|Complete B1/G2 x MSA factorial using existing no-MSA controls|
|K1_G2_MSA|One OF3 edge-attention block, MSA after TCN|Core geometry/evolution complementarity|
|K2_G2_MSA_preTCN|Move MSA residual before TCN|Can task-specific local processing exploit MSA earlier?|
|K3_G2_MSA_twohop|Two geometry attention blocks|Does multi-hop surface context help?|
|K4_G2_MSA_dualgraph|ESM2 contact block before OF3 geometry attention|Are PLM contacts and predicted geometry complementary?|
|K5_G2_MSA_postgeo|Move MSA residual after geometry attention|Is late evidence correction better than letting MSA change graph messages?|
|K6_G2_MSA_fullcontext|One full sequence-attention block instead of global mean/max broadcast|Does geometry need richer global context?|
|K7_G2_MSA_residue_mean|Change CE aggregation from chain_mean to residue_mean only|Does residue weighting improve pooled ranking?|
|K8_G2_MSA_pure_CE|Set ratio loss weight0, keep chain_mean CE|Does ratio supervision constrain ranking?|

K0-K6 all keep D loss (unweighted chain-mean CE +0.75 ratio MSE); K7 and K8 are loss controls, not architecture-only rows. All use the same features except the explicitly required ESM2 contact graph in K0/K4. New modules/feature branches consume different constructor/dropout RNG; this is a single-seed system screen, not perfectly parameter-matched causality.

## Acceptance and provenance

- Hash all three frozen parent configs, every generated config and the deployed source archive.
- Require full6667-sequence geometry/MSA receipts, exact/full-length queries, train-only structure normalization, zero labels read during feature generation and unchanged feature-builder hashes.
- Run two real training updates plus two isolated validation chains before each formal run; require finite loss, no AMP-skipped smoke updates, backbone/MSA/each zero-init geometry block updated, full prediction export and zero test samples.
- Retain code/config/logs/runs in a fresh HPC3 experiment directory. Do not edit completed/running experiment snapshots.
- Check run health and AP at early/middle/final epochs. If no candidate clears both pooled target and macro guardrail, do not declare success or optimize on test.

## Next decisions

If G2+MSA improves both AP axes, retain it as the next common backbone. If deeper/hybrid/global-context models mainly raise train fitting, do not infer a useful validation gain. Evaluate a small set of prespecified equal-logit blends only after the single-model results are frozen.

A future bidirectional Gated DeltaNet remains a separate candidate, requiring a cloned environment, audited dependency dry run and approval before installation. It is not included in these nine jobs.
