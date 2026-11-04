# Geometry depth x ratio objective

This follow-up was proposed from incomplete validation curves, not independent confirmation. At the decision snapshot, K3 two-hop reached macro0.761524 / pooled0.784884 at epoch13, and K8 one-hop pure CE reached0.751340 /0.785054 at epoch17. Training was still running, so these are not final outcomes.

The hypothesis is that broader geometry context and removal of ratio regularization may help different parts of ranking. Complete a small factorial instead of adding an unstructured mixture.

|Geometry blocks|D: chain-mean CE +0.75 ratio MSE|Pure chain-mean CE|
|---:|---|---|
|1|K1,799902_1,already running|K8,799902_8,already running|
|2|K3,799902_3,already running|K11,new|
|3|K12,new|K13,new|

Only K11-K13 are new submissions. Reuse the same ESM-C MLC3456D, corrected91D OF3 descriptors and graph, frozen768D MSA, seed49, full sequences, original union-v2 split, optimized batch64/token65536/workers8 and50-epoch scratch training. Each within-depth pair changes only ratio loss weight, apart from output paths. This does not use the new MSA edge-bias operator; K9/K10 remain a separate mechanism test.

Validation pooled AP is the checkpoint priority; require same-checkpoint macro AP>=0.741795 for promotion. Preserve all epochs and both AP-best checkpoints. No test evaluation, checkpoint warm-start, seed extension or weight sweep. Initializing additional graph blocks changes later constructor/dropout RNG, so depth comparisons are single-seed system comparisons, not perfect identical-initialization controls.

Operationally use acd_ue and no artificial concurrency cap. All three new jobs require the existing real-payload CUDA gate before formal training. Use a fresh code/config/output snapshot and explicit --variants selection; do not regenerate previous jobs.
