# Completed one-layer evolutionary-edge control

Both jobs completed50 epochs with exit0. Same union-v2 validation1075 exact sequences /404267 residues, corrected91D OF3, ESM-C MLC3456D, frozen768D MSA, D loss and seed49. No test was evaluated and no validation batches were skipped.

|System|Pooled-best epoch|Macro AP at same epoch|Pooled AP|AUROC|Best F1@gate|Best MCC@gate|
|---|---:|---:|---:|---:|---|---|
|K1 G2+MSA|22|0.749978|0.783074|0.889157|0.704725@0.48|0.592231@0.59|
|K9 +MSA edge bias|19|0.756167|0.787362|0.893259|0.709239@0.29|0.600848@0.47|

At the common pooled-AP selection policy, K9 improves macro AP by0.006189 and pooled AP by0.004288. K1 also has a separate macro-best epoch20 (macro0.754561 / pooled0.782451); against that macro value, K9's macro gain is only0.001606. Do not attach K1's macro-best score to epoch22.

The8328 extra parameters only modulate existing sparse geometry edges using projected MSA information. Initial backbone parameters/RNG and zero-bias function were checked equal, and the actual CUDA three-update gate demonstrated scale/query/key learning without weight decay. This supports retaining the mechanism as a candidate, but one seed and validation checkpoint selection do not establish a statistically stable causal improvement.

K9's ratio MAE at the pooled-selected point is0.051386 versus K1's0.059563, diagnostic only. Runtime37m16s versus51m54s is not a speed claim because cold-cache and host conditions differ. Raw metric SHA-256 and both AP-selected checkpoints are in the [machine-readable result](evolutionary_geometry_edges_20261105_K1_K9_results.json).

The0.80 target is not yet met by this single model. The earlier fixed G2+D MSA ensemble remains0.759398 macro /0.790671 pooled. Current deeper/new-loss trials and the explicitly planned fixed follow-up blends remain separate, not completed evidence.
