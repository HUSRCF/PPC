# Completed post-MSA context screen

Array799764_[0-2] completed all50 epochs with exit0. C_bigru and B_pre_tcn took38m58s each, D_capacity_mlp33m02s, including acceptance gates and read-only train/validation diagnostics. No test split was evaluated.

## Matched macro-selected results

All four arms use seed49, D loss, the original frozen90D descriptors, ESM-C MLC,768D MSA Transformer and the ESM2 contact graph. The union split contains4868 train and1075 validation sequences. Each row uses its own validation macro-AP-best checkpoint; pooled AP in that row comes from the same checkpoint.

|Arm|Job|Best epoch|Val macro AP|Val pooled AP|Train macro AP|Train pooled AP|
|---|---|---:|---:|---:|---:|---:|
|A: D_encoder reference|Previously completed|13|0.734337|0.760760|0.807470|0.863030|
|B: MSA before TCN|799764_1|14|0.736004|0.757229|0.816062|0.869467|
|C: Post-MSA residual BiGRU|799764_0|14|0.734465|0.760431|0.820858|0.870180|
|D: Capacity-matched residual MLP|799764_2|13|0.733437|0.763060|0.812483|0.866452|

Relative to A, B changes macro/pooled AP by+0.001668/-0.003531; C by+0.000129/-0.000329; D by-0.000900/+0.002300. The GRU's higher train macro AP does not transfer to a clear joint improvement in validation. This supports prioritizing generalization/feature-interaction controls over simply increasing recurrent capacity; it does not prove that every larger or recurrent model will fail.

Each new run recorded two AMP-skipped updates in50 epochs, zero skipped validation batches and no test metrics. This is distinct from the severe C1 numerical failure in the geometry screen. Read-only evaluation reports gradients disabled, zero parameter gradients and zero parameter version changes for both train and validation.

## Pooled-selected alternatives

These are separate checkpoints, not pooled scores to attach to the preceding macro-selected rows.

|Arm|Pooled-best epoch|Macro AP at that epoch|Pooled AP|
|---|---:|---:|---:|
|B: MSA before TCN|18|0.734161|0.766166|
|C: BiGRU|27|0.732065|0.764363|
|D: Capacity MLP|13|0.733437|0.763060|

B epoch18 is the most useful additional operating point: relative to A epoch13, macro AP changes by-0.000175 while pooled AP improves+0.005406. This is exploratory checkpoint selection on validation, not independent evidence of superiority.

## Train/validation diagnostic

The best-macro checkpoint was reevaluated with model.eval(), no optimizer and isolated gradients. Valid residue counts were1754693 train and404267 validation for every arm. Tiny pooled AP differences of a few1e-9 between training-time validation and the standalone diagnostic reflect numerical/batching effects; the main table retains the original epoch metrics.

|Arm|Train D loss|Val D loss|Train-minus-val macro AP|Train-minus-val pooled AP|
|---|---:|---:|---:|---:|
|A|0.330866|0.460884|0.073133|0.102270|
|B|0.341410|0.471928|0.080058|0.112238|
|C|0.316108|0.463838|0.086392|0.109749|
|D|0.305888|0.456414|0.079046|0.103391|

The reduced training loss/stronger training ranking in C/D is not by itself validation progress. No new seed confirmation or test evaluation was launched in this result collection.

## Next controlled comparisons

1. Complete B1+MSA and G2+MSA to isolate the geometry-by-MSA interaction. G2 without direct MSA is currently the stronger single-seed validation candidate (macro0.746795, pooled0.780250), but differs in descriptors and graph architecture from A-D.
2. Keep B epoch18 and the fixed existing MSA blends as validation candidates without further large weight searches.
3. Consider one bidirectional Gated DeltaNet as a later sequence-context control. Do not infer its performance from either the BiGRU result or the numerically failed C1 linear-attention run.

[Protocol](msa_context_screen_20261104.md), [full metrics, diagnostics and SHA-256](msa_context_screen_20261104_results.json), [geometry results](geometry_screen_20261104_results.md), [DeltaNet feasibility](deltanet_feasibility_20261104.md).

Raw runs reside under /data/user/shuang886/PPC_M18/sequence_union_v2_20261030/msa_context_screen_20261104/runs/. Each contains metrics.jsonl, epoch checkpoints, best.pt and train_val_best.json. The machine-readable result records both raw-file hashes per new arm.
