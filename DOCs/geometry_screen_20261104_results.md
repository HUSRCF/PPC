# Completed geometry/global-context screen

All eight jobs exited COMPLETED/0 and recorded50 epochs. Runtime ranges28m22s--33m34s. This does not imply all eight are numerically valid architecture comparisons: C1 has a serious AMP precision failure.

Same union-v2/OF3 validation:1,075 exact sequences, seed49 and D loss. No test was evaluated. Table values use the maximum validation macro AP checkpoint; F1 and MCC are pooled metrics with separate validation-selected gates.

|Job|Arm|Epoch|Macro AP|Pooled AP|AUROC|Best F1@gate|Best MCC@gate|AMP skips|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|799523_0|B0_fixed90|27|0.7261|0.7563|0.8696|0.6826@0.45|0.5678@0.60|2|
|799523_1|B1_fixed91q|18|0.7271|0.7554|0.8717|0.6816@0.43|0.5591@0.49|2|
|799523_2|G0_of3_mean|20|0.7398|0.7573|0.8740|0.6794@0.24|0.5620@0.38|2|
|799523_3|G1_edge_mlp|20|0.7391|0.7677|0.8787|0.6898@0.25|0.5747@0.35|2|
|799523_4|G2_edge_attention|27|0.7468|0.7803|0.8865|0.7034@0.41|0.5954@0.62|2|
|799523_5|G3_gvp|44|0.7414|0.7614|0.8745|0.6818@0.27|0.5658@0.56|1|
|799523_6|C0_full_attention|19|0.7264|0.7578|0.8714|0.6805@0.51|0.5618@0.75|1|
|799523_7|C1_linear_full|50|0.6522|0.6788|0.8297|0.6267@0.44|0.4788@0.47|166|

## Identity of the four just-finished jobs

- `799523_4` / G2: query-conditioned sparse attention with OF3 distance, local-direction and confidence edge features. This is the strongest valid arm; epoch27 maximizes both macro and pooled AP.
- `799523_5` / G3: GVP-style scalar/vector geometry messages with invariant readout.
- `799523_6` / C0: one full sequence-attention block replacing global mean/max broadcast, retaining the ESM2 contact graph.
- `799523_7` / C1: positive-kernel linear attention followed by one full-attention block. Its old result is numerically compromised, not evidence that the architecture is intrinsically poor.

## Main observations

G2 versus its B1 input-matched control improves macro AP by0.0197168 and pooled AP by0.0248334. Versus G1 edge MLP it improves0.0076864 and0.0125860, respectively. G2's point estimates also exceed the prior fixed baseline+MSA blend (0.7413547/0.7721858), but this is a system comparison with different features/architectures, not a parameter-matched causal claim. No multi-seed or independent-test confirmation is claimed.

G0 and G1 have pooled-selected alternate points: G0 epoch21 gives0.7392678 macro /0.7698698 pooled; G1 epoch27 gives0.7383557 /0.7706073. These must not be mixed with their macro-best epochs. G2 is stronger than both alternate points on both AP axes.

## C1 precision defect

C0 had one AMP skipped update and one sanitized gradient element. C1 had166 skipped updates and11,388,738,978 cumulative sanitized elements across updates (not distinct parameters). Validation batches remained finite because downstream sanitization can conceal an upstream training instability.

The intended FP32 positive-kernel sums were still inside outer autocast. Calling.float() on q/k/v did not protect einsum, whose contractions lowered to FP16 again. A same-environment PyTorch2.7.1 toy with length4096 and head width32 turned the correct denominator131072 into FP16 infinity. A4096-long module regression with nonzero output weights failed before the patch and passes after it.

The fix wraps the ELU feature map, contractions, normalization and division in an autocast-disabled FP32 subregion; projections remain under AMP. This does not change model parameters, source features or the non-AMP evaluation function. [PyTorch AMP2.7](https://docs.pytorch.org/docs/2.7/amp.html) documents explicit disabled subregions for controlling execution dtype.

Old C1 artifacts remain immutable and flagged numerical failure. Only C1 is retrained from scratch with the identical frozen config except a new output path and the precision correction. A dedicated long CUDA test and the original real-payload two-update gate must pass before its50-epoch run. Confirming improved AP requires the retry; the identified overflow alone does not quantify how much of the old loss it caused.

The separate retry is job799805 on acd_ue. Both gates passed on H10080GB: the old CUDA contraction produced32768 infinite denominators, while the fixed block's output and gradients were finite (maximum output difference from FP32 reference0.00035524). At the latest check it reached13 formal epochs with one AMP skip and one cumulative sanitized gradient element, not the severe recurring overflow of the original run. Current best macro/pooled AP is0.724127/0.752397 at epoch13; these are unfinished-run diagnostics, not a final result. Detailed retry provenance is in C1_precision_retry_20261104.json. No Torch packages or other runs were changed.

Detailed paths, hashes and pooled-selected points are in geometry_screen_20261104_results.json.
