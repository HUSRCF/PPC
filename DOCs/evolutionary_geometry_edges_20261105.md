# Evolution-conditioned geometry attention

## Why add this control

The two prespecified geometry/MSA validation blends improved both AP axes. G2+D MSA reaches macro0.759398 / pooled0.790671, compared with G2 alone0.746795 /0.780250. This is evidence of system-level error complementarity, not proof that MSA alone caused the gain: the source systems have other descriptor/architecture differences.

The current K1/G2+MSA model injects MSA into residue nodes. A small additional operator can test whether MSA should also influence which existing spatial neighbors exchange information. This is a proposed PPIS adaptation, not a claim of a new established method or guaranteed AP improvement.

## Operator

Keep the exact G2 sparse OF3 graph and its distance/orientation/confidence features. For projected MSA features u_i in64D, normalize then apply separate learned query/key projections into8 heads of rank8. L2-normalize each head. Add to the existing attention logit:

    evolutionary_bias_ijh = 4 * tanh(alpha_h) * cosine(Q_h(u_i), K_h(u_j))

Use the same destination-query/source-key direction as G2. Each head bias is bounded by[-4,4]. Alpha starts at zero, so the initial geometry function is unchanged. Preserve constructor RNG around the new module so all existing backbone parameters remain identical to the corresponding K1/K3 control at initialization.

This adds8328 parameters per geometry block, no new features, no new edges, no new structure sources and no new dependency. Existing edge bounds, sample boundaries and padding checks run before MSA edge lookup. Missing MSA features are an error, not a silent fallback. All normalization and dot products for the bias execute in FP32 after the projected q/k values; no sum over sequence length is introduced.

## Matrix

|Arm|Matched control|New behavior|Added parameters|
|---|---|---|---:|
|K9_G2_MSA_evo_edges|K1, one geometry block|MSA-conditioned edge bias|8328|
|K10_G2_MSA_evo_edges_twohop|K3, two geometry blocks|Independent MSA-conditioned bias in each block|16656|

Keep seed49,50 epochs from scratch, original union-v2 split, corrected91D OF3, ESM-C MLC and frozen768D MSA. D loss and optimized batch/loader contract remain unchanged. Select validation pooled AP, retain macro-best separately and require same-checkpoint macro>=0.741795 for promotion. No test evaluation, new seeds or weight sweep.

Acceptance uses three real training updates instead of two: the geometry output begins at zero, so the first update opens the geometry path, the second can update the zero alpha, and the third can train query/key projections. Set weight decay0 only in this acceptance smoke so parameter-change checks cannot be explained by decay. Formal training keeps weight_decay0.1.

The running799902 K0-K8 snapshot is immutable. These two variants must run from a new source/config/output directory; do not regenerate or duplicate the existing nine jobs.

## DeltaNet status

HPC3's direct PyPI metadata request failed with a network address error. Wheels were subsequently prepared locally and an offline dependency dry run passed; see [the separate installation plan](deltanet_20261105/README.md). No package installation or environment modification occurred. Gated DeltaNet remains an approval-dependent isolated-environment candidate; this edge-bias trial uses only the already-installed PyTorch stack.
