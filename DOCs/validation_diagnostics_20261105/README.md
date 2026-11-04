# Selected-validation ranking diagnostics

## Contract

These are exploratory diagnostics of already-selected checkpoints, not an
independent confirmation or a new held-out test result. All 11 systems were
evaluated on the same 1,075 union-v2 validation sequences / 404,267 residues.
No test data, blend, or blend-weight search was used by this analysis.

The point exporter checked the complete UID set, sequence hashes, residue
indices and frozen union-label hashes. Each exported NPZ reproduced its source
macro and pooled AP within 1e-8. The diagnostic then rechecked NPZ ordering,
lengths, offsets, labels, score bounds and AP reconstruction. Covariate UID,
length, positive ratio and original SI component were checked before joining.

Source code: commit `9f28699297ad5f2dfcf4767a2c9d01b442fe899b`.
The scoped gpt-6-luna max review identified a nonfinite covariate-validation
gap; it was fixed and regression-tested before execution. The full local suite
subsequently passed 79 tests with two optional asset-dependent skips.

Remote analysis root:
`/data/user/shuang886/PPC_M18/sequence_union_v2_20261030/validation_diagnostics_20261105`.
The frozen point specification is [../validation_diagnostics_20261105_spec.json](../validation_diagnostics_20261105_spec.json).
Raw point NPZ files remain under the remote root's `points/`; the reproducible
diagnostic command is:

```bash
ROOT=/data/user/shuang886/PPC_M18/sequence_union_v2_20261030/validation_diagnostics_20261105
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /data/user/shuang886/.conda/envs/openfold/bin/python -u \
  "$ROOT/code/analyze_validation_error_complementarity.py" \
  --base /data/user/shuang886/PPC_M18/sequence_union_v2_20261030 \
  --points "$ROOT/points" --reference K1 --covariates "$ROOT/covariates.csv" \
  --output "$ROOT/diagnostics" --bootstrap 2000 --workers 8
```

The output directory must not already exist. The archived run is complete;
do not rerun into that directory.

## Same-Checkpoint Results

K1 is the one-layer G2 + MSA Transformer reference. The displayed values belong
to each system's selected checkpoint, rather than independently maximizing each
column. G2 and MSA_D are the historical fixed epochs 27 and 13.

| System | Macro AP | Pooled AP | Macro delta vs K1 | Improved / worsened chains |
|---|---:|---:|---:|---:|
| G2, no direct MSA | 0.746795 | 0.780250 | -0.003183 | 510 / 565 |
| MSA_D, old descriptor architecture | 0.734337 | 0.760760 | -0.015642 | 364 / 711 |
| D_none seed49 | 0.728226 | 0.760506 | -0.021752 | 369 / 706 |
| D_none seed50, previously submitted | 0.728187 | 0.756420 | -0.021791 | 297 / 778 |
| K0, corrected descriptors + MSA, no G2 | 0.733136 | 0.759938 | -0.016842 | 379 / 696 |
| K1, G2 + MSA | 0.749978 | 0.783074 | 0 | reference |
| K4, additional ESM2 contact graph | 0.749947 | 0.780176 | -0.000031 | 466 / 609 |
| K6, full sequence attention | 0.746651 | 0.780222 | -0.003328 | 527 / 548 |
| K7, residue-mean CE | 0.757930 | 0.784903 | +0.007952 | 620 / 455 |
| K8, pure chain-mean CE | 0.750891 | 0.786341 | +0.000912 | 590 / 485 |
| K9, MSA-conditioned geometric attention | 0.756167 | 0.787362 | +0.006189 | 656 / 419 |

## Component Bootstrap

There are 55 original SI30 validation components. The largest contains 210
sequences; the next largest contain 69, 63, 59 and 59. Each of 2,000 paired
replicates resamples whole components and retains all their chains and residues.
The metrics themselves remain chain-macro AP and pooled AP: these are not
equal-component-weighted AP estimates.

| Contrast | Macro AP delta [95% CI] | Pooled AP delta [95% CI] |
|---|---|---|
| G2 - K1 | -0.00318 [-0.01286, +0.00416] | -0.00282 [-0.01018, +0.00285] |
| K7 - K1 | +0.00795 [-0.00147, +0.01778] | +0.00183 [-0.00540, +0.01057] |
| K8 - K1 | +0.00091 [-0.00617, +0.00650] | +0.00327 [-0.00173, +0.00833] |
| K9 - K1 | +0.00619 [+0.00216, +0.01124] | +0.00429 [-0.00010, +0.01032] |

K9 has a positive conditional macro interval, but its pooled interval still
crosses zero. These intervals condition on validation-selected checkpoints;
they do not correct for checkpoint/architecture selection, multiple comparisons,
or training seed variation. They must not be promoted to a confirmatory
significance claim.

## Where the Changes Occur

K9's mean per-chain AP changes are positive in all four length bins:

| Length | Chains | K9 - K1 mean AP | Fraction improved |
|---|---:|---:|---:|
| <200 | 126 | +0.00921 | 54.8% |
| 200-399 | 607 | +0.00489 | 61.8% |
| 400-799 | 294 | +0.00844 | 62.2% |
| >=800 | 48 | +0.00089 | 60.4% |

These are descriptive chain-macro changes, not stratum-specific pooled AP or
confidence intervals. K9 gains are weak for the longest group and negative in
the small positive-ratio <0.1 group (25 chains, mean -0.00499). K7 gains most in
the <200 group (+0.02499), while its >=800 group has mean -0.00468. Thus
residue-mean weighting does not uniformly improve every length group.

MSA depth here is the existing profile's **Kish effective depth**, not HHsuite
Neff, nor the actual number of MSA rows given to the Transformer. There are
1,060 chains with depth >=128, nine with depth 32-127 and six below eight.
No depth 8-31 or missing-depth chains occur. This distribution does not justify
a strong shallow-MSA subgroup claim or prioritizing a quality gate from this
validation analysis alone.

## Complementarity and Next Decisions

Mean within-chain score Spearman is 0.84576 between G2 and MSA_D, versus 0.97923
between K1 and K9. This suggests the separate geometry and evolution systems
have more diverse rankings than the small edge-prior modification, but ranking
correlation alone does not establish useful error complementarity. The existing
fixed G2 + MSA_D ensemble independently supplies the relevant observed evidence:
macro AP 0.759398 / pooled AP 0.790671, still below 0.80.

1. Finish the already-running two-/three-layer depth controls before expanding
   the training matrix. Evaluate only the two fixed K3 blends defined in
   [the follow-up policy](../geometry_msa_followup_blends_20261105.json), after
   K3 completes all 50 epochs.
2. Preserve the one-/two-layer geometry + MSA route; K9 is a useful candidate,
   not a statistically established pooled-AP improvement.
3. If the target remains unmet, the next focused architecture hypothesis is
   separate geometry/evolution residual streams with late supervised fusion,
   accompanied by a parameter-matched control. A narrower alternative is to
   test the current MSA adapter bottleneck (64 versus 128/256), not a broad
   unconstrained search over ensemble weights.
4. Do not prioritize generic full-sequence attention, additional ESM2 graph
   propagation, or an MSA depth gate solely from these results. No such new
   training task was submitted to generate this diagnostic.

## Files

- `summary.json`: source hashes, full effects, bootstrap and interpretation.
- `per_chain.csv`: chain AP, score means, covariates and deltas for all systems.
- `stratified.csv`: length, positive-ratio and MSA-depth subgroup summaries.
- `pairwise.csv`: between-system pooled and within-chain score correlations.

No test result, experimental guarantee, or new model replacement is implied.
