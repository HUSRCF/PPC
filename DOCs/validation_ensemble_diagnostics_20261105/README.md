# Fixed-Ensemble Integrity and Conditional Uncertainty

## Scope and Reproduction

The three original sources are G2 e27, MSA_D e13 and K3 e21. The only two
ensembles are the previously specified equal-logit G2+K3 and G2+MSA_D+K3.
There is no new source, combination, weight, checkpoint or test search.

The analysis first rebuilt the same outputs from the original per-residue PT
predictions using the frozen source specification. Original prediction and
metrics hashes match. All five NPZ files, not only their AP values, have
**identical SHA-256 hashes** to the original cached files. Every UID, length,
offset, label and score array was also checked explicitly. See
[the reproduction receipt](reproduction_receipt.json).

The diagnostic then rechecked the 1,075-query / 404,267-residue validation
universe, frozen union-label hashes and AP reconstruction. Existing ensembles
were independently checked against their exact declared equal clipped-logit
formula before analysis. Only generic logit/sigmoid utilities are imported from
the legacy ensemble tool; its weight-search entrypoint is not invoked.

Execution source is commit `5af793878689343573a21bdb456e9aeeebd2434d`.
The scoped gpt-6-luna max review found no blockers; 14 focused tests passed.
The full current suite passed 84 tests with two optional asset-dependent skips.
No package or PyTorch dependency was changed.

## Results Relative to K3 e21

| System | Macro AP | Pooled AP | Improved / worsened chains |
|---|---:|---:|---:|
| K3 reference | 0.757084 | 0.789054 | reference |
| G2 + K3 | 0.760765 | 0.794711 | 685 / 390 |
| G2 + MSA_D + K3 | 0.766068 | 0.798613 | 739 / 336 |

The three-member ensemble improves 68.74% of the 1,075 chains. Its mean
per-chain AP change is +0.008983 and median is +0.007629, a descriptive
indication that the improvement is not confined to a few large positive
chain-level outliers.

## SI-Component Paired Bootstrap

Each of 2,000 replicates resamples the 55 original SI30 components and retains
all associated chains and residues. Point metrics remain chain-macro and
residue-pooled AP, not equal-component-weighted AP. The bootstrap seed is
20261104. The following are **intervals for the difference versus K3**, not
absolute AP intervals.

| Contrast vs K3 | Macro AP delta [95% CI] | Pooled AP delta [95% CI] |
|---|---|---|
| G2 + K3 | +0.003680 [-0.002537, +0.008306] | +0.005658 [+0.000773, +0.010407] |
| G2 + MSA_D + K3 | +0.008983 [+0.003168, +0.015401] | +0.009560 [+0.005202, +0.016386] |

The three-member intervals exclude zero on both axes **conditional on these
already-selected validation systems**. They do not account for checkpoint or
architecture selection, multiple comparisons or training-seed variation, and
are not independent confirmation or a proof of generalization.

## Where the Three-Member Ensemble Helps

| Chain length | Chains | Mean chain AP delta vs K3 | Fraction improved |
|---|---:|---:|---:|
| <200 | 126 | +0.011782 | 68.3% |
| 200-399 | 607 | +0.007378 | 73.1% |
| 400-799 | 294 | +0.012232 | 60.5% |
| >=800 | 48 | +0.002036 | 64.6% |

All length-bin mean effects are positive, but the longest group has a smaller
mean gain and only 48 chains. These subgroup summaries are descriptive, not
tests of between-group differences. The 25 chains with positive ratio <0.1
have mean delta -0.001504 and median -0.003068; no uniform-improvement claim
is justified.

MSA Kish depth is >=128 for 1,060 chains. Only 15 chains occur in lower-depth
bins, so their negative mean effects do not establish a reliable depth-gating
rule. Kish depth is not HHsuite Neff or the actual Transformer row count.

## Experiment Implication

The ensemble changes within-chain ranking as well as pooled ranking. Its gain
therefore cannot be explained solely by a global monotone probability
calibration. It supports retaining complementary systems and motivates a later
separate geometry/evolution-stream hypothesis. Different descriptors and
architectures among the source systems prevent attribution solely to MSA.

The pooled AP target **has not been reached**: 0.798613 is 0.001386779 below
0.80. K14-K17 test representation width versus matched capacity; K18 completes
the same-parameter fusion-position x geometry-depth matrix. Their full50-epoch
results and predeclared fixed blends must be read before changing direction.
No additional ensemble weight search was performed to cross the target.

## Assets

- `summary.json`: full effects, component intervals and runtime/source hashes.
- `per_chain.csv`, `stratified.csv`, `pairwise.csv`: detailed diagnostics.
- `execution.json`, `reproduction_receipt.json`: source/review and exact-data audit.
- [Frozen plan](../validation_ensemble_diagnostics_20261105_plan.json).

Raw reproduced point NPZs and frozen analysis code remain at
`/data/user/shuang886/PPC_M18/sequence_union_v2_20261030/validation_ensemble_diagnostics_20261105/`.
No inference, labels or test scores were changed by the diagnostic.
