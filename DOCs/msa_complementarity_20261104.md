# Fixed MSA complementarity analysis

## Contract and provenance

- Current PDBTM exact-sequence union-v2, OF3 monomer input;1,075 validation samples /404,267 residues. Each sample is an exact sequence, not one repeated experimental chain context.
- Original55 validation SI30 components, resampled as complete blocks2,000 times. No test predictions or test labels read, no retraining, no ensemble weight search.
- Execution source commit e923cf5. Independent narrow review passed after correcting constant-correlation handling and aligning Top-L rounding with the current trainer. Full local suite:53 tests,51 passed and2 optional-asset skips. Real data checks reproduced source pooled/macro AP to1e-8 and verified UID, sequence hash, union-label hash, labels and residue indices.
- The two blends and their epochs were fixed by the user's proposal before these ensemble scores were inspected. Both weights are exactly0.5.
- Saved files contain probabilities, not raw logits. We clip to [1e-7,1-1e-7], transform to logits, average equally, then apply sigmoid. This is explicitly a clipped-probability logit blend.
- All F1/MCC gates below are validation-selected; their best gates are separate. Top-L uses ceil(L/denominator), matching current training.

## Fixed-point results

All non-ensemble rows use the named fixed epoch. E_encoder e28 is the pooled-selected Pareto endpoint, not its macro-selected e18.

| Fixed candidate | Macro AP | Pooled AP | AUROC | Best pooled F1@gate | Best pooled MCC@gate | Ratio MAE |
|---|---:|---:|---:|---:|---:|---:|
| D_none e27 | 0.7282 | 0.7605 | 0.8717 | 0.6844@0.34 | 0.5687@0.41 | 0.0552 |
| D_encoder e13 | 0.7343 | 0.7608 | 0.8752 | 0.6838@0.50 | 0.5625@0.68 | 0.0746 |
| E_encoder e28 (pooled-selected) | 0.7302 | 0.7636 | 0.8737 | 0.6867@0.38 | 0.5739@0.54 | 0.0538 |
| Equal D_none27 + D_encoder13 | 0.7414 | 0.7722 | 0.8802 | 0.6927@0.42 | 0.5790@0.54 | 0.0503 |
| Equal D_encoder13 + E_encoder28 | 0.7386 | 0.7709 | 0.8798 | 0.6919@0.45 | 0.5767@0.57 | 0.0550 |

The baseline+MSA blend is stronger than the two-Pareto-member blend on both AP metrics. Relative to D_none, its gains are+0.0131284 macro AP and+0.0116802 pooled AP. It also exceeds D_encoder's macro point estimate by+0.0070182 and E_encoder e28's pooled point estimate by+0.0085903. These are validation comparisons, not independent test results. MAE is retained as a diagnostic, not the promotion criterion.

## Descriptor controls

Both completed50 epochs with zero skipped validation batches and no test evaluation. Values use each run's maximum validation macro AP checkpoint.

| Control | Epoch | Macro AP | Pooled AP | Ratio MAE |
|---|---:|---:|---:|---:|
| B0_fixed90_seed49 | 27 | 0.7261 | 0.7563 | 0.0592 |
| B1_fixed91q_seed49 | 18 | 0.7271 | 0.7554 | 0.0645 |

B0/B1 do not improve both AP metrics over D_none. Correcting descriptor semantics is not evidence that AP must increase. B1 versus B0 changes input dimension and constructor RNG consumption, so it is not a parameter-matched causal estimate of pLDDT alone. Geometry G/C runs remain an architecture screen; partial curves in summary.json are progress snapshots, not final winner selections.

## Component-paired intervals

| Contrast | Macro AP difference [95% interval] | Pooled AP difference [95% interval] |
|---|---:|---:|
| D_encoder_minus_D_none | 0.0061 [-0.0062, 0.0224] | 0.0003 [-0.0139, 0.0230] |
| blend_baseline_msa_minus_D_none | 0.0131 [0.0066, 0.0219] | 0.0117 [0.0043, 0.0242] |
| blend_pareto_minus_D_encoder | 0.0043 [-0.0044, 0.0110] | 0.0101 [-0.0009, 0.0162] |

MSA alone is not resolved from D_none by these intervals. The baseline+MSA blend has intervals above zero relative to D_none on both metrics. The Pareto blend versus D_encoder does not exclude zero.

**Critical limitation:** these are fixed-prediction, selected-validation conditional percentile intervals. They do not correct checkpoint/architecture selection, do not include training-seed variance, and are not an independent confirmation of significance. The baseline+MSA versus D_encoder interval was not among this script's prespecified contrasts; its positive point difference must not inherit the D_none interval.

## Per-chain heterogeneity

D_none to D_encoder improves514/1,075 sequences (47.8%) and worsens561 (52.2%). Mean delta AP is+0.0061102 but median is-0.0014658. Gains in a minority of chains are larger than losses elsewhere; this is not a uniform improvement. Symmetrically trimmed means remain positive:5% each tail+0.0046767,10% each tail+0.0029460; these are descriptive, not additional hypothesis tests.

| Length | Samples | Mean delta AP | Median delta AP | Improved fraction |
|---|---:|---:|---:|---:|
| <200 | 126 | 0.0166 | 0.0095 | 61.1% |
| 200-399 | 607 | -0.0025 | -0.0060 | 38.7% |
| 400-799 | 294 | 0.0185 | 0.0060 | 56.5% |
| 800-999 | 27 | 0.0090 | 0.0059 | 70.4% |
| >=1000 | 21 | 0.0153 | 0.0220 | 81.0% |

There is no evidence here for a general long-chain failure: the21 length>=1,000 sequences all use two encoder windows, with mean delta AP+0.0153491 and17/21 improved. Their small, correlated cohort does not establish long-chain superiority. The200--399 aa group is the main negative stratum.

The0.25--0.50 positive-ratio group (578 samples) has mean delta AP-0.0027859; the0.10--0.25 group (416 samples) improves+0.0166381. These strata overlap the length groups and are not independent causal explanations.

## Probability scale and MSA depth

- Chain mean probability increases by0.0921798 on average, with across-chain standard deviation0.0386626. The shift is heterogeneous, not one uniform global monotonic rescaling.
- Length versus delta mean score Spearman=-0.3959, while length versus delta AP is0.0030. This supports examining cross-chain score comparability, but does not by itself explain pooled AP.
- Residue-weighted mean chain delta AP is+0.0077037; positive-count-weighted mean chain delta AP is+0.0034027. Neither is pooled AP or an exact decomposition of its change.
- Effective depth is mean Kish depth over informative columns, recovered from the cached profile's bounded log-depth channel. It is not HHsuite Neff. The table separately records unique aligned rows, actual encoder depth and number of windows.
-1,060/1,075 sequences have profile Kish depth>=128;1,069 encoders use depth64. Only6 use depth5/6. Thus the cohort is too unbalanced to robustly conclude that benefits depend on shallow versus deep MSA. The shallow subgroup's negative mean is descriptive only.

## Next decision

Keep the existing baseline+MSA fixed blend as a validation candidate. Do not replace the formal model or touch test based on these results alone. Finish the G/C screen; if a genuinely competitive architecture emerges, complete the B1/common-feature2x2 with B1+MSA and winning-architecture+MSA. New architectures must be compared to the old D/MSA frontier as well as the weaker B1 control. No extra training or seed jobs were submitted for this analysis.

## Assets

- Frozen compact statistics: DOCs/msa_complementarity_20261104/summary.json.
- Per-sequence decomposition and strata: per_chain_deltas.csv and stratified_deltas.csv in the same directory.
- All six packed validation NPZs exist locally under tmp/msa_complementarity_20261104 and on HPC3 at /data/user/shuang886/PPC_M18/sequence_union_v2_20261030/msa_factorial_20261104/analysis/complementarity_v1. They are not added as large Git blobs; source paths and hashes are retained in summary/index.

