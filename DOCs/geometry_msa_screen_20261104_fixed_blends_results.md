# Fixed geometry/MSA validation blends

Both blends were specified before calculation and use equal clipped-probability logit means (epsilon1e-7), not tuned weights or probability means. Source models and checkpoints were previously selected on validation. This remains exploratory validation evidence, not independent confirmation.

|System|Source epochs|Macro AP|Pooled AP|
|---|---|---:|---:|
|G2 geometry|27|0.746795|0.780250|
|D MSA Transformer|13|0.734337|0.760760|
|MSA before TCN|18|0.734161|0.766166|
|G2 + D MSA Transformer|27+13|**0.759398**|**0.790671**|
|G2 + MSA before TCN|27+18|0.757551|0.790578|

G2+D MSA improves macro AP by0.012603 and pooled AP by0.010420 relative to G2. The gap to the pooled AP0.80 objective is0.009329. Both AP axes improve rather than exchanging within-chain ranking for pooled ranking alone.

The different source systems have other architectural/descriptor differences, so this supports system-level error complementarity, not a causal attribution solely to MSA. It strengthens the motivation to test the new G2+MSA arms and, if needed, later separate geometry/evolution streams. No additional blend weights, checkpoints or combinations were searched here.

Alignment was checked on all1075 validation queries and404267 residues: full UID coverage, sequence SHA-1, union-label SHA-256, residue indices and target arrays. Each source macro/pooled AP was reconstructed to1e-8 before blending. No test data was read. Full metric sets, source hashes and spec/code hashes are in [the raw results](geometry_msa_screen_20261104_fixed_blends_results.json).

The separate gpt-6-luna max scoped review found no issue. The local69-test suite passed67 with2 optional asset skips; the reviewer could not reproduce that suite in its default ROCm environment because of an unrelated amdsmi binary error. The actual fixed-blend analysis completed successfully in the existing HPC3 openfold environment. No environment was installed or modified.
