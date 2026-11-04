# Fixed K3 Follow-up Blends

## Frozen Calculation

The [original policy](geometry_msa_followup_blends_20261105.json), SHA-256
`1423d8a45548f78790572d840c6cc2e091a3ec13f059471da60eaf0537011045`,
specified exactly two equal-logit blends. K3 selection was deferred until the
Slurm job completed all 50 epochs. Job `799902_3` completed successfully in
1h35m38s; all epochs retained 1,075 validation sequences / 404,267 residues,
with no skipped validation batches or test evaluation. One AMP update was
skipped and one gradient element sanitized across the run.

The rule selected **K3 epoch 21**, maximizing validation pooled AP subject to
macro AP >=0.7417948466002535 (earliest epoch breaks ties). The separately best
macro checkpoint is epoch 13: macro AP 0.761524 / pooled AP 0.784884. Do not
combine epoch-13 macro AP with epoch-21 pooled AP into a nonexistent result.

The [resolved source specification](geometry_msa_followup_blends_20261105_resolved.json)
contains the completed K3 receipt and immutable source paths. K3 final metrics
SHA-256 is `b6812358ccbd0d4fde2883fb23820e25875ae677b32f91c2e9a1233c3c78eed6`.

## Results

| System | Source epochs | Macro AP | Pooled AP | AUROC |
|---|---|---:|---:|---:|
| G2, geometry only | 27 | 0.746795 | 0.780250 | 0.886516 |
| MSA_D, original MSA model | 13 | 0.734337 | 0.760760 | 0.875200 |
| K3, two-layer G2 + MSA | 21 | 0.757084 | 0.789054 | 0.894820 |
| G2 + K3 | 27 + 21 | 0.760765 | 0.794711 | 0.896381 |
| G2 + MSA_D + K3 | 27 + 13 + 21 | **0.766068** | **0.798613** | **0.898284** |

The three-member system has validation F1 0.716431@0.40 and best validation MCC
0.612142@0.56. At the literal 0.50 threshold, F1 is 0.714618 and MCC 0.610627.
These are descriptive validation gates, not independently confirmed operating
points. Chain-macro raw ratio MAE is 0.049463.

The three-member system improves pooled AP by 0.009560 and macro AP by 0.008983
over the K3 source checkpoint. Relative to the previous fixed G2 + MSA_D blend,
the corresponding changes are +0.007943 and +0.006670. The gap to pooled AP 0.80
is **0.0013867793**. It has not reached the target; rounding must not turn this
into a claim of achieving 0.80. The strongest completed single model remains
K3 at 0.789054 pooled AP in the currently completed screen.

## What Was Not Searched

The two blends use equal means of logits reconstructed from probabilities
clipped to [1e-7, 1-1e-7]. They do not use raw stored logits. No weights, alpha,
extra combinations, source epochs or test gates were searched by this analysis.
Each source's AP was reconstructed within 1e-8 after checking complete UID,
sequence, union-label, residue-index and target alignment. All source metrics,
prediction hashes and analysis-code hashes are retained in
[the complete result JSON](geometry_msa_followup_blends_20261105_results.json).

The policy itself was chosen after observing the architecture campaign's
intermediate validation results. This is exploratory, selected-validation
evidence, not a preregistered independent test or a statistically confirmed
improvement. The ensemble also mixes descriptor/architecture variants, so its
gain cannot be attributed solely to MSA.

Remote outputs, including per-residue NPZ files:
`/data/user/shuang886/PPC_M18/sequence_union_v2_20261030/validation_diagnostics_20261105/fixed_K3_blends/`.
The existing reviewed analysis utility was reused without a code change.

## Next Architecture Shortlist, Not Submitted Here

1. **MSA residual bottleneck control.** On the completed two-layer K3 backbone,
   vary the current 64D MSA projection / 64D adapter hidden width to 128/128 and
   256/128. Keep all cached features, loss, data, seed and 50-epoch budget fixed.
   This tests information compression, rather than adding another context
   operator that the current full-attention/BiGRU controls have not supported.
2. **Separate evolution and geometry processing.** Preserve a geometry stream
   without early MSA mixing; process the MSA residual through its own small
   task-supervised branch, then fuse residuals late. Use a parameter-matched
   shared-stream control. Existing fixed-output complementarity motivates this
   hypothesis but does not prove a jointly trained version will retain it.
3. **Finish existing depth/loss controls first.** Three layers did not improve
   pooled AP over two; do not assume more graph layers alone will close the gap.
   Do not replace this experiment sequence with validation weight tuning merely
   to cross the numerical target.

No new seed or training matrix was submitted to produce this analysis. DeltaNet
remains a separate dependency proposal awaiting explicit isolated-install
approval; the original environment and core PyTorch packages are unchanged.
