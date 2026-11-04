# Completed Geometry Depth x MSA Position Control

K18 (two geometry blocks; MSA before the TCN) completed all50epochs in46m21s,
Slurm800231_0 COMPLETED exit0:0. Every epoch retained1,075 validation queries
and404,267 residues, with no validation skips and no test evaluation. Training
recorded one AMP-skipped update and three sanitized gradient elements.

Checkpoint selection used validation pooled AP subject to the existing
same-checkpoint macro AP guardrail0.7417948466002535. The prospective equal
thirds G2+MSA_D+K18 blend was evaluated only after successful completion;
no weights, additional combinations or test outcomes were searched.

| System | Selected epoch(s) | Macro AP | Pooled AP |
|---|---:|---:|---:|
| K3: two blocks, MSA after TCN | 21 | 0.757084 | 0.789054 |
| K18: two blocks, MSA before TCN | 31 | 0.759645 | 0.788916 |
| G2+MSA_D+K3 | 27+13+21 | 0.766068 | 0.798613 |
| G2+MSA_D+K15 | 27+13+32 | 0.768019 | 0.798762 |
| G2+MSA_D+K18 | 27+13+31 | **0.768986** | **0.799705** |

The last row is the highest completed fixed-system validation point observed
so far: pooled AP **0.7997048805648456**, gap to0.80 **0.0002951194351544**.
It does not meet the milestone and must not be rounded into success.
The same output has AUROC0.898287719, validation F1 0.717471@0.49,
MCC 0.614268@0.60 and diagnostic ratio MAE0.05473588.

K18 macro-best epoch17 has macro AP0.76149418 but pooled AP0.77955665;
those values must not be substituted into the epoch31 row. The single-model
pooled AP is essentially unchanged relative to K3; the stronger fixed blend
suggests validation error complementarity, not proven standalone improvement.
All comparisons are single-seed, adaptively selected validation observations,
not independent evidence or selection-adjusted statistical claims.

The test remains closed. Six regularization arms and K14 were still running
at the2026-11-04T19:20Z snapshot. See the
[resolved selection and source paths](geometry_position_depth_20261105_resolved.json),
[full fixed-system metrics and provenance](geometry_position_depth_20261105_results.json),
and [post-target contract](pooled_AP080_milestones_20261105.md).
