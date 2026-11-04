# Completed K17 Capacity Control

K17 completed50epochs, exit0, with all1,075 validation queries /404,267 residues
each epoch, no validation skips or test evaluations. It recorded one AMP-skipped
update and one sanitized gradient element. The original fixed-blend policy was
followed without weight or additional-combination search.

| System | Epoch | Macro AP | Pooled AP |
|---|---:|---:|---:|
| K15, projection256 / hidden128 | 32 | 0.758065 | 0.785722 |
| K17, matched capacity projection64 / hidden428 | 20 | 0.754603 | 0.781691 |
| G2+MSA_D+K15 | 27+13+32 | 0.768019 | 0.798762 |
| G2+MSA_D+K17 | 27+13+20 | 0.765060 | 0.794999 |

K15 exceeds its near-parameter-matched K17 control by0.004031 pooled AP and
0.003462 macro AP. Both single models remain below K3 pooled AP0.789054.
This is a single-seed selected-validation observation, not independent proof
of a wider-MSA benefit; initialization/dropout RNG differs with architecture.
K14 is still required to finish the other matched pair against K16.

At this width-control readout, the highest completed fixed-system validation
point was0.7987617942906267. The later K18 readout reached0.7997048805648456,
still below0.80; see `geometry_position_depth_20261105_results.md`.
Test stays closed under the user's milestone contract. See
[source/checkpoint selection receipt](msa_bottleneck_width_20261105_K17_resolved.json)
and [full fixed-system metrics and hashes](msa_bottleneck_width_20261105_K17_results.json).
