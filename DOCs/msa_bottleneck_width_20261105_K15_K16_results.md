# First Completed Width/Capacity Readouts

K15 and K16 completed50epochs and exited0. The original fixed-blend policy
SHA-256 is `5111e4cc82359c2a1ef2e10ee7b83bea77029fe103425a0e35c2bf73cd3df233`.
Source selection used maximum validation pooled AP with macro guardrail
>=0.7417948466002535; no test or extra alpha search.

| System | Selected epoch | Macro AP | Pooled AP |
|---|---|---:|---:|
| K3 reference | 21 | 0.757084 | 0.789054 |
| K15, MSA projection256/hidden128 | 32 | 0.758065 | 0.785722 |
| K16, projection64/hidden228 | 31 | 0.752491 | 0.781085 |
| G2 + MSA_D + K3, existing fixed system | 27+13+21 | 0.766068 | 0.798613 |
| G2 + MSA_D + K15, fixed thirds | 27+13+32 | **0.768019** | **0.798762** |
| G2 + MSA_D + K16, fixed thirds | 27+13+31 | 0.766156 | 0.797028 |

The K15 ensemble's full-precision pooled AP is0.7987617942906267, leaving
0.0012382057093733 to0.80. It has not reached the test-evaluation trigger.
Its gain over the existing K3 ensemble is only0.000149 pooled AP; this is a
validation point estimate, not a statistically established improvement.

K15 and K16 are not a matched pair: K15's capacity control is K17, while K16
controls K14. Those counterpart runs are still in progress. Do not infer a
causal width-versus-capacity result from these two completed models alone.

All50epochs have full1,075-query /404,267-residue validation coverage, no
skipped validation batches or test evaluations. K15 recorded one AMP-skipped
update and one sanitized gradient element; K16 recorded two and three.
The saved source predictions reproduce their AP values and pass UID, sequence,
residue-index and frozen-label checks before the fixed ensembles are computed.

See [the resolved source specification and completion receipts](msa_bottleneck_width_20261105_K15_K16_resolved.json)
and [full metrics/source hashes](msa_bottleneck_width_20261105_K15_K16_results.json).
K15/K16 remain exploratory validation results, not a new frozen test result.
