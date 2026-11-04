# Completed Width128 Control and Width-Screen Closeout

K14 completed all50epochs in2h01m08s, Slurm800186_0 COMPLETED exit0:0.
Every epoch had1,075validation queries/404,267residues, no validation skips
and no test evaluation. One AMP-skipped update/one sanitized gradient element
were recorded. Epoch19 maximized eligible pooled AP;macro-best epoch25 is
retained separately and must not be mixed into the epoch19 result.

| Completed arm | Projection/hidden | Selected epoch | Macro AP | Pooled AP |
|---|---|---:|---:|---:|
| K14 | 128/128 | 19 | 0.753184 | 0.782841 |
| K16 near-capacity control | 64/228 | 31 | 0.752491 | 0.781085 |
| K15 | 256/128 | 32 | 0.758065 | 0.785722 |
| K17 near-capacity control | 64/428 | 20 | 0.754603 | 0.781691 |

K14 minus K16 pooled AP is+0.00175641;K15 minus K17 is+0.00403140.
These single-seed selected-validation differences motivate the wider-MSA
interaction cell, but do not establish independent statistical benefits.
All four width arms remain below K23 single-model pooled AP0.79036428.

The predeclared G2e27+MSA_De13+K14e19 equal-thirds blend gives macro
AP0.7642433030469789/pooled AP0.7963834506433967. It does not improve the
leader0.7997048805648456 and does not unlock test. All width-screen fixed
readouts are now complete;no extra mixtures/weights were searched.

See the same-stem resolved JSON for terminal and source hashes, and results
JSON for complete fixed-system metrics.
