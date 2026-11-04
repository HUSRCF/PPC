# Completed K26/K27 Validation Readouts

Both arms completed50epochs with exit0,full1075validationqueries /
404267residues,no validation skips and no test events. Exact terminal/runtime/
configuration/source/cohort/prediction hashes are recorded by the reviewed
completed-arm reader and the existing fixed-blend analyzer.

| Arm | Selected epoch | Macro AP | Pooled AP |
|---|---:|---:|---:|
| K26 pre-TCN/twohop/pure CE | 36 | 0.7550346663360386 | 0.7862070787197295 |
| K27 pre-TCN/twohop/MSA projection256 | 25 | 0.7589113448005674 | 0.7855999746536794 |
| Fixed G2e27+MSA_De13+K26e36 | - | 0.7684925298590854 | 0.8005468730628196 |
| Fixed G2e27+MSA_De13+K27e25 | - | 0.7678619432069664 | 0.7980211331282288 |

Only the previously declared equal-logit thirds were computed;no membership
or alpha search. K26's individual pooled AP is below K18 and K23,but its
predeclared fixed blend is more complementary on this selected validation
cohort. This is descriptive,single-seed adaptive-validation evidence,not a
causal or statistically confirmed superiority claim.

K26 was the first verified declared system to cross0.80 and was locked before
test. K27 remained validation-only and does not replace the locked result.
The fixed test outcome is separately retained in
`locked_AP080_K26_20261105.md`. Full K26/K27 completion audits,resolved specs
and metrics are in`preTCN_interactions_20261105/K26/`and`K27/`.
