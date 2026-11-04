# Completed geometry follow-ups

All values below are validation-only on the same 1,075 union-v2 sequences and
404,267 residues. These are seed49, from-scratch 50-epoch runs. Each row uses
the single checkpoint selected by pooled AP; macro-best is retained separately
in [the machine-readable record](geometry_completed_followups_20261105.json).
No test evaluation was performed. This is an exploratory architecture screen.

| Model | Main change | Epoch | Macro AP | Pooled AP | AUROC | F1@validation gate | MCC@validation gate |
|---|---|---:|---:|---:|---:|---:|---:|
| K10 | Two geometry blocks + MSA-conditioned edge bias | 32 | 0.756124 | 0.788385 | 0.893081 | 0.709573@0.32 | 0.600792@0.52 |
| K12 | Three geometry blocks, original ratio-regularized loss | 40 | 0.761793 | 0.786104 | 0.891058 | 0.704980@0.22 | 0.592224@0.30 |
| K13 | Three geometry blocks, pure chain-mean CE | 16 | 0.756258 | 0.782729 | 0.890016 | 0.702703@0.30 | 0.593941@0.47 |

K10 did not exceed the currently observed K3 two-block pooled AP of 0.789054;
K3 was still running when this three-model record was assembled. The extra
edge prior does not automatically improve a deeper backbone. K12 has the best
macro AP among these three rows, but adding a third block did not improve
pooled AP over the observed two-block result. K13 also does not establish a
monotonic benefit from removing the ratio penalty as geometry depth increases.
The two-block pure-CE control K11 remains necessary to finish that small matrix.

All three completed jobs exited 0 and retain all 50 consecutive epoch records,
with full validation coverage every epoch and no skipped validation batches.
K10 recorded two AMP-skipped updates and three sanitized gradient elements;
K12/K13 each recorded one and one. These are disclosed numerical-health
diagnostics, not the severe repeated overflow observed in the rejected old C1.

K13 is the strict fresh-output retry of a terminal pin-memory CUDA error on
ACD1-28. The retry only excludes that node; it does not change the training
configuration or establish that node hardware caused the original failure.
Original failed outputs remain preserved. Full paths, job IDs, source commits
and final metrics hashes are recorded in the JSON.

F1 and MCC gates above are independently selected on this validation set and
are diagnostics, not newly confirmed test operating points. The campaign's
promotion rule is pooled AP with macro AP >=0.7417948466002535, not a search for
the largest F1/MCC from a different epoch.
