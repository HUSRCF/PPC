# Prospective Pre-TCN Interaction Controls

The completed validation leader remains G2e27+MSA_De13+K18e31,
macro AP0.7689864022799697/pooled AP0.7997048805648456. Five completed
regularization arms did not improve that ensemble. K23 label smoothing0.02
did improve standalone pooled AP to0.7903642797571038, motivating a matched
fusion-position interaction rather than another broad dropout sweep.

| New arm | Matched post-TCN control | Only model-axis change vs control | Other defining setting |
|---|---|---|---|
| K25 preTCN+LS0.02 | K23 | MSA residual before TCN | ratio MSE weight0.75,label smoothing0.02 |
| K26 preTCN+pureCE | K11 | MSA residual before TCN | ratio loss0,label smoothing0 |
| K27 preTCN+width256 | K15 | MSA residual before TCN | MSA projection256/hidden128,ratio MSE0.75 |

K18 supplies the pre-TCN baseline. Together these fill the missing cells of
position x smoothing, position x ratio supervision, and position x MSA width.
Effects are not assumed additive. These are single-seed exploratory controls,
not independent confirmation or a promise of improved AP.

All runs start from scratch with seed49,50epochs, unchanged union-v2 labels
and SI component splits,4868train/1075val/0test, corrected91D OF3 inputs,
ESM-C3456D MLC and frozen768D MSA Transformer. They retain batch64,
tokens65536,8workers,prefetch4,pinned memory and persistent workers. No
packages,Torch,model implementation or trainer behavior changes are required.
K0-K24 generated configurations must remain byte-equivalent at the same root.

Submission uses acd_ue,one GPU/12CPUs/192GiB/12h per task, no artificial
array throttle, and absolute Python with `-u`. A frozen source archive and
reviewed two-update CUDA gate precede each formal training process. Failed
gates or terminal failures cannot enter completed-model comparisons.

Checkpoint selection remains maximum validation pooled AP subject to the
same-checkpoint macro AP>=0.7417948466002535;ties use the earliest epoch.
Retain all epochs and macro-best separately. Full1075queries/404267residues
and no validation skips must hold for all50epochs before final readout.

Before calculation, declare only three fixed equal-logit thirds:
G2e27+MSA_De13+each of K25,K26,K27 at its eligible selected epoch.
Clipping epsilon1e-7;no alpha,membership or test threshold search.
The attached JSON is the prospective policy. This remains an adaptive
validation campaign, not a fresh holdout. If any completed candidate reaches
full-precision0.80, freeze before test and follow the existing post-milestone
contract;remaining jobs may finish as validation analyses only.
