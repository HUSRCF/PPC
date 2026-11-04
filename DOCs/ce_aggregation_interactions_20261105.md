# Prospective CE Aggregation x Geometry Controls

Rationale from completed50-epoch validation runs at seed49:

| One-block control | Selected epoch | Macro AP | Pooled AP |
|---|---:|---:|---:|
| K1 chain-mean CE | 22 | 0.7499782548820884 | 0.7830735423491580 |
| K7 residue-mean CE | 41 | 0.7579300472491337 | 0.7849030934897115 |

This is a small positive selected-validation signal, not independent causal
or seed confirmation. It motivates extending the loss aggregation axis to
the stronger two-block geometry models, without new packages or architecture.

| New arm | Matched completed control | Only optimization change |
|---|---|---|
| K28 twohop/post-TCN/residue-mean | K3 twohop/post-TCN/chain-mean | CE loss_reduction=residue_mean |
| K29 twohop/pre-TCN/residue-mean | K18 twohop/pre-TCN/chain-mean | CE loss_reduction=residue_mean |

K3,K18,K28,K29 form the complete aggregation x fusion-position four-cell
matrix. K1,K7,K3,K28 additionally form the one/two-block x aggregation matrix.
Both new arms retain ratio-MSE weight0.75,label smoothing0,all data/features,
model parameters,initialization,optimizer settings,batching and50epoch budget.
Changing within-batch CE aggregation is not identical to directly optimizing
pooled AP;dynamic batches and the unchanged chain-ratio penalty still matter.

Use seed49 from scratch,4868train/1075val/0test,corrected91D OF3,ESM-C3456D
MLC and frozen768D MSA Transformer. Batch64,tokens65536,workers8,prefetch4,
pinned/persistent/cache8 settings remain fixed. No seed controls or checkpoints
are resumed. OldK0-K27 generated configurations must remain unchanged.

Submit acd_ue,one GPU/12CPUs/192GiB/12h per arm,array0-1 without a percent
cap. Require the reviewed two-update CUDA gate and frozen source/config hashes.
Every epoch and both AP-best checkpoints are retained. Final selection uses
maximum validation pooled AP with same-epoch macro AP>=0.7417948466002535;
ties are earliest. Failed/incomplete runs or incomplete1075/404267validation
coverage cannot enter final readouts.

Before training/readout, declare only equal-logit thirds G2e27+MSA_De13+each
new model's eligible selected epoch,epsilon1e-7. No new mixture search or
test-oracle gate. All analyses remain adaptive validation evidence. The0.80
milestone contract still requires freezing before test,then an RCSB single-
chain union training attempt;no RCSB training is started by this matrix.
