# Prospective TCN Depth x MSA Position Screen

The frozen ESM-C and MSA Transformer already encode sequence context. The
question here is narrower:whether additional task-supervised local processing
helps the current two-block OF3 geometry system. No assumption of insufficient
capacity is established by the existing validation scores.

| New arm | Matched completed control | Combined architecture change |
|---|---|---|
| K30 twohop/post-TCN/TCN4 | K3 twohop/post-TCN/TCN2 | TCN layers2->4;dilations[1,2]->[1,2,4,8] |
| K31 twohop/pre-TCN/TCN4 | K18 twohop/pre-TCN/TCN2 | Same TCN change;MSA enters before TCN |

Kernel7 gives an added TCN receptive-field span of
`1+6*(1+2)=19` versus`1+6*(1+2+4+8)=91` residues. This is the downstream
convolutional span,not the total model context:PLM embeddings and geometry
propagation already carry longer-range information. Both depth and parameter
count increase;this is not a parameter-matched or pure receptive-field causal
test. K30/K31 themselves have identical initial parameters and constructor RNG,
and differ only in MSA position. The actual task remains one full chain per
sample,not concatenated partners or an assembly-conditioned input.

Keep all data/features,train-only normalization,loss,optimizer,batch packing,
dropout and seed49 unchanged relative to K3/K18. In particular,use chain-mean
CE,ratio-MSE weight0.75,label smoothing0 and scratch50epochs;no new seed
controls,resume,packages or model implementation edits. Existing K0-K29
generated configs must be byte-equivalent at identical output roots.

Local preflight confirms byte-equivalence for all30previous configurations.
The matched2TCN-layer systems contain3693034trainable parameters;K30/K31
each contain4360682,an increase of667648. This capacity difference is retained
as an explicit interpretation limit,not hidden as a parameter-matched test.

Narrow read-only gpt-6-luna max review passed. The reviewer independently
confirmed old-config byte equivalence,matching initial K30/K31 tensors/RNG,
16focused tests,and additional ragged97/61-residue checks for padding
invariance,finite gradients and off-center dilation/TCN/MSA/geometry learning.
The complete local suite passed110tests with2optional skips. Model and
training implementations were not changed.

Use corrected91D OF3,ESM-C3456D MLC and frozen768D MSA Transformer. Batch64,
token budget65536,workers8,prefetch4,pinned/persistent/cache8 remain fixed.
Submit one GPU/12CPUs/192GiB/12h per arm on acd_ue,array0-1 with no percent
cap. First require the existing real-payload two-update CUDA gate,including
finite gradients and learning through both geometry blocks and the zero-init
MSA branch. Local tests additionally exercise all4TCN convolution updates,
masked outputs,actual dilations and the paired constructor-state contract.

Only COMPLETED exit0 runs with50consecutiveepochs,full1075queries /
404267validationresidues and no skipped validation batches can enter final
readouts. Select max validation pooled AP subject to the existing same-point
macro AP guardrail0.7417948466002535;earliest ties. Retain macro-best separately
without mixing epochs. Before training/readout,declare only equal clipped-
probability logit thirds:G2e27+MSA_De13+each selected new model,epsilon1e-7.
Do not search weights or modify membership after observing these results.

Interpret positive or negative differences within this adaptive,single-seed
validation campaign only. A more expressive TCN can overfit or be redundant
with frozen encoders;no improvement is presumed. Any0.80 result still requires
the existing validation lock,strict GPU replay and fixed test evaluation,
followed by a whole-RCSB single-chain union training attempt. Test remains
closed and no RCSB training is started by these two experiments.
