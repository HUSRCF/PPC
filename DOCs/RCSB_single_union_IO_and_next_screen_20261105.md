# Native RCSB: I/O Evidence And Next AP Screen

## Fixed Scope

The PDBTM system is locked after validation pooled AP 0.8005469. Its
one-time test evaluation is complete. None of the work below changes that
ensemble, checkpoint epochs, weights or validation-selected thresholds.

The new native-RCSB target is the 251,267 exact observed-sequence union
universe. Current R0 uses all 201,013 training and 25,127 validation
queries, from scratch, with ESM-C MLC only. Its native test is disabled.
Three full epochs are the accepted pilot budget, not a statement that
three epochs establish convergence. Existing PDBTM test is not an
independent holdout for this new training universe.

## Actual Read-Only Runtime Evidence

The first full epoch reports 97.14% of its application-timer interval
waiting for batches. A five-second `/proc` sample independently observed
4 then 7 processes in uninterruptible I/O wait, 585 new major faults,
and 1,096,310,784 bytes read in 5.00535 seconds, approximately 209 MiB/s.
The original PID and worker counters, node and CPU affinities are in
`RCSB_single_union_pilot_20261105/runtime_io_probe_v1.json`, SHA-256
`1271811b3de0259f40dd1f0cac5fc14bb1b348d6ba122cbf4504698319fa81ad`.
This short observation is not a full profiler or a stable-bandwidth claim.

The training embeddings alone contain 822,425,688,576 float32 bytes;
validation adds 107,620,807,680 bytes. Together they are approximately
866 GiB before serialization/metadata. Dividing the training bytes by
the short observed read rate gives about 1.04 hours, consistent in order
of magnitude with the measured first-epoch training time. This does not
prove that every second of waiting is physical storage I/O.

The training node has about 14 TB available in `/tmp`. No full feature
bank has been staged, no global caches cleared and no current training
loader or configuration changed. Sixteen source tensors failed exact
FP16 and BF16 round trips, so no supposedly lossless dtype conversion
was made.

## Component Probe: Prepared, Not Executed Remotely

`scripts/analysis/benchmark_frozen_embedding_io.py` selects training-only
length ranks, copies a small number of immutable inputs to a temporary
directory, and compares shared/local and mmap/eager readers. It checks
float32 bit patterns, including signed zero; source and copied-file
SHA-256 values are retained. All temporary copies are cleaned up. It
uses `weights_only=True`, which was separately verified on a real
accepted source payload. Canonical IDs and explicit accepted source-bank
containment permit the input pack's intentional external symlinks while
rejecting traversal or unrelated targets. Source banks are not edited.

This measures load plus materialization, not full DataLoader throughput.
The operating-system cache is not cleared. Initial reads are not claimed
to be cold; repeated comparisons are warm and counterbalanced. Copy time
is explicitly warm-source timing, not a full cold-bank staging estimate.
The all-split sequence-length manifest contains aggregate label-count
columns, which are unused; no label tensor is opened and only training
IDs determine the sampled queries. These boundaries are recorded in the
receipt rather than claiming that no label-related metadata was read.

The first independent gpt-6-luna max review requested tighter safe loading,
source-path boundaries, signed-zero equality and copied-hash retention.
Those changes have six passing focused regressions. The independent
re-review could not resume because the agent service reported its thread
limit. Therefore **the revised component probe has not been run remotely**
and has no performance result or acceptance claim. This limitation does
not stop existing reviewed training or folding jobs.

## Next Controlled AP Experiments

Only native validation selects checkpoints and any predeclared ensemble.
Native test remains sealed. Keep seed 49, exact query universe, union
labels and the plain chain-mean CE initially fixed. Do not introduce
sampling, loss and input changes in the same comparison.

| Candidate | New information | Gate before launch |
|---|---|---|
| R0, running | ESM-C MLC only | Complete its accepted three epochs and read the curve |
| R1 | Corrected raw PS descriptors plus absolute confidence | Full source/QC coverage, consistent secondary-structure backend, train-only normalization |
| R2 | R1 plus geometry-aware residue message passing | Same inputs as R1, exact coordinate order, audited graph construction |
| R3 | Best eligible geometry backbone plus frozen MSA representation | Native MSA coverage and residue alignment accepted; do not silently use only the old PDBTM subset |
| Fixed logit ensemble | Complementarity of eligible native-validation candidates | Same queries/labels, simple weights fixed on validation only |

R1/R2/R3 are planned conditions, not submitted jobs and not available AP
results. Missing structures or MSA may not silently remove native queries.
If an explicit missing-feature-mask model is desired, name it as a
separate input condition rather than calling it the complete M18 system.
The 85 historical failures are all native train; they are retained in R0.
The standalone DSSP tool is still approval-pending and was not installed.

The immediate performance decision is to test exact-value node-local
staging before requesting more GPUs or increasing batch size. A component
gain would still require an actual full-loader test. Future staging must
use a new immutable input snapshot, source hashes and a startup gate; do
not rewrite the current running pack in place. Further epoch budget
should follow the R0 training/validation trend, not a test-set readout.

## Long-Query Repair Boundary

The full 4,579-by-4,579-by-139 relative-position operator matched actual
stock CPU output exactly on GPU. This resolved the original allocation
stage, not all model memory requirements. First attempt of job 801226
reached `DiffusionModule.prepare_cache`, then failed in
`linear_no_bias_z(layernorm_z(pair_z))`, requesting 10.00 GiB with
8.57 GiB free; PyTorch reported 67.70 GiB allocated and 2.22 GiB reserved
but unused. Its second bounded attempt is running. No successful
4,579-aa structure is claimed and no sequence was cropped.

If that attempt fails, a separate allocator-only trial is a reasonable
next diagnostic because the later memory profile differs from the first
one-hot failure. It must retain model weights, random seed, complete query
and inference parameters and re-pass the operator proof. No such trial
has yet been submitted; no installed Torch/Protenix file was changed.
