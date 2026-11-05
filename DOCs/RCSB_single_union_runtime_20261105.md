# RCSB Single-Chain Whole-Data Pilot: Actual Start

Observed at 2026-11-05 01:56 UTC. This is the next stage after the frozen
PDBTM validation target and its one-time test evaluation, not another test
selection round.

## Frozen PDBTM Result

The predeclared equal-logit ensemble is G2 epoch 27, MSA-D epoch 13, and
K26 epoch 36, with weight 1/3 each. Validation pooled AP is 0.8005469.
Its fixed test pooled AP is 0.7693981, macro AP 0.7229288, AUROC 0.8994801,
F1 0.6930927 at the validation-selected 0.50 gate, and MCC 0.5958782 at
the validation-selected 0.61 gate. No test-based replacement, threshold
search, or additional ensemble selection was performed. The operational
0.80 target was reached by an ensemble, not by a single checkpoint.

The locked result and prediction artifacts remain under
`locked_AP080_K26_20261105/`.

## Actual RCSB Run

| Item | Actual value |
|---|---:|
| Native exact observed sequences | 251,267 |
| Train | 201,013 |
| Validation | 25,127 |
| Native test, held out and not loaded | 25,127 |
| Total residues across native splits | 74,646,793 |
| Initial train-batch estimate | 3,170 |
| Actual epoch-1 successful train batches | 3,333 |
| Validation batches | 396 |
| Test batches | 0 |
| Trainable parameters | 2,379,010 |
| Seed / full epoch budget | 49 / 3 |

Job **800877**, partition `acd_u`, node `ACD1-54`, uses an H100 80GB.
The complete input-reference and metadata/label identity checks passed.
The downstream optimizer is verified running: first logged step 25;
step 150 was observed at the timestamp above. No full-epoch validation
result is available yet, so this document does not claim a RCSB AP result.

Initialization is from scratch, not a PDBTM supervised warm start. Labels
are exact-sequence, all-partner, ever-observed binary unions under the
observed-polymer heavy-atom 5.5-A assembly contract. Partners are not
filtered for length or feature availability before label union.

Input is frozen ESM-C 600M MLC, layers 12/24/36, 3,456 dimensions. This
first pilot intentionally has **no structure, ESM2-contact, or MSA branch**.
It is not the full M18 geometry/MSA system and is not a controlled estimate
of the causal effect of expanding only the training set. The configuration
uses all native train/validation queries without cropping or subsampling.

Plain chain-mean CE is used without class weighting, smoothing, ratio
loss, or an auxiliary ratio head. The pilot selects by native validation
pooled AP while recording macro AP and probability diagnostics. Native
test is disabled. Old PDBTM test contains native-RCSB training sequences,
so it cannot be claimed as independent for a future RCSB-trained model.

The existing split uses detected SI30 / bidirectional cov80 connections
and same-PDB components. This is not a new exhaustive global Biopython
pairwise-identity guarantee.

## Evidence And Paths

Archived actual-start receipt and runtime configuration:

- `RCSB_single_union_pilot_20261105/formal_training_started.json`
- `RCSB_single_union_pilot_20261105/GPU_started.json`
- `RCSB_single_union_pilot_20261105/runtime_config.json`
- Input seal and original submission receipts in the same directory.

HPC3 runtime root:
`/data/user/shuang886/PPC_RCSB_single_union/20261105`.

Actual model outputs:
`training/R0_esmc_only_pilot_seed49/`.

Frozen code and startup state:
`whole_pilot_20261105/code/` and `whole_pilot_20261105/state/`.

## Parallel Structure Work

All eight Star-to-HPC3 CIF/confidence transfers completed with exit zero.
The independent existing-structure content audit reached 187,000 accepted
queries out of 224,276, with zero reported failures/pending rows at this
observation. Final inventory acceptance is still required; transfer or
partial audit success alone does not permit geometry training.

Short-structure repair was moved from array 800847 to **801016** solely
to remove repeated diagnostic-directory scans. Folding model, input
manifests, seed, precision, strict QC, and retry budgets are unchanged.
The old queue contracts and complete retry ledgers were preserved.
Eight real outputs gave identical old/new QC and error matches, and
the four validation/input functions have identical ASTs.

Source commit `22ee880` passed gpt-6-luna max focused review, 14 focused
tests, and 173 full-suite tests with two optional skips. A measured old
per-query error scan took 1.8--4.4 seconds; the update scans once per
completed process chunk. End-to-end speedup is not yet established.

The 85 historical structure failures remain explicitly separate; 82
exceed the installed ESM process's 4,094 truncation setting. Full-length
rescue has not been launched and no sequence has been shortened.

## First Full Native Validation

Observed at 2026-11-05 03:28 UTC. Job 800877 remains running; epoch 2
has passed step 1,550. The complete first epoch processed all 201,013
training queries and 59,492,599 residues. The native test remains disabled.
Length-bucket shuffling changes actual batch packing between epochs;
3,333 actual epoch-1 batches versus the initial 3,170 estimate is not a
change in the sequence universe.

| Native validation, epoch 1 | Value |
|---|---:|
| Queries / residues | 25,127 / 7,785,070 |
| Pooled AP | 0.5729257 |
| Chain-macro AP | 0.5618773 |
| AUROC | 0.8109841 |
| Chain-mean CE | 0.4727979 |
| F1 / MCC at 0.50 | 0.5339438 / 0.4168759 |
| Validation best F1 | 0.5487913 at 0.35 |
| Validation best MCC | 0.4177860 at 0.45 |
| Ratio MAE / RMSE | 0.0916856 / 0.1222054 |
| Ratio bias | +0.0270539 |
| Top-L/20 macro precision | 0.6476543 |

These are native validation diagnostics, not test results and not frozen
final gates. This scratch, ESM-C-only, first-epoch pilot must not be
directly ranked against the converged PDBTM geometry/MSA ensemble. It
cannot isolate the causal effect of adding RCSB training data.

The training chain-mean CE was 0.4423475. There were 3,332 optimizer
updates, one AMP-skipped update and two sanitized gradient elements in
3,333 successful batches; no samples or batches were dropped as nonfinite.
This rare event is recorded rather than silently treated as zero.
Validation gradient-isolation checks passed: gradient mode disabled,
logits without gradients, zero parameter gradients and zero parameter
version changes. Predictions contain all 25,127 validation queries.

The original epoch record, including all validation threshold diagnostics,
is archived in `RCSB_single_union_pilot_20261105/metrics_observed.jsonl`.
Its runtime configuration and original input seal are in the same folder.

### Data-Loading Performance

The epoch-1 training loop took 3,537.26 seconds, with 3,436.23 seconds
reported waiting for batches (97.14%) and 101.03 seconds in its compute
timer. These application timers are not a substitute for a full GPU
profiler. Independent short GPU-utilization observations also showed
little activity. The complete train-plus-validation epoch took 4,056.83
seconds. Current training is not hot-patched or restarted to optimize I/O.

Sixteen native-train feature tensors were checked for exact BF16 and
FP16 round-trip equality. All sixteen failed both checks; no conversion
or feature-bank rewrite was made. See `lossless_dtype_probe16.json`.
The loader currently uses memory-mapped Torch files and materializes
float32 features. Future optimization should compare reader behavior
and node-local staging without changing values or query order. A read-only
check found approximately 14 TB free on this training node's `/tmp`;
no bank has been copied there and no storage-speed benefit is claimed.

### Structure Work Updated

The independent existing-structure audit and full confidence-slice
readback have now accepted all 224,276 existing queries, without failures.
This does not include the short-query repairs or all 85 historical gaps.
At the latest short-queue checkpoints, 18,948 of 26,906 queries were
strictly accepted; both original repair jobs remain active.

Long-cache smoke job 801103 terminated after accepting the complete
4,103-aa prediction and recording two OOM attempts for the 4,579-aa
query. Both full-length ESM caches are accepted. A separate row-chunked
relative-position operator preserves the stock integer formulas and
float32 output. Before any model forward, its gated smoke must prove
exact equality against the actual stock CPU operator at all 4,579
positions and all 139 channels. Passing local tests alone is not a
folding success. Original failures, good structures and cache provenance
remain intact. The standalone DSSP installation is still awaiting
approval; no package, Torch or original environment was changed.
