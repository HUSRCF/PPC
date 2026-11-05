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
| Train batches per full epoch | 3,170 |
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
