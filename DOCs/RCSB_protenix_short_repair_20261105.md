# RCSB Short-Chain Predicted-Structure Repair

This is a subsequent input-preparation stage for the whole-RCSB single-chain
union task. It does not change the fixed PDBTM system or the already frozen
ESM-C-only RCSB pilot. No label or prediction score selects these requests.

## Exact Scope

The original production folding scope was>=100aa. The full native ge50
universe therefore lacks26906previously unattempted50--99aa structures,
comprising2080123residues. The separate85historical longer failures remain
tracked as356229residues;this short repair does not silently count them as
completed or repeat their failed folding attempts.

All requests are projected from the previously validated missing manifest
SHA-256`fae3ff2654df51f661e8d690c59f23346f4077e0f91522f1cf73e9b10c266f4e`.
Sequence SHA-256,length and unique ID are checked for every row. The new
FASTA is parsed back and compared exactly. Two non-overlapping,alphabetic
shards contain13453queries each and cover the complete short request set.
There are347X-containing queries,including50all-X sequences. No unknown
residue is substituted. These unknown-rich inputs need explicit structural
QC,not a claim that their biological structures are uniquely identifiable.

Independent root:
`/data/user/shuang886/PPC_RCSB_single_union/20261105/protenix_short_repair`.
The`inputs/summary.json`records all manifests/hashes and the four boundary
smoke inputs:canonical50/99aa,partial-X,and all-X. Existing224276raw structure
features and154876local validated CIFs are not recomputed.

## Predictor and Runtime

Reuse the existing approved`fold`environment,Python3.12.11,Torch2.7.1+cu128,
Protenix1.1.0,Gemmi0.7.5,and already present weights. Predictor remains
`protenix_mini_esm_v0.5.0`,ESM2-3B,protein-only,seed101,one sample,BF16,
no MSA/templates,torch triangle kernels,no fusion. No package or Torch
installation is performed. This is Protenix,not ESMFold or OpenFold3.

The queue derives from the previous production script SHA-256
`0ac7ba922146322b14d69542bf1bd61f7abaaaa125a4c1d93070d3395ec8a85e`.
Its new minimum is50aa,with all manifest rows required:below-minimum or
invalid queries fail loudly rather than disappear through a>=100filter.
The contract pins manifest/model/seed/source/chunk size/retry budget.
The chunk size is targets per CLI invocation,not a neural-network batch.

## QC and Recovery

Accepted outputs require one model and one chain,the exact unchanged query
sequence,a C-alpha for each residue,finite atom coordinates/B_iso,and the
official summary-confidence JSON. The SDK can catch an ESM input error and
continue with zero features. Acceptance therefore also requires the exact
query/part mapping in the SDK CSV,a finite nonzero tensor of shape
`[L,2560]`,and no ESM error record. A broken tensor/QC runtime stops the
supervisor before it can mutate the queue or start inference. Per-query ESM
diagnostics count as bounded query failures,not an infrastructure abort.
Unsupported or failed X inputs remain
explicit failures rather than successful coverage. No confidence threshold
or PPIS score filters valid structures.

The previous progress ledger,atomic summary,signal forwarding,validated
reuse,and bounded per-query retries are retained. Paths are pinned before
changing child directories. The inference child inherits the queue lock so
a killed supervisor cannot allow a replacement while the old child remains
alive. Reused valid smoke outputs will not be folded again in the full queue.

Review:gpt-6-luna max,read-only narrow review of queue/test files. Two P2
issues (relative paths and orphan ownership) were fixed and regression-tested;
re-review found no remaining issue. A subsequent review of the ESM input guard
also passed after adding runtime and error-diagnostic regression tests.
Nine focused tests pass,including real CPU-only fake-CLI relative-path and
hard-kill/lock-inheritance tests. Full suite151passed/2optional snapshot
tests skipped,using the approved absolute AIAA Python;no library was changed.

## Actual GPU Smoke

Job800810 completed successfully in1m18s on an H10080GB using the unchanged
approved fold environment. All four boundary queries produced exact-sequence
CIFs and official confidence outputs,including partial-X and all-X queries.
The predictor forward passes took approximately0.35--0.93s each;these are
not end-to-end queue throughput estimates. ESM model loading/precomputation
and process startup contribute additional cost. All-X output establishes
runtime compatibility,not biological identifiability or model quality.

The original smoke source and outputs remain immutable. The strengthened
input-QC version will revalidate its saved ESM caches without refolding,and
accepted smoke outputs can be reused. The full repair uses two independent
acd_u jobs,13453queries each,process chunks of512 and at most two attempts
per query. The chunk parameter amortizes CLI startup;it does not change the
predictor's neural batch size. Each formal job has a12hour bound and a
recoverable per-query ledger. The85historical long failures remain separate.
