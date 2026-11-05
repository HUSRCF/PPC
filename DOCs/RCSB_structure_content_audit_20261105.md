# RCSB Existing-Structure Content Audit

## Scope

The native single-chain universe has251267queries. This audit covers the
224276queries with previously generated Protenix Mini ESM structures and
raw v4 feature records. It does not claim full native structure coverage:
26906short folding requests and85historical failures remain separate.
It does not modify the frozen PDBTM target/test system or the independently
submitted ESM-C-only native-RCSB pilot800877.

All eight Star-to-HPC3 transfer streams completed with exit0. Their final
transport receipt is archived in
`RCSB_structure_readiness_20261105/parallel_transfer/completed.json`.
Transport completion is not content acceptance.

## Accepted Content

For each query,the validator requires:

1. One CIF model/chain,exact query length and sequence SHA-256,and a C-alpha
   at every position. Atom coordinates/confidence must be finite.
2. C-alpha B_iso and official global pLDDT in0--100range. The cached output
   is C-alpha pLDDT divided by100,in canonical query order,not PPIS probability.
3. Correct raw-feature UID,sequence/hash,Protenix model identity and raw
   normalization declaration. Required tensors must have their declared
   shapes,float32 dtype,no gradient,and finite values.
4. The complete ordered89-name v4 scalar schema,not merely a few anchor
   columns. Raw and unnormalized scalar fields must match;an all-zero SASA
   fallback cannot be accepted.
5. C-alpha and all heavy-atom identities/coordinates matching the cached
   raw features within0.0011Angstrom. This allows the original temporary
   PDB's rounding;it does not align or substitute another conformation.
   Duplicate CIF heavy-atom names/alternate conformers are rejected.
6. Stable content fingerprints while each source is read,and a final
   source-identity inventory pass before publication. Catalog and audit
   code hashes must remain unchanged. Source banks must remain immutable
   through their later use;this is not a hostile-root-proof snapshot.

Source code is frozen at commit`81c5953`. The validator's complete schema
was independently compared with the canonical v4 name generator and matched
all89positions. Two P2 review findings (permitted non-anchor permutations
and silently overwritten duplicate atoms)were fixed with regressions.
gpt-6-luna max re-review passed. Eleven focused tests and168full tests
passed;twooptional snapshot tests were skipped. A real spawned-worker CLI
test covers the multiprocessing path.

## Outputs And Claim Boundaries

`audit.jsonl`stores UID,length,native split,CSR offset,status,source content
hashes/filesystem identities,and pose/confidence diagnostics. The companion
`absolute_CA_plddt_div100.npy`is float32. Pending/failed rows stay NaN,
not fabricated zeros. Only successful full existing-scope acceptance can
publish`existing_content_ready.json`;an eight-query smoke cannot publish it.

No labels are read,no normalization is fitted,and no geometry-training
inputs or model checkpoint are published. The validator does not recompute
all descriptor mathematics or retrospectively audit old ESM2 input caches.
The two known raw descriptor corrections and train-only normalization remain
subsequent preparation,not implied accomplishments of this source audit.

## Actual Smoke And Runtime Repair

The first real smoke checked four local and four newly transferred CIFs:
8queries/3219residues,all accepted. Its post-read source inventory also passed.

The first full CPU launch terminated with`BrokenProcessPool`. Its log shows
OpenBLAS attempting64threads per process and failing to create threads.
The login cgroup has a500-task cap;the log additionally reports an
RLIMIT_NPROC soft limit2048. Eight workers plus the parent,each initializing
64BLAS threads,can exceed the cgroup cap. This evidence supports thread
oversubscription,not a claim that NumPy was improperly installed. The
observed8GiB memory limit/fail counter alone cannot establish an OOM cause.
The failed output and logs remain preserved,with no ready marker.

No package or source code was changed. Before starting Python,we set
`OMP_NUM_THREADS=1`,`OPENBLAS_NUM_THREADS=1`,`MKL_NUM_THREADS=1`,and
`NUMEXPR_NUM_THREADS=1`. A real8-worker smoke then accepted all8queries.
Its confidence array was byte-identical to the original smoke,and runtime
threadpool inspection confirmed one BLAS/OpenMP thread.

The full retry started at2026-11-05T01:17:13UTC,PID39689,eight CPU workers,
nice10,no GPU. At01:24:54UTC,the exact process handle remained live and
38000of224276queries had passed,with no pending or failed records reported.
The final source-inventory pass and publication were still incomplete.

Runtime root:
`/data/user/shuang886/PPC_RCSB_single_union/20261105/structure_source_catalog`.
Current outputs/log are`content_full_v2/`and`content_full_retry.log`.
Original failed outputs are`content_full_v1/`. Launch,smoke and failure
diagnosis receipts are archived in this document's matching directory.

## Remaining Structure Requests

The85historical failures have lengths232--4579. Exactly82exceed4094;
the remaining three are232,305and402aa. The installed official ESM process
path explicitly passes a4094truncation length. This offers a concrete
repair hypothesis for the82long inputs,but does not by itself prove the
failure cause for every case. Full-length precomputed ESM caches and
bounded long-chain folding require separate validation;no such rescue has
been launched here,and no query is shortened or silently omitted.

## Full Acceptance And Subsequent Repairs

The existing-scope run is now complete: **224,276 queries / 72,210,441
residues**, all accepted, zero failed/pending. The final source-identity
inventory passed. The producer took 3,821.65 seconds. Its formal receipt
SHA-256 is `3f47c6d1fccf16aa97daea730fca45dd4205088620f2a55d3640b352a66a552a`.
The complete receipt is archived as `existing_content_ready.json` in the
matching directory.

An independent full readback checked every confidence slice against its
per-query hash, finite 0--1 range, sequential CSR offsets and unique query
identity. All 224,276 slices and both full output-file hashes matched.
See `qbank_readback_acceptance.json`. This is existing-scope acceptance,
not full native coverage, descriptor re-extraction, normalization, or
geometry-training readiness.

The postprocessing optimization in commit `22ee880` completed a real
512-query process cycle in 314.96 seconds, versus 1,975.48 seconds for the
old first cycle. This is an observed operating comparison, not a matched
query timing trial. Strict QC and model parameters stayed unchanged.
See `first_cycle_performance.json`.

### Historical85: Two Different Input Problems

The full 85-row manifest was checked without changing any sequence:

- 82 canonical/X queries are 4,103--4,579 aa, above the stock 4,094 cap.
- Three queries have lengths 232, 305 and 402 and contain B/Z ambiguity
  symbols. Their hashes and lengths are valid. The actual stock prediction
  parser's `PROTEIN_1to3` mapping has no B or Z entries, although the generic
  mmCIF conversion table does. No B/Z-to-X replacement was made.
- All 85 belong to native train. Whole-data ESM-C-only training continues
  to retain all of them; the cohort has not been narrowed.

Job `801071` stopped in two seconds at strict alphabet QC before model
initialization. It is not a GPU failure or OOM. Separate 82-query and
three-query request tables preserve the complete original universe under
`protenix_long_repair/inputs/`.

Commit `432c045` adds a full-length ESM-cache adapter without editing the
installed Protenix or Torch packages. Source hashes, local ESM weights,
query identities, complete residue tensors and per-query provenance are
pinned. Unaccepted predictions stay in a temporary output directory until
input/cache/source checks pass; failed checks cannot be reused as a
successful parent-queue result. It passed gpt-6-luna max review, eight
focused tests, and 181 suite tests with two optional skips.

Job `801103` tests the two canonical length extremes, 4,103 and 4,579 aa.
Both full ESM tensors have been accepted on GPU, proving the cache is no
longer truncated. The 4,579-aa Mini prediction nevertheless ran out of
memory in `RelativePositionEncoding.generate_relp`, where full int64
one-hot tensors are concatenated and converted to float32. The failed
allocation was 10.86 GiB with 69.61 GiB already allocated; only 84.85 MiB
was reserved but unused. This supports a peak-allocation limitation, not
a fragmentation diagnosis. Bounded folding retries are retained. No
all-82 repair success or complete long-folding success is claimed.

### Secondary-Structure Backend

Twelve existing raw-feature probes contain B/G/I/T/S assignments outside
the H/E/coil states available in the present pyDSSP fallback. They cannot
be silently treated as the same backend. This sample evidence, not a
full-bank backend census, is in `secondary_structure_backend_probe.json`.

The official Conda channel was unreachable. A mirror dry-run against the
frozen ML environment was unsatisfiable because of existing C++ runtime
pins. A tool-only dry-run for DSSP 4.4.11 succeeds with six linked packages
and no Python, Torch, CUDA packages or removals. The new prefix was not
created. Actual installation was requested for user approval, not performed.
See `dssp_tools_dry_run.json`. Compatibility/output checks remain required
before extracting new short-chain features.
