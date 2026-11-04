# Whole-RCSB Single-Chain Union Transfer Preparation

The PDBTM validation target and its fixed test evaluation are complete. This
next stage remains single-chain/all-partner-ever-observed union PPIS;it does
not resume the old pairwise task or introduce any ligand-task datasets.

New independent preparation root:
`/data/user/shuang886/PPC_RCSB_single_union/20261105`

Existing raw data and caches remain unchanged under
`/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1` and
`/data/user/shuang886/tminter3/artifacts/`. No shared raw mmCIF is moved.

## Scope and Required Labels

The existing native ge50 universe contains251267exact observed sequences,
not251267PDB entries and not all proteins in current RCSB. Its source manifest
contains contact-participating chain instances from a completed-entry snapshot.
The native split is201013train/25127val/25127test,using the existing detected
SI30/bidirectional-cov80 and same-PDB connected components. It is not an
exhaustive Biopython global-pairwise SI certificate.

The original tag generator was inspected:it restricts entity_poly to
polypeptides,removes H/D/unknown-element atoms and nonpositive occupancy,
selects one observed conformer,retains label_seq_id,author numbering and
insertion codes,and stores source sequences in observed-residue order.
Raw contract:`observed-polymer-heavy-atom-5p5-assembly-v2`.

Build the new union from original entry pairs and residue maps,not from the
old both-partner-length/feature-ready pair table. A short or feature-missing
partner must still contribute contacts to a qualified query. Exact-sequence
groups retain every observed index,including inserted residues;no UniProt
bridge or expected-polymer-sequence filling is introduced. This union build
and final loader normalization/coverage acceptance are not yet complete.

## Native RCSB vs PDBTM Holdouts

An exact-sequence-only lower-bound audit found:

| PDBTM partition | In RCSB train | In RCSB val | In RCSB test | Not in native exact universe |
|---|---:|---:|---:|---:|
| Train4868 | 4010 | 287 | 281 | 290,including8below50aa |
| Val1075 | 769 | 139 | 75 | 92 |
| Test724 | 455 | 90 | 92 | 87 |

Therefore a future model trained on native RCSB cannot reuse the current
PDBTM test as an independent post-training comparison. Protecting PDBTM val
and test by excluding already-touched native components would remove at least
156639of201013native train sequences,leaving no more than44374before a fuller
homology audit. This conservative component effect is not evidence that each
removed sequence is directly similar to a PDBTM query.

A user preference question is pending:retain the complete native RCSB split
and keep the current PDBTM result as pre-RCSB-only,or protect PDBTM holdouts and
accept a much smaller RCSB training pool. Feature preparation below is label-
free and independent of that choice;no split was changed and no RCSB model
training has started. No test performance is used to decide architecture.

## Feature Gap vs the Full Universe

The new filename-only inventory is against all251267native IDs,not the old
214097feature-ready pair endpoints. It found no unexpected IDs or duplicate
filenames;this does not replace a full tensor/sequence/content audit.

| Input | Present | Missing | Missing50--99aa | Missing>=100aa |
|---|---:|---:|---:|---:|
| ESM-C MLC | 214097 | 37170 | 26906 | 10264 |
| Protenix descriptors | 224276 | 26991 | 26906 | 85 |

The short-chain gap comes from the previous>=100folding/feature scope,not
failed remapping. The85longer structure gaps must remain explicitly tracked.
Existing caches are not discarded or recomputed. Direct whole-RCSB MSA
Transformer coverage is not established;do not fabricate a required MSA branch.
Structural descriptors/geometry require source-specific confidence checks,
the recorded corrections and normalization fitted only to final train IDs.

## Prepared Requests

`requests/esmc_missing.tsv` and matching FASTA contain37170queries /
5386023residues;`requests/protenix_missing.tsv` and FASTA contain26991queries /
2436352residues. All IDs,lengths and SHA-256 digests were checked against the
native split and original observed-sequence table,then every FASTA was parsed
back and compared exactly. Length range is50--4579aa. No label or fitted
normalization enters these requests.

The ESM-C missing manifest also passes the existing production extractor's
manifest validator. A7-query GPU input spans50,64,99,256,1025,4103,4579aa.
Reuse the installed `tminter3_esmc` environment and the byte-identical original
extractor;no Torch or package installation is needed. Model remains
ESM-C600M revision`e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c`,layers12/24/36,
final pretrained LayerNorm,BF16 inference,float32stored,no external FlashAttention.
Formal extraction must wait for the actual new smoke acceptance;request
generation alone is not feature completion or an actual RCSB training attempt.

Exact-overlap and filename-coverage previews,input hashes and request QC are
archived in the matching directory. The full union/feature/loader audit and an
actual training attempt remain required work.

## Actual ESM-C Gap Repair Launch

New GPU smoke800636 completed exit0 in1m31s. All7queries,including both
50aa and4579aa boundaries,were checked for exact sequence hashes,full[L,3456]
shape,finite/nonzero embeddings,correct layers/revision/normalization and
float32stored/BF16inference metadata. The installed environment's full freeze
hash exactly matches the previously approved environment;Torch/vision/audio
and SDK versions are unchanged. No packages were installed.

Formal missing-only extraction is array800645_[0-1] on acd_u,without a percent
cap,oneGPU/8CPUs/96GiB/12h per task. Each length-interleaved shard contains
18585queries,with2691987and2694036residues. It reuses the original model/cache
and byte-identical extractor,max_tokens4096/max_batch_size64. Output goes to
the existing SHA-256-sharded ESM-C bank;valid existing outputs are checked
and skipped,not regenerated. Per-shard locks and atomic writes preserve
restart safety. An OOM fails explicitly and is retained in logs;do not
silently count it as completed coverage or restart a live shard.

Both shards now completed successfully:800645_0 in9m21s and800645_1 in6m35s.
All37170missing outputs were produced with full-length finite tensor checks
inside the unchanged extractor. A post-run native filename audit found
251267present,zero missing,zero unexpected,zero duplicate and zero misplaced
files. Independent selected boundary rereads also passed at50and4579aa.
This is not yet an independent full-bank tensor reread;the new loader-input
packer will perform that audit before publishing formal training readiness.

This is feature extraction,not RCSB model training. The Protenix missing
list remains outstanding. The full raw union-label build has now completed
separately;see `RCSB_single_union_label_build_20261105.md`.
Launch,environment and smoke acceptance receipts are under the document's
matching directory. Neither PDBTM test performance nor a new split choice is
used by these frozen,label-free requests.

## Whole-Sequence Training Preparation

The new input packer is an explicit ESM-C-only preparation path,not a claim
that complete M18 geometry/MSA inputs already cover the whole RCSB universe.
It reuses the existing frozen ESM-C bank through links,checks every tensor
and its exact label alignment,and writes conventional compact metadata and
labels for the unchanged PPC loader. It does not modify embedding values,
invent missing MSA/structure features or fit normalization on held-out data.
An ESM-C-only whole-RCSB pretraining baseline can precede geometry/MSA
adaptation;it must be reported under that input condition.

The full raw union and input preparation can serve either split choice
discussed above. Actual model training,its final train/val IDs,and any
subsequent membrane adaptation still require an explicit frozen run contract.
The PDBTM target/test system will not be replaced using its test scores.

For a native-RCSB validation/test claim,the downstream PPC network must start
from scratch,not warm-start a PDBTM-supervised checkpoint:the exact-overlap
preview already places287PDBTM train sequences in native RCSB val and281in
native RCSB test. Frozen unsupervised PLM weights are a separate input
condition. A PDBTM-initialized RCSB run could be exploratory only unless its
evaluation partitions are rebuilt to exclude all corresponding supervised
training overlap and homologous components.

## Input Packer Acceptance Contract

`pack_rcsb_sequence_inputs.py` independently reads every original ESM-C
payload,checks sequence/hash/length/layers/revision/normalization,finite
float32[L,3456] tensors,and the exact CSR label hash. It records each feature
content SHA-256 plus device/inode/size/mtime/ctime,and checks this inventory
and the resolved links again before publishing readiness. Every mandatory
union artifact fingerprint must exist and match before/after processing.
All original PDB contexts and native components must stay split-disjoint.

This avoids copying the approximately1.03TB float32 embedding values. The
feature bank must remain immutable throughout the reference lifetime,including
training;training preflight must recheck the inventory. This is not a
hostile-root-proof filesystem snapshot. The known ESM-C extraction jobs are
terminal before packing. No other producer may rewrite this bank during use.

Packer review:gpt-6-luna max PASS after two fixes and regressions for a
post-audit source rewrite and missing mandatory context fingerprints.
6focused tests pass,including strict existing PPC loader/collation and spawn
execution. Full local suite136passed/2optional snapshot tests skipped.
The existing training/model code is unchanged.

A planned3epoch,seed49,whole-native-train/validation ESM-C-only pilot is in
`configs/experiments/20261105_rcsb_single_union/`. It uses the existing
256D gated-depthwise TCN/MLP backbone,plain chain-mean CE,batch64/token65536,
8workers/prefetch4/pinning,scratch initialization,and validation-only
checkpoint selection. It is not a full-convergence result or an isolated
data-size ablation. The planned job may launch only after complete packer
acceptance,configuration review and a real GPU smoke. No test is enabled.
