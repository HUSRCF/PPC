# RCSB Single-Chain Union Asset Preflight

Read-only HPC3 inventory,2026-11-05 local time. No new RCSB training has
started in this stage. The user requires the PDBTM validation milestone and
locked test evaluation first. These are existing artifact counts, not a fresh
full tensor audit or proof that the entire RCSB is ready.

## Existing Assets

HPC3 dataset root:
`/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1`

| Artifact | Reported scope |
|---|---|
| `sequences/observed_full_success_v1/summary.json` | 242,944 completed entries;1,072,528 interacting chain instances;271,397 exact observed sequences;75,064,917 unique-sequence residues. Explicitly a completed-entry snapshot, not a final completeness claim. |
| `training/ppis_ge50_v1/split/summary.json` | 251,267 length>=50 sequences;201,013/25,127/25,127 train/val/test;MMseqs-detected SI30/bidirectional cov80 plus same-PDB components. No detected cross-split SI edges, exact sequences, PDBs or assemblies;not an exhaustive global-SI guarantee. |
| `training/ppis_ge50_v1/pairs/summary.json` | Old pairwise task:1,903,031 raw contacting contexts;both-chain>=50 filter excludes195,166 contexts. Retained train/val/test pairs1,420,430/136,802/150,633. |
| `training/ppis_ge50_v1/ready_v1/summary.json` | Feature-ready old pairwise endpoints170,465/21,662/21,970, total214,097. This is not the new single-chain union universe. |
| `_esmc/completion_coverage_audit.json` | 214,097 expected and present filenames;zero missing/unexpected/misplaced;69,260,770 residues. Scope is filenames plus extractor summaries, not independent rereading of every tensor. |

ESM-C root is `TMINTER3-ESMC/features/unique/`. A real cached sample was
opened and contains `[108,3456]` embeddings from ESM-C600M, layers12/24/36
concatenated, final-layernorm, float32 stored tensors, revision
`e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c`. This matches the current PPC
main embedding interface at the inspected sample;full coverage/schema/hash
verification still needs to be performed against the new union manifest.

Predicted-structure descriptors are under `protenix/features/complete_raw_v1`.
They are **Protenix**, not the current PDBTM OpenFold3 inputs. Existing
pairwise-ready receipts validate Protenix contents, and normalization used
170,465 training endpoints/55,003,867 residues. Those old normalization values
must not be silently applied after the union training sample/split changes.

Raw label/sequence mapping sources remain under:
`/data/user/shuang886/tminter3/artifacts/rcsb_tags_v1` and
`/data/user/shuang886/tminter3/artifacts/rcsb_sequences_full_success_v1`.
Shared raw mmCIF files must not be changed.

## Required Conversion

1. Build single-query **ever-observed, all-partner union** labels from the
   original observed-heavy-atom5.5A biological-assembly labels and residue
   mappings, including insertion codes. Preserve provenance/counts per UID.
2. Do not union only the feature-ready or both-partner-length-filtered pair
   manifest: that would omit contacts to short or feature-missing partners.
   A partner need not have a folding/embedding cache to contribute a query's
   experimental union label.
3. Reconcile sequence SHA-1/SHA-256 ID conventions explicitly. Audit sequence,
   length and residue coordinates before reusing any ESM-C or structure file.
4. Audit homology and exact overlap with the frozen PDBTM evaluation cohort
   before choosing RCSB pretraining/fine-tuning splits. The old RCSB split's
   internal isolation does not prove isolation from PDBTM.
5. Rebuild corrected scalar descriptors/confidence semantics and geometry
   inputs for the actual Protenix source, or prepare OF3 inputs separately;
   refit normalization on the final training partition only.
6. Inventory direct MSA Transformer coverage. None was established in this
   preflight. If absent, prepare it or explicitly use an ESM-C/geometry-only
   pretraining stage followed by MSA adaptation;do not fill a required branch
   with zeros or silently substitute a different representation.
7. Report all eligibility losses separately:raw snapshot,length filtering,
   union construction,SI/PDBTM isolation and each feature's missing counts.

This preflight establishes reusable locations and identifies conversion work.
It does not authorize test-guided configuration changes or claim that the
pairwise-ready214,097 endpoints are already a whole-RCSB union training set.
