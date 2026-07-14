# Benchmark Baseline Progress (2026-07-14)

Canonical evaluation universe:

- split: `splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708/chain_filtered_training`
- test set: 2,392 case-sensitive chains / 1,036,256 residues
- exact-sequence unique set: 724 sequences / 352,746 residues
- task: residue-level single-chain effect-site prediction and chain effect-site ratio

This note records deployment and QC status. Partial smoke metrics below are not
formal benchmark rows and must not be merged into the main comparison table.

## DeepTMInter official

The full CPU pipeline is active on star in tmux session
`deeptminter_full_8x16_0714`, using eight workers with 16 threads per worker.
The feature stack is HHblits/UniRef30 2020-06, licensed Phobius 1.01,
FreeContact, GaussDCA, the five released DeepTMInter models, and the released
stacker compatibility path.

Run root:

`/data/husrcf/ProtBind/PPC/benchmark/runs/chain_filtered_global_si30_20260709/deeptminter_official_full`

At 2026-07-14 23:30 CST:

- 203/724 unique sequences (28.04%) had complete predictions.
- completed predictions reverse-map to 495/2,392 test chains.
- Phobius/alignment/coupling QC had reached 214/208/205 unique sequences.
- 11 distinct workers were active; no stage failure marker was present.
- the most recent one-hour window completed 28 unique sequences. If that rate
  remains representative, the remaining wall time is approximately 18.6 hours.

Formal collection remains blocked until all 724 unique predictions pass QC.
The strict collector is committed in star PPC as `5829502` and rejects
incomplete final coverage.

## MBPred adapted

Classification: **official random-forest checkpoint with adapted profile
source**. The released `AllFit.rf` model and historical MBPred feature formulas
are retained, but the MSA source is HHblits/UniRef30 2020-06 rather than the
paper's UniProt20 2015-06. This row must not be called an exact official feature
reproduction.

Reviewed production adapters:

- `benchmark/scripts/mbpred/build_mbpred_features.py`
- `benchmark/scripts/mbpred/materialize_mbpred_predictions.py`
- `benchmark/scripts/mbpred/validate_mbpred_full.py`
- star PPC commits: `216c719` (initial strict adapter) and `973f87a`
  (official omission handling and hashed zero-fill provenance)

The builder preserves both checkpoint-relevant historical behaviors:

1. the PSSM accumulator is reused across residue positions;
2. the official coevolution header/value order mismatch is retained.

It also verifies all `L*(L-1)/2` FreeContact pairs, exact sequence letters,
continuous Phobius segments, the 38-column checkpoint order, and finite values.
The released feature script legitimately omits SIGNAL residues and singleton
topology segments. Formal collection therefore permits only omissions declared
by the hashed per-sequence feature QC and assigns score zero only to those
positions; the official RF positions must exactly equal the complement.

Smoke validation:

- `dtm0600`: 1,145/1,145 feature rows, 654,940/654,940 coupling pairs,
  65,535 alignment rows, 21 topology segments, no omitted residues.
- feature construction: 12.88 s wall time, 26 MiB peak RSS.
- HPC2 Slurm RF smoke job `9978114`: 2.70 s, 182 MiB peak RSS, exit 0.
- 14-sequence RF batch job `9978133`: 37 s, exit 0.
- strict smoke collection: 14 unique / 14 chains / 15,897 residues, exact
  sequence/position/residue checks and 1/500 vote-grid checks passed.
- collected TSV SHA-256:
  `23eed4c5bab8483ffd322b38fd86b6827075a110189ce99c59df8e2760a16f35`.
- real SIGNAL-chain validation (`dtm0184`, length 1,025): feature rows cover
  original positions 27--1,025, while QC declares positions 1--26 as SIGNAL.
  HPC2 Slurm RF job `9979797` completed in 3 s. Strict reverse mapping expands
  the sequence to six chains / 6,150 residues and zero-fills exactly 26 unique
  residues / 156 chain-expanded residues. The independent validator passed on
  a frozen one-sequence source bundle; output SHA-256 is
  `a97ea7347aa6f0fda6009a90d70566f178d1c5259feebaea1d46f40434b28eb6`.

The 14-chain smoke AUPRC/AUROC were 0.1059/0.5561. These longest-first chains
are a nonrepresentative engineering smoke subset, so these metrics are excluded
from the formal table.

Current run root:

`/data/husrcf/ProtBind/PPC/benchmark/runs/chain_filtered_global_si30_20260709/mbpred_adapted`

As DeepTMInter emits each `.evfold`, MBPred features are generated without
waiting for GaussDCA or DeepTMInter model inference. The repaired two-worker
watcher resumed at 23:30 CST and had reached 182 feature/QC pairs at the latest
snapshot, including the formerly failing SIGNAL case. There are currently 22
official RF outputs. Final publication still requires 724 RF files and the
hard-locked 2,392-chain / 1,036,256-residue collector pass.

## EDLMPPI

Six official H5 checkpoints are present, but a clean exact-official row is not
currently defensible. Its 39-channel biological feature stack requires legacy
ANCHOR, GENN/ASAquick RSA, historical ECO/PSSM databases, and a label-informed
HSP channel.

The independent HSP audit confirmed:

- 202/724 test unique sequences have SI30/COV80 hits to the labeled HSP
  reference;
- those sequences represent 536/2,392 test chains;
- three exact sequence overlaps represent 34 test chains.

The reference labels affect transferred residue scores even though they do not
select the raw alignments. Therefore full-HSP EDLMPPI is excluded from the clean
primary table. A future defensible row would be labeled
`official-weights/adapted-features`, use all six checkpoints, set HSP to zero or
rebuild it after SI filtering, and declare every substituted feature source.

ProtT5 itself is not a blocker: the complete 11 GiB model exists on star at
`/home/husrcf/Code/gpsite_star/models/prot_t5_xl_uniref50`.

## Spatom

The user explicitly approved the isolated FreeSASA installation. A locally
built `freesasa==2.2.1` wheel was installed into
`/home/husrcf/anaconda3/envs/SPATOM` without changing other packages. Real-PDB
and five-length smoke tests passed before the formal run.

The two-shard formal run completed all 724 exact-sequence unique inputs with
zero errors in 21 min 8 s. Strict reverse mapping passed for 2,392/2,392 chains
and 1,036,256/1,036,256 residues. The formal row is:

| AUPRC | AUROC | AP | F1@0.5 | MCC@0.5 | best F1@gate | best MCC@gate | L/20 | L/10 | L/5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4393 | 0.7186 | 0.4380 | 0.1597 | 0.1041 | 0.5482@0.127 | 0.3250@0.126 | 0.4661 | 0.4616 | 0.4697 |

The row uses the released checkpoint, exact-sequence chain-only ESMFold,
DSSP, FreeSASA, and PSSMs converted from ScanNet's MMseqs2/UniRef50 A3Ms. It is
therefore classified as official-checkpoint/adapted-profile. The low 0.5 result
reflects score calibration shift; AUPRC/AUROC and test-oracle best-threshold
diagnostics are retained under the same policy as the other methods.

## Reporting policy

- DeepTMInter will be reported as official only after complete strict coverage.
- MBPred will be reported as adapted because its MSA database differs from the
  historical training pipeline.
- Spatom is now a complete official-checkpoint/adapted-profile row with full
  test-universe coverage.
- Partial smoke results remain outside the main benchmark table.
- Full-reference EDLMPPI HSP results are not eligible for the clean primary
  comparison; at most they may appear as an explicitly contaminated upper
  bound with a leakage-controlled ablation beside them.
- Structure-based methods continue to use exact-sequence, chain-only ESMFold
  structures; no UniProt/PDBTM/TmAlphaFold/AFDB bridge is permitted.

## Continuation checkpoint (2026-07-14 23:35 CST)

- Three lock-safe DeepTMInter workers were added without restarting the
  original eight. The active sessions are `deeptminter_full_8x16_0714` and
  `deeptminter_extra3_0714`; 11 distinct run IDs were confirmed active. The
  theoretical CPU ceiling is 176/192 threads, leaving 16 for I/O and system
  work. At launch, star still reported 342 GiB available memory.
- The reviewed two-worker MBPred sidecar is active as
  `mbpred_feature_watch_0714`. It polls every 60 seconds, requires the
  FreeContact completion marker and all input files, retries failed batches at
  most three consecutive times, and hard-stops at the 724-unique contract.
  Its current fix is star PPC commit `973f87a`.
- The repaired feature path reached 182 unique feature tables at this snapshot,
  and the formerly failing `dtm0184` SIGNAL case now passes. There are 22 RF
  outputs, including the independently submitted real-SIGNAL smoke.
- Spatom is complete and has been promoted to the main benchmark table.
  DeepTMInter and MBPred remain in-progress rows and still require all 724
  unique sequences plus the complete 2,392-chain collector pass.
