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

At 2026-07-14 16:40 CST:

- 13/724 unique sequences had complete predictions.
- 18/724 had completed alignment QC.
- 17 `.evfold` and 14 `.gdca` files were available.
- no stage failure marker was present.

The queue is longest-sequence-first, so early wall times are not representative
of the full length distribution. Formal collection is blocked until all 724
unique predictions pass QC. The strict collector is committed in star PPC as
`5829502` and rejects incomplete final coverage.

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
- star PPC commit: `216c719`

The builder preserves both checkpoint-relevant historical behaviors:

1. the PSSM accumulator is reused across residue positions;
2. the official coevolution header/value order mismatch is retained.

It also verifies all `L*(L-1)/2` FreeContact pairs, exact sequence letters,
continuous Phobius segments, the 38-column checkpoint order, and finite values.
Formal runs use `--require-full-residue-coverage`.

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

The 14-chain smoke AUPRC/AUROC were 0.1059/0.5561. These longest-first chains
are a nonrepresentative engineering smoke subset, so these metrics are excluded
from the formal table.

Current run root:

`/data/husrcf/ProtBind/PPC/benchmark/runs/chain_filtered_global_si30_20260709/mbpred_adapted`

As DeepTMInter emits each `.evfold`, MBPred features can be generated without
waiting for GaussDCA or DeepTMInter model inference. Final publication requires
724 RF files and the hard-locked 2,392-chain / 1,036,256-residue collector pass.

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

The staged Spatom pipeline has exact ESMFold structures, profiles, DSSP,
checkpoint, and adapter coverage for all 724 unique sequences. Its isolated
environment is `/home/husrcf/anaconda3/envs/SPATOM`. The only remaining runtime
dependency is FreeSASA. No installation has been performed because the project
rule requires explicit approval before adding a package to an isolated cloned
environment.

## Reporting policy

- DeepTMInter will be reported as official only after complete strict coverage.
- MBPred will be reported as adapted because its MSA database differs from the
  historical training pipeline.
- Partial smoke results remain outside the main benchmark table.
- Full-reference EDLMPPI HSP results are not eligible for the clean primary
  comparison; at most they may appear as an explicitly contaminated upper
  bound with a leakage-controlled ablation beside them.
- Structure-based methods continue to use exact-sequence, chain-only ESMFold
  structures; no UniProt/PDBTM/TmAlphaFold/AFDB bridge is permitted.

## Continuation checkpoint (2026-07-14 16:56 CST)

- Three lock-safe DeepTMInter workers were added without restarting the
  original eight. The active sessions are `deeptminter_full_8x16_0714` and
  `deeptminter_extra3_0714`; 11 distinct run IDs were confirmed active. The
  theoretical CPU ceiling is 176/192 threads, leaving 16 for I/O and system
  work. At launch, star still reported 342 GiB available memory.
- The reviewed two-worker MBPred sidecar is active as
  `mbpred_feature_watch_0714`. It polls every five minutes, requires the
  FreeContact completion marker and all input files, retries failed batches at
  most three consecutive times, and hard-stops at the 724-unique contract.
  Its star PPC commit is `a64d173`.
- The incremental path reached 21 unique feature tables and 21 RF outputs.
  Strict reverse mapping produced 25 case-sensitive chains / 27,440 residues;
  all source hashes, residue coordinates, and 1/500 score-grid checks passed.
  The partial TSV SHA-256 is
  `1dc7dede04bb80a3158e7a625ad73c2d5ba349bb7ca3f6adca89adf43f1b3563`.
- This remains a smoke checkpoint. Formal DeepTMInter and MBPred rows still
  require all 724 unique sequences and the complete 2,392-chain collector.
