# D/E x Direct MSA: Paired-Seed Pilot

## Scope

User-approved continuation of the exact-sequence union/OpenFold3 single-chain
experiments. This is not TMINTER3 pairwise training or a new test-set evaluation.
Primary reporting is validation chain-macro AP and pooled AP (average precision).
Ratio MAE, bias, F1/MCC and Brier remain diagnostics, not a reason to accept an AP loss.

| Loss | No direct MSA | MSA profile | Frozen MSA encoder |
|---|---|---|---|
| D: unweighted unsmoothed chain-mean CE + 0.75 rate MSE | D_none | D_profile | D_encoder |
| E: same CE + 1.0 Bernoulli-rate KL | E_none | E_profile | E_encoder |

Each cell uses seeds 49/50/51, 50 epochs, no checkpoint initialization/resume,
no independent ratio head, and the same frozen train4868/val1075/test724 assignment.
Seed49 is the priority observation lane on acd_ue; seeds50/51 use acd_u.
Two GPU slots per lane, with dependencies between stages, cap this matrix at
four simultaneous GPUs. Every job has a 12-hour limit.

HPC3 ACD rejects CPU-only batch jobs. Profile preparation therefore combines16
CPU shards/eight workers into one8CPU/64GB/GPU allocation after the priority
no-MSA runs finish; the short CPU audit also follows the same lane. This does
not add GPUs on top of the four-slot cap. The dedicated encoder feature smoke
has a30-minute limit and must succeed before full encoder extraction.

## Inputs and Model

The first completed seed49 no-MSA controls are in
[their frozen summary](msa_factorial_20261104_none_seed49.json). Both completed50
epochs: D macroAP0.728226/pooledAP0.760506; E0.728589/0.754809, evaluated at
each run's macroAP-best epoch. This is a single-seed tradeoff, not an E win or
statistical superiority claim. Other seeds and direct-MSA arms are still pending.

ESM-C MLC12/24/36, OF3 90D descriptors and frozen ESM2 contacts are unchanged.
Reuse accepted unpaired UniRef90 2026_03 A3Ms for all6667 exact sequences.
No templates, partner sequence/pose, interface labels or CATH/BioLiP annotations
enter MSA feature extraction.

Profile: exact aligned-row deduplication, Henikoff position weights,20 weighted
amino-acid frequencies, weighted gap fraction, normalized entropy, query-residue
frequency, log1p column Kish effective depth divided by log1p16384, availability.
This is25D, not24D, because query-residue frequency is an extra channel.
Kish effective depth is not HHsuite identity-cluster Neff. Query-only or columns
with fewer than two canonical observed residues receive zero features/availability.
All statistics have fixed bounded definitions; no val/test normalization is fitted.

Encoder: official frozen ESM-MSA-1b12-layer100M, final query-row768D representation.
Keep query and deterministically stratify unique rows by query identity up to64.
Long sequences use1000-position windows with128 overlap and average overlapping
query features, preserving every residue. Ambiguous U/Z/O/B/J tokens become X only
inside the encoder, without changing stored query identity or residue coordinates.
OOM fallback decreases selected depth, records the actual depth, and never silently
drops a chain. This windowed representation is not full-length global attention
for proteins longer than1000 residues.

Both feature types reuse the generic frozen-embedding loader historically named
prottrans; these are not ProtT5 features. A25/768->64 projection and zero-initialized
64D conditional residual adapter act before the existing contact block.
The new allow_random_init_residual switch explicitly permits training both backbone
and residual from scratch; the old checkpoint-required default remains unchanged.

The MSA arms add parameters and consume constructor RNG. They are augmented-system
comparisons, not matched-parameter or identical-backbone-initialization controls
against none. D/E within a feature/seed have identical architecture and initialization.

## Gates

1. Unit tests, narrow external code review and existing-environment CPU regression.
2. Exact query and per-row A3M match-column length validation; frozen A3M SHA256.
3. For all6667 feature files: sequence/hash, shape, finite tensor, MSA hash,
   extractor hash, feature mode, explicit encoder weight hash and requested depth.
   Resume must validate the same fields, not merely accept an existing filename.
4. CPU audit receipt must match feature dimension/mode and extractor version.
5. Each MSA training job first runs a separate real-payload scratch smoke:
   two optimizer updates, finite train/val loss, updated backbone/residual weights,
   n_test=0. Formal training is a new process with the original seed and output.
6. Retain every formal epoch/checkpoint/validation prediction and metric-specific
   best. No automatic promotion or test-oracle threshold search.

## Runtime Locations

- HPC3 root: /data/user/shuang886/PPC_M18/sequence_union_v2_20261030/msa_factorial_20261104
- Existing Python: /data/user/shuang886/.conda/envs/openfold/bin/python
- Code snapshot: root/code
- Configs/job mapping: root/configs and root/jobs.json
- Features/audit receipts: root/features/profile and root/features/encoder
- Runs: root/runs/{D,E}_{none,profile,encoder}_seed{49,50,51}
- Logs and failed-run archives remain under this root.

No Torch/package install is performed. Official encoder weight transfer is verified
by byte size and local/HPC3 SHA256 agreement before GPU extraction starts.

## Reviewer Suggestions Kept Separate

Code inspection confirms that current protein_features_v4 feeds v2_surface[:,0]
(curvature) into a parameter named sasa_values, and surface roughness uses a standard
deviation over flattened absolute XYZ coordinates. Neither issue's AP impact has
been measured here. Keep the existing feature bundle frozen for this matrix;
corrected descriptors require a separate version, train-only normalization and
paired retraining control. Do not silently modify old assets or historical results.

Pure CE, chain/residue loss reduction, geometry graph updates, multi-seed ensembles
and D-I-TASSER are subsequent candidates, not unapproved additions to this matrix.
