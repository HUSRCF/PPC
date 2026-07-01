# PPC Sequence-only DeepTMInter-Aligned PPIS TODO

Last updated: 2026-06-24 23:43 Asia/Taipei

## Scope

Current mainline is strict sequence-only, DeepTMInter-aligned chain-level PPIS. Predicted-structure and PDBTM-derived
structure branches are excluded from the primary model path because the PDBTM to
UniProt to predicted-structure alignment is not reliable enough for a defensible
main result. Structure-dependent branches remain future engineering work or
supplementary baselines only.

Fixed protocol:

- Task: one chain -> per-residue inter-chain interaction-site probability.
- Label unit: chain-level union over all other chains in the same biological assembly, not residue-pair contact map and not selected-partner chain-pair prediction.
- Input: one protein sequence plus sequence-derived features only.
- Label: 5.5A heavy-atom inter-chain contact tag in `features/contact_labels_5p5A/`.
- Split: strict SI30 cluster-disjoint split.
- Checkpoint/threshold: selected by validation only.
- Test: evaluated every epoch for logging if desired, but never used for model
  selection.
- Metrics: F1, MCC, PR-AUC, AUROC, P@L/20, P@L/10.
- Excluded from mainline: predicted-structure contact, predicted-structure
  DSSP/RSA/SASA, GVP/EGNN coordinate branches, bound-chain PDB features.

## TODO 1: Backbone Matrix

Goal: replace the residue-wise MLP bottleneck with sequence-context encoders and
establish a clean sequence-only backbone before adding extra features.

Priority configs:

| ID | Name | Input | Module | Purpose |
| --- | --- | --- | --- | --- |
| B0 | esm_mlc_mlp_baseline | ESM multilayer | MLP | Existing clean anchor. |
| B1 | esm_mlc_tcn | ESM multilayer | TCN | Local motif/window context. |
| B2 | esm_mlc_dilated_tcn | ESM multilayer | Dilated TCN | Larger local receptive field. |
| B3 | esm_mlc_global_attn | ESM multilayer | Lightweight global attention | Long-range sequence dependency. |
| B4 | esm_mlc_tcn_global | ESM multilayer | TCN + global attention | Candidate main backbone. |

Required ablations:

- B0 vs B1: test whether local context beats residue-wise MLP.
- B1 vs B2: test whether expanded receptive field matters.
- B1/B2 vs B3: separate local and global context effects.
- B4 vs best single-context backbone: test local/global complementarity.

Decision rule:

- If B4 improves MCC or PR-AUC without hurting F1, use B4 as the default
  backbone for TODO 2.
- If only TCN improves, keep the simpler TCN backbone.
- If none improve over B0, stop feature stacking and inspect implementation,
  split difficulty, and label noise first.

## TODO 2: Sequence-derived Feature Branches

Goal: test whether traditional sequence-derived features still add signal beyond
PLM embeddings. These features are allowed because they are generated from
sequence/MSA/predictors rather than experimental or predicted structure mapping.

Priority configs:

| ID | Name | Extra input | Module | Purpose |
| --- | --- | --- | --- | --- |
| F0 | best_backbone_only | none | best TODO 1 backbone | Control. |
| F1 | backbone_netsurfp | NetSurfP RSA/ASA/SS/disorder | small feature encoder | PIPENN/GPSite-style surface/SS prior. |
| F2 | backbone_pssm | PSSM | small feature encoder | PIPENN/DELPHI/CoGANPPIS conservation prior. |
| F3 | backbone_hmm | HMM | small feature encoder | Remote homology prior; lower priority. |
| F4 | backbone_pssm_netsurfp | PSSM + NetSurfP | separate encoders | Main traditional feature combination. |
| F5 | backbone_pssm_hmm_netsurfp | PSSM + HMM + NetSurfP | separate encoders | Full sequence-derived branch. |

Feature readiness:

- PSSM: MMseqs/PSI-BLAST style profile, train/val/test generated with fixed
  database and documented parameters.
- NetSurfP: standalone NetSurfP3 or compatible adapter. Keep length cap fixed
  at 2000 for fairness if used.
- HMM: HHblits/HH-suite branch only after PSSM and NetSurfP are stable.

Required ablations:

- F1 vs F0: surface/secondary-structure contribution.
- F2 vs F0: conservation contribution.
- F4 vs F1/F2: complementarity between conservation and surface/SS.
- F5 vs F4: whether HMM justifies extra cost.

Decision rule:

- Keep a feature branch only if it improves at least one ranking metric
  (PR-AUC or P@L/10) and does not degrade MCC/F1 materially.
- If feature gains are unstable across seeds, report as auxiliary signal rather
  than main contribution.

## TODO 3: Fusion, PLM Contact Prior, and Finalists

Goal: improve how branches interact while keeping the input sequence-only.
ESM-derived contact/attention is allowed; predicted-structure contact is not
allowed in the mainline.

Priority configs:

| ID | Name | Components | Fusion/prior | Purpose |
| --- | --- | --- | --- | --- |
| U0 | concat_fusion | best backbone + best features | concat | Simple fusion baseline. |
| U1 | gated_residual_fusion | best backbone + best features | gated residual | Stable low-parameter fusion. |
| U2 | weighted_residue_fusion | ESM/local/global/feature branches | per-residue weights | Interpret branch usage. |
| C1 | esm_contact_sparse | best backbone | ESM contact graph | Clean PLM contact prior. |
| C2 | esm_contact_bias | best backbone | attention logit bias | Contact prior inside attention. |
| C3 | best_fusion_contact | best fusion + best PLM contact prior | combined | Final candidate. |

Required ablations:

- U0 vs U1: whether gating avoids weak-feature pollution.
- U1 vs U2: whether dynamic branch weighting is useful.
- C1 vs best backbone: whether ESM contact graph adds signal.
- C2 vs C1: whether attention-bias injection is better than post-hoc graph use.
- C3 vs best non-contact fusion: whether clean PLM contact prior is worth keeping.

Finalist protocol:

- Pick at most three finalists:
  - best backbone-only model;
  - best backbone + sequence-derived feature model;
  - best full fusion/contact-prior model.
- Run 3 seeds for each finalist.
- Report mean +/- std on test using validation-selected checkpoint and threshold.
- Main table columns: F1, MCC, PR-AUC, AUROC, P@L/20, P@L/10.

Do not prioritize now:

- Predicted-structure contact or scalar features: strict alignment coverage is
  too low for a main claim.
- GVP/EGNN: coordinate-dependent and therefore out of current mainline.
- DELPHI full feature stack: useful for benchmark, too heavy for first mainline
  iteration.
- DCA/co-evolution branch: promising but second-stage due to MSA cost.
- GPSite teacher multitask distillation: possible later, but adds claim
  complexity before the sequence-only mainline is stable.

## TODO 4: External Benchmark Queue

Goal: run defensible DeepTMInter-aligned external baselines on the PPC strict
test set without mixing incompatible protocols.

Benchmark scope:

- Primary external benchmarks must predict chain-level residue interaction-site
  probabilities, aligned with DeepTMInter.
- Contact-map / selected-partner chain-pair methods are out of the primary
  benchmark table. They can be used later only as a separate auxiliary analysis.
- Paper-reported values and our strict-test values must be clearly separated.

Priority queue:

| ID | Benchmark | Status | Action |
| --- | --- | --- | --- |
| X0 | GraphPPIS fast | Done | Keep completed strict-test result as deployable structure-assisted baseline. |
| X1 | GPSite protein-binding channel | Active strict inference | After a session breakdown, resumed with `benchmark/scripts/run_gpsite_resume.py`: chunk000 is running in tmux `ppc_gpsite_resume_chunk000` on GPU0 and chunk001 in `ppc_gpsite_resume_chunk001` on GPU1. Official MAX_SEQ_LEN=2000 remains unchanged. |
| X2 | DELPHI CPU | Active extraction | BLAST nr `nr.000`-`nr.162` download and md5 files are complete; extraction resumed in tmux `ppc_delphi_nr_resume_extract` and currently has `.extracted` markers through `nr.029`. |
| X3 | PIPENN / ensnet_p | Blocked for strict-test | Official dnet checkpoint now runs on official prepared BioDL-P files through CPU TensorFlow; train/val/test MMseqs PSSM are complete, but strict-test still needs NetSurfP/RSA/Q3-compatible official feature payload before real inference. |
| X4 | DeepTMInter / MBPred | Reference | Use as protocol anchor and paper-number reference unless runnable code is reproduced. |
| X5 | Gated-GPS | Optional/deferred | Repo exists, but official `ckpt/` and `datasets/feature/` are absent; `data.py` also expects `psepos/blosum/dssp/resAF/dismap`, so this is not an immediate ESM-only run. |
| X6 | CoGANPPIS | Optional/reference | Keep as sequence PPIS reference; defer deployment due MSA/co-evolution cost. |
| X7 | EquiPPIS | Supplement, smoke done | Isolated `EquiPPIS_cpu` environment runs the official 60-chain sample with the released checkpoint; next decision is whether to generate strict-test EquiPPIS features. |
| X8 | AGAT/GTE/EGCPPIS | Supplement only | Structure-heavy single-chain PPIS baselines; defer unless broad supplementary table is needed. |
| X9 | BIPSPI/PPLM/DeepInteract/GLINTER/CDPred | Out of primary scope | Partner-specific/contact-map models; do not spend mainline compute now. |

Immediate checklist:

- [x] Archive benchmark retention plan in `benchmark/plans/benchmark_retention_deeptminter_aligned.md`.
- [x] Archive corrected task definition in `DOCs/deeptminter_aligned_task_definition.md`.
- [x] Complete GPSite ESMFold weight download.
- [x] Prepare GPSite chain-level strict FASTA/metadata/labels under `benchmark/gpsite_ours_strict/chain_level`.
- [x] Verify GPSite model forward on bundled demo features in BIO/ROCm.
- [x] Complete GPSite ESM2 trunk and contact-regression weight downloads.
- [x] Start GPSite post-weight watcher `ppc_gpsite_after_weights` to run official demo smoke and strict-test chunks once both weights are complete.
- [x] Run GPSite official demo smoke in BIO with `PYTHONNOUSERSITE=1`, `TORCH_HOME=benchmark/models/torchhub`, and `GPSITE_PROTTRANS_PATH=benchmark/models/prottrans/ProtT5-XL-UniRef50`.
- [ ] Run GPSite strict-test inference with MAX_SEQ_LEN=2000 unchanged and chain-level evaluation. Current resumed runs: tmux `ppc_gpsite_resume_chunk000` for chunk000 on GPU0 and tmux `ppc_gpsite_resume_chunk001` for chunk001 on GPU1.
- [x] Finish DELPHI nr download through `nr.162.tar.gz` plus md5 files.
- [x] Start DELPHI post-download/resume watcher `ppc_delphi_nr_resume_extract` to md5-check and extract all nr tar chunks after download completion.
- [ ] Validate DELPHI nr md5 checksums and extraction. At 2026-06-24 23:41, all 163 tar/md5 files are present, `.aria2` sidecars are gone, and extraction markers have reached `nr.029.tar.gz.extracted`.
- [ ] Decide whether to obtain ANCHOR/IUPred wrapper or mark DELPHI as partially blocked.
- [x] PIPENN: complete train/val/test MMseqs PSSM generation under `benchmark/pipenn_ours_strict/features/*.mm_pssm`.
- [ ] Finish PIPENN NetSurfP-compatible feature generation or document why it remains blocked. `/media/990Pro/aa` contains ProtT5/ProtTrans assets only, not NetSurfP3 standalone weights.
- [x] PIPENN: official BioDL-P prepared-file checkpoint smoke passes using `benchmark/scripts/run_pipenn_dnet_tfkeras.py`; output root `benchmark/results/pipenn_official_feature_smoke_cpu_20260624_215737`.
- [x] EquiPPIS: create isolated `EquiPPIS_cpu` dependency path with CPU PyTorch 1.12.0 and DGL 0.9.0; BIO remains untouched.
- [x] EquiPPIS: official 60-chain sample smoke passes with CPU `map_location` wrapper; output root `benchmark/results/equippis_official_smoke_20260624_214407_cpu_map`.
- [ ] EquiPPIS: decide whether to generate strict-test PDB/DSSP/PSSM/ESM2 features for a supplement-only table.
- [ ] Gated-GPS: defer Google Drive data/weight download and strict feature-builder work until GPSite/DELPHI/PIPENN are controlled.

Reporting rule:

- Main results: our strict sequence/PLM models.
- External MSA-free/deployable: GraphPPIS fast, GPSite.
- External feature-heavy: PIPENN, DELPHI, DeepTMInter/MBPred references.
- Structure-heavy supplementary: Gated-GPS, EquiPPIS, AGAT/GTE/EGCPPIS if run.
