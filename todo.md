# PPC Sequence-only PPIS TODO

Last updated: 2026-06-22

## Scope

Current mainline is strict sequence-only PPIS. Predicted-structure and PDBTM-derived
structure branches are excluded from the primary model path because the PDBTM to
UniProt to predicted-structure alignment is not reliable enough for a defensible
main result. Structure-dependent branches remain future engineering work or
supplementary baselines only.

Fixed protocol:

- Task: non-partner-specific PPIS residue prediction.
- Input: one protein sequence plus sequence-derived features only.
- Label: 5.5A residue contact tag.
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
