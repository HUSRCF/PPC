# 20260622 sequence-only stage 2 feature-branch matrix

Purpose: test sequence-derived feature branches on top of the clean stage 1
sequence-only backbone. This first batch uses already available strict
sequence-only features:

- `features/sequence_v1/pt`: 130 dims, amino-acid identity/groups,
  physicochemical statistics, and sequence-window features.
- `features/sequence_v2/pt`: 252 dims, sequence_v1 plus BLOSUM62, Atchley,
  positional encodings, extended window statistics, and hydrophobic moments.

These features do not use PDB coordinates, DSSP, SASA/RSA from structures,
predicted structures, or partner-chain information. PSSM, NetSurfP, and HMM
branches are intentionally not submitted in this batch because their full
train/val/test feature payloads are not yet complete.

Fixed protocol:

- Label root: `features/contact_labels_5p5A`
- Split: `features/contact_labels_5p5A/splits_pairwise_si30_cov0_s7p5_tmk_no_len_limit_predstruct_relaxed_largetrain_balanced`
- Input: ESM multilayer embeddings plus sequence-derived features only.
- No predicted-structure features, no predicted-structure contact graph, no PDB coordinates.
- Checkpoint and threshold are selected by validation only.
- Test is evaluated each epoch for logging and reported at validation-selected threshold.
- Metrics include F1, MCC, PR-AUC, AUROC, P@L/20, P@L/10, and P@L/5.
- New dependencies: none.

Configs:

| ID | File | Model change |
| --- | --- | --- |
| F1 | `train_contact_site_esm_stage2_f1_b4_seqv1_seed42.yaml` | Stage1 B4 backbone + sequence_v1 branch. |
| F2 | `train_contact_site_esm_stage2_f2_b4_seqv2_seed42.yaml` | Stage1 B4 backbone + sequence_v2 branch. |
| F3 | `train_contact_site_esm_stage2_f3_tcn_seqv2_seed42.yaml` | TCN-only backbone + sequence_v2 branch. |
| F4 | `train_contact_site_esm_stage2_f4_mlp_seqv2_seed42.yaml` | MLP anchor + sequence_v2 branch. |

Interpretation:

- F2 vs stage1 B4: sequence_v2 feature contribution on the intended backbone.
- F1 vs F2: whether v2 additions improve over v1.
- F3 vs F2: whether global attention is still useful after adding sequence_v2.
- F4 vs stage1 B0 and F2: separate feature gains from backbone gains.

Recommended submission:

```bash
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage2/train_contact_site_esm_stage2_f1_b4_seqv1_seed42.yaml scripts/slurm/run_contact_site_esm_stage2_u.slurm
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage2/train_contact_site_esm_stage2_f2_b4_seqv2_seed42.yaml scripts/slurm/run_contact_site_esm_stage2_u.slurm
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage2/train_contact_site_esm_stage2_f3_tcn_seqv2_seed42.yaml scripts/slurm/run_contact_site_esm_stage2_u.slurm
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage2/train_contact_site_esm_stage2_f4_mlp_seqv2_seed42.yaml scripts/slurm/run_contact_site_esm_stage2_u.slurm
```
