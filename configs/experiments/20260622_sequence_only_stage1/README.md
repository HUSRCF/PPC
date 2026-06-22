# 20260622 sequence-only stage 1 backbone matrix

Purpose: evaluate clean sequence-only backbone upgrades before adding
sequence-derived PSSM/NetSurfP/HMM feature branches.

Fixed protocol:

- Label root: `features/contact_labels_5p5A`
- Split: `features/contact_labels_5p5A/splits_pairwise_si30_cov0_s7p5_tmk_no_len_limit_predstruct_relaxed_largetrain_balanced`
- Input: ESM multilayer embeddings plus sequence metadata/basic amino-acid features only.
- No predicted-structure features, no predicted-structure contact graph, no PDB coordinates.
- Checkpoint and threshold are selected by validation only.
- Test is evaluated each epoch for logging and reported at validation-selected threshold.
- Metrics include F1, MCC, PR-AUC, AUROC, P@L/20, P@L/10, and P@L/5.
- New dependency check: none. TCN uses only `torch.nn.Conv1d`; global attention uses `torch.nn.TransformerEncoder`.

Configs:

| ID | File | Model change |
| --- | --- | --- |
| B0 | `train_contact_site_esm_stage1_b0_mlc_mlp_seed42.yaml` | MLC MLP anchor. |
| B1 | `train_contact_site_esm_stage1_b1_mlc_tcn_seed42.yaml` | Add two local TCN layers. |
| B2 | `train_contact_site_esm_stage1_b2_mlc_dilated_tcn_seed42.yaml` | Add four dilated TCN layers. |
| B3 | `train_contact_site_esm_stage1_b3_mlc_global_attn_seed42.yaml` | Add one lightweight TransformerEncoder layer. |
| B4 | `train_contact_site_esm_stage1_b4_mlc_tcn_global_seed42.yaml` | TCN plus one lightweight TransformerEncoder layer. |

Recommended submission:

```bash
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage1/train_contact_site_esm_stage1_b0_mlc_mlp_seed42.yaml scripts/slurm/run_contact_site_esm_stage1_u.slurm
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage1/train_contact_site_esm_stage1_b1_mlc_tcn_seed42.yaml scripts/slurm/run_contact_site_esm_stage1_u.slurm
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage1/train_contact_site_esm_stage1_b2_mlc_dilated_tcn_seed42.yaml scripts/slurm/run_contact_site_esm_stage1_u.slurm
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage1/train_contact_site_esm_stage1_b3_mlc_global_attn_seed42.yaml scripts/slurm/run_contact_site_esm_stage1_u.slurm
sbatch --export=ALL,CONFIG=configs/experiments/20260622_sequence_only_stage1/train_contact_site_esm_stage1_b4_mlc_tcn_global_seed42.yaml scripts/slurm/run_contact_site_esm_stage1_u.slurm
```
