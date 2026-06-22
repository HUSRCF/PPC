#!/bin/bash
set -euo pipefail

CONFIG_DIR="configs/experiments/20260622_sequence_only_stage1"
SLURM_SCRIPT="scripts/slurm/run_contact_site_esm_stage1_u.slurm"

sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage1_b0_mlc_mlp_seed42.yaml" "${SLURM_SCRIPT}"
sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage1_b1_mlc_tcn_seed42.yaml" "${SLURM_SCRIPT}"
sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage1_b2_mlc_dilated_tcn_seed42.yaml" "${SLURM_SCRIPT}"
sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage1_b3_mlc_global_attn_seed42.yaml" "${SLURM_SCRIPT}"
sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage1_b4_mlc_tcn_global_seed42.yaml" "${SLURM_SCRIPT}"
