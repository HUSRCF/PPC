#!/bin/bash
set -euo pipefail

CONFIG_DIR="configs/experiments/20260622_sequence_only_stage2"
SLURM_SCRIPT="scripts/slurm/run_contact_site_esm_stage2_u.slurm"

sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage2_f1_b4_seqv1_seed42.yaml" "${SLURM_SCRIPT}"
sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage2_f2_b4_seqv2_seed42.yaml" "${SLURM_SCRIPT}"
sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage2_f3_tcn_seqv2_seed42.yaml" "${SLURM_SCRIPT}"
sbatch --export=ALL,CONFIG="${CONFIG_DIR}/train_contact_site_esm_stage2_f4_mlp_seqv2_seed42.yaml" "${SLURM_SCRIPT}"
