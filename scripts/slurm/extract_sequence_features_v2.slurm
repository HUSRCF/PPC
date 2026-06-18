#!/bin/bash
#SBATCH -J ppc_seq_v2
#SBATCH -p a128m512u
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH -o logs/ppc_seq_v2_%j.out
#SBATCH -e logs/ppc_seq_v2_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
cd "${PROJECT_DIR}"
mkdir -p logs features/sequence_v2

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-128}"

srun --cpu-bind=none conda run -n af3 python scripts/preparations/augment_sequence_features_v2.py \
  --parent-root features/sequence_v1/pt \
  --output-root features/sequence_v2/pt \
  --split-dir features/contact_labels/splits_mmseq30_tmk_no_len_limit \
  --report features/sequence_v2/extract_report.csv \
  --workers "${SLURM_CPUS_PER_TASK:-128}"

echo "End time: $(date)"
