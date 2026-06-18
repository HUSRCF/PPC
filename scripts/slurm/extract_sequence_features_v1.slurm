#!/bin/bash
#SBATCH -J ppc_seq_v1
#SBATCH -p a128m512u
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH -o logs/ppc_seq_v1_%j.out
#SBATCH -e logs/ppc_seq_v1_%j.err

set -euo pipefail

cd /hpc2hdd/home/shuang886/jhspoolers/PPC
mkdir -p logs features/sequence_v1

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-128}"

srun --cpu-bind=none conda run -n af3 python scripts/preparations/extract_sequence_features_v1.py \
  --esm-root features/esm2_t33_650M_UR50D/pt \
  --label-root features/contact_labels \
  --pdb-root data/pdbtm/cplx_pdbfixer \
  --split-dir features/contact_labels/splits_mmseq30_tmk_no_len_limit \
  --output-root features/sequence_v1/pt \
  --report features/sequence_v1/extract_report.csv \
  --workers "${SLURM_CPUS_PER_TASK:-128}"

echo "End time: $(date)"
