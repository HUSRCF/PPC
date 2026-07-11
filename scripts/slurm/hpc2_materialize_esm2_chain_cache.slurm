#!/bin/bash
#SBATCH --job-name=ppc_esm2cache
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
PYTHON_BIN="${PYTHON_BIN:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
SPLIT_ROOT="features/contact_labels_5p5A/splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708"

cd "${PROJECT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

"${PYTHON_BIN}" scripts/preparations/materialize_chain_embedding_cache.py \
    --chain-manifest features/esmc_600m_2024_12_mlc_l12_l24_l36/manifest/all_chains.tsv \
    --source-esm-root features/esm2_t33_650M_UR50D_mlc/pt \
    --output-root features/esm2_mlc_chain_cache_20260711
