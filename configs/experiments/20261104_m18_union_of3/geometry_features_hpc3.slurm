#!/usr/bin/env bash
#SBATCH --job-name=ppc_geom_prep
#SBATCH --partition=acd_ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PPC_REPO_ROOT:?}"
srun --unbuffered "${PPC_PYTHON:?}" -u scripts/preparations/m18_union/build_geometry_features.py \
  --base "${PPC_BASE:?}" --output "${PPC_EXPERIMENT_ROOT:?}/features" --workers 12
