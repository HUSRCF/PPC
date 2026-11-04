#!/usr/bin/env bash
#SBATCH --job-name=ppc_geom_msa
#SBATCH --partition=acd_ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
ROOT=${PPC_REPO_ROOT:?}
EXPERIMENT=${PPC_EXPERIMENT_ROOT:?}
PY=${PPC_PYTHON:?}
TASK=${SLURM_ARRAY_TASK_ID:?}
[[ "$ROOT" = /* && "$EXPERIMENT" = /* && "$PY" = /* ]] || {
    printf 'PPC_REPO_ROOT, PPC_EXPERIMENT_ROOT and PPC_PYTHON must be absolute paths\n' >&2
    exit 2
}
export PYTHONPATH="$ROOT/src" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
CONFIG=$("$PY" -u scripts/preparations/m18_union/prepare_geometry_msa_screen.py verify \
    --repo "$ROOT" --experiment "$EXPERIMENT" --task "$TASK")
mkdir -p "$EXPERIMENT/locks"
exec 8>"$EXPERIMENT/locks/${TASK}.lock"
flock -n 8 || exit 1
srun --unbuffered "$PY" -u -c 'import json,torch; assert torch.cuda.is_available(); free,total=torch.cuda.mem_get_info(); print(json.dumps({"device":torch.cuda.get_device_name(0),"free_GiB":free/2**30,"total_GiB":total/2**30}),flush=True); assert free>24*2**30'
srun --unbuffered "$PY" -u scripts/preparations/m18_union/smoke_geometry_training.py --config "$CONFIG"
srun --unbuffered "$PY" -u scripts/training/train_esm_site.py --config "$CONFIG"
