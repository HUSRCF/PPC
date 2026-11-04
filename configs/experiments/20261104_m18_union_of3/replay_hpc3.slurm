#!/usr/bin/env bash
#SBATCH --job-name=ppc_union_replay
#SBATCH --partition=acd_ue
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=ppc_union_replay_%A_%a.out
#SBATCH --error=ppc_union_replay_%A_%a.err
set -euo pipefail
ROOT=${PPC_REPO_ROOT:?Set PPC_REPO_ROOT to this PPC checkout}
PY=${PPC_PYTHON:?Set PPC_PYTHON to an existing verified Python; do not install Torch}
MATRIX=${PPC_MATRIX:-ratio-head}
if [[ "$MATRIX" == ratio-head ]]; then
    variants=(R0 R1 R2)
elif [[ "$MATRIX" == loss ]]; then
    variants=(A B C D E)
else
    printf 'PPC_MATRIX must be loss or ratio-head\n' >&2
    exit 1
fi
variant=${variants[$SLURM_ARRAY_TASK_ID]}
CONFIG="$ROOT/configs/experiments/20261104_m18_union_of3/${variant}.json"
LOCK_PATH=$("$PY" -c 'import json, pathlib, sys; print(pathlib.Path(json.load(open(sys.argv[1]))["training"]["output_dir"]).with_suffix(".lock"))' "$CONFIG")
mkdir -p "$(dirname "$LOCK_PATH")"
exec 8>"$LOCK_PATH"
flock -n 8 || exit 1
export PYTHONPATH="$ROOT/src" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
srun --unbuffered "$PY" -u -c 'import torch; assert torch.cuda.is_available(), "Allocated CUDA device required"; print(torch.cuda.get_device_name(0), flush=True)'
srun --unbuffered "$PY" -u scripts/training/train_esm_site.py \
    --config "$CONFIG"
