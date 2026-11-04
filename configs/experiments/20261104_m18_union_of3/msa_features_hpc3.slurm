#!/usr/bin/env bash
#SBATCH --job-name=ppc_msa_feature
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
ROOT=${PPC_REPO_ROOT:?}
EXPERIMENT=${PPC_EXPERIMENT_ROOT:?}
PY=${PPC_PYTHON:?}
MODE=${PPC_MSA_MODE:?}
INDEX=${PPC_MSA_INDEX:?}
FASTA=${PPC_MSA_FASTA:?}
if [[ "$MODE" == profile-all ]]; then
    srun --unbuffered "$PY" -u "$ROOT/scripts/preparations/m18_union/run_msa_profile.py" \
        --index "$INDEX" --fasta "$FASTA" --output-root "$EXPERIMENT/features/profile"
    exit 0
fi
SHARDS=${PPC_MSA_SHARDS:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
FEATURE=${MODE#audit-}
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
args=(--mode "$MODE" --index "$INDEX" --fasta "$FASTA"
      --output-root "$EXPERIMENT/features/$FEATURE" --shard "$SHARD" --shards "$SHARDS")
if [[ "$FEATURE" == encoder ]]; then
    args+=(--checkpoint "$EXPERIMENT/weights/esm_msa1b_t12_100M_UR50S.pt" --depth 64)
fi
if [[ "$MODE" == encoder ]]; then
    srun --unbuffered "$PY" -u -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)'
fi
mkdir -p "$EXPERIMENT/locks"
exec 8>"$EXPERIMENT/locks/feature_${MODE}_${SHARD}.lock"
flock -n 8 || exit 1
srun --unbuffered "$PY" -u "$ROOT/scripts/preparations/m18_union/build_msa_features.py" "${args[@]}"
