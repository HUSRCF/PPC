#!/bin/bash
#SBATCH --job-name=ppc_esmc
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
PYTHON_BIN="${PYTHON_BIN:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CONFIG="${CONFIG:?CONFIG is required}"
GPU_LOG="${PROJECT_DIR}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv"

cd "${PROJECT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "timestamp,index,utilization_gpu,memory_used,memory_total,power_draw" > "${GPU_LOG}"
(
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits >> "${GPU_LOG}" || true
        sleep 5
    done
) &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config: ${CONFIG}"
echo "Python: ${PYTHON_BIN}"
echo "GPU log: ${GPU_LOG}"
"${PYTHON_BIN}" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
"${PYTHON_BIN}" scripts/training/train_esm_site.py --config "${CONFIG}"
