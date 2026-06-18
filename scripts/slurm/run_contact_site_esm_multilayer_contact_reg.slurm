#!/bin/bash
#SBATCH --job-name=ppc_esm_mlc_reg
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ppc_esm_mlc_reg_%j.out
#SBATCH --error=logs/ppc_esm_mlc_reg_%j.err

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
CONFIG="${CONFIG:-configs/train_contact_site_esm_multilayer_contact_reg_mmseq30.yaml}"

mkdir -p logs

if command -v module &> /dev/null; then
    module purge || true
fi

if command -v conda &> /dev/null; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda command not found in PATH"
    exit 1
fi
conda activate af3

cd "${PROJECT_DIR}" || exit 1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/training/train_esm_site.py --config "${CONFIG}")

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Config: ${CONFIG}"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
