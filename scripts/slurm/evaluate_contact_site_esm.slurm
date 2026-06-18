#!/bin/bash
#SBATCH --job-name=ppc_eval_esm
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ppc_eval_esm_%j.out
#SBATCH --error=logs/ppc_eval_esm_%j.err

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
CONFIG="${CONFIG:-configs/train_contact_site_esm_seqv1_reg_mmseq30.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
SPLITS="${SPLITS:-val,test}"

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

CMD=(python scripts/training/evaluate_esm_site.py --config "${CONFIG}" --splits "${SPLITS}")
if [ -n "${CHECKPOINT}" ]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [ -n "${OUTPUT_JSON}" ]; then
    CMD+=(--output-json "${OUTPUT_JSON}")
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Config: ${CONFIG}"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
