#!/bin/bash
#SBATCH --job-name=ppc_eval_strict
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ppc_eval_strict_%j.out
#SBATCH --error=logs/ppc_eval_strict_%j.err

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
CONFIG="${CONFIG:-configs/train_contact_site_esm_multilayer_contact_reg_mmseq30.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
STRICT_OUTPUT_JSON="${STRICT_OUTPUT_JSON:-}"
STRICT_OUTPUT_CSV="${STRICT_OUTPUT_CSV:-}"
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

if [ -z "${OUTPUT_JSON}" ]; then
    OUTPUT_JSON="runs/eval_${SLURM_JOB_ID}.json"
fi

CMD=(python scripts/training/evaluate_esm_site.py --config "${CONFIG}" --splits "${SPLITS}" --output-json "${OUTPUT_JSON}")
if [ -n "${CHECKPOINT}" ]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Config: ${CONFIG}"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

SUMMARY_CMD=(python scripts/training/summarize_eval_at_val_threshold.py --eval-json "${OUTPUT_JSON}")
if [ -n "${STRICT_OUTPUT_JSON}" ]; then
    SUMMARY_CMD+=(--output-json "${STRICT_OUTPUT_JSON}")
fi
if [ -n "${STRICT_OUTPUT_CSV}" ]; then
    SUMMARY_CMD+=(--output-csv "${STRICT_OUTPUT_CSV}")
fi

echo "Summary command: ${SUMMARY_CMD[*]}"
"${SUMMARY_CMD[@]}"
