#!/bin/bash
#SBATCH --job-name=ppc_gvp_esm
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/ppc_gvp_esm_%j.out
#SBATCH --error=logs/ppc_gvp_esm_%j.err

# ============================================================
# PPC contact-site GVP + ESM training on HPC.
# ============================================================
# Submit:
#   sbatch scripts/slurm/run_contact_site_gvp_esm_mmseq30.slurm
#
# Override examples:
#   sbatch --export=ALL,BATCH_SIZE=6,MAX_RESIDUES=3072 scripts/slurm/run_contact_site_gvp_esm_mmseq30.slurm
#   sbatch --export=ALL,RUN_NAME=my_run,EPOCHS=5,LR=0.00005 scripts/slurm/run_contact_site_gvp_esm_mmseq30.slurm
#   sbatch --export=ALL,MAX_RESIDUES=0,BATCH_SIZE=1 scripts/slurm/run_contact_site_gvp_esm_mmseq30.slurm
#
# Matching interactive allocation:
#   srun -p i64m1tga800ue -n 1 --mem=96G -c 8 --gres=gpu:1 --time=36:00:00 --pty bash
#
# Notes:
#   - Follows the PLC Slurm environment style used by Wang Tingzhi:
#     conda environment "af3" and the i64m1tga800ue GPU partition.
#   - MAX_RESIDUES=0 disables cropping; default 3072 is the safer first HPC run.
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
RUN_NAME="${RUN_NAME:-contact_site_gvp_esm_t33_seed42_r3072_e3_lr5e5_warmup5p_freezenorm_safeclip_mmseq30_tmk_no_len_limit_hpc_b4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"

FEATURES_ROOT="${FEATURES_ROOT:-features/protein_v4/pt}"
LABEL_ROOT="${LABEL_ROOT:-features/contact_labels}"
ESM_ROOT="${ESM_ROOT:-features/esm2_t33_650M_UR50D/pt}"
MANIFEST="${MANIFEST:-features/contact_labels/manifest.csv}"
SPLIT_DIR="${SPLIT_DIR:-features/contact_labels/splits_mmseq30_tmk_no_len_limit}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_RESIDUES="${MAX_RESIDUES:-3072}"
LR="${LR:-0.00005}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MIN_LR_SCALE="${MIN_LR_SCALE:-0.02}"
SEED="${SEED:-42}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-}"
AMP_FLAG="${AMP_FLAG:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Run name: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Split: ${SPLIT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max residues: ${MAX_RESIDUES}"
echo "LR: ${LR}"
echo "Warmup ratio: ${WARMUP_RATIO}"
echo "============================================"

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

echo "Python: $(which python)"
python --version

echo "Torch/CUDA:"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

echo "GPU devices:"
if command -v rocm-smi &> /dev/null; then
    rocm-smi
elif command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No rocm-smi/nvidia-smi found"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
    python scripts/training/train_contact_site.py
    --features-root "${FEATURES_ROOT}"
    --label-root "${LABEL_ROOT}"
    --esm-root "${ESM_ROOT}"
    --manifest "${MANIFEST}"
    --split-dir "${SPLIT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --device cuda:0
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-residues "${MAX_RESIDUES}"
    --max-val-samples "${MAX_VAL_SAMPLES}"
    --lr "${LR}"
    --warmup-ratio "${WARMUP_RATIO}"
    --min-lr-scale "${MIN_LR_SCALE}"
    --use-esm
    --d-esm 1280
    --use-rmsnorm
    --freeze-norms
    --safe-grad-clip
    --grad-clip-norm 1.0
    --grad-value-clip 100.0
    --adam-eps 0.0000001
    --no-adam-foreach
    --progress
)

if [ -n "${EVAL_MAX_BATCHES}" ]; then
    CMD+=(--eval-max-batches "${EVAL_MAX_BATCHES}")
fi
if [ -n "${AMP_FLAG}" ]; then
    CMD+=(--amp)
fi
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "WARNING: RESUME_CHECKPOINT is set, but train_contact_site.py does not currently implement resume loading."
fi

echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}
