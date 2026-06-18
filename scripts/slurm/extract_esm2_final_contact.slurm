#!/bin/bash
#SBATCH --job-name=ppc_esm_contact
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ppc_esm_contact_%j.out
#SBATCH --error=logs/ppc_esm_contact_%j.err

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
FEATURES_ROOT="${FEATURES_ROOT:-features/protein_v4/pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-features/esm2_t33_650M_UR50D_contact/pt}"
SPLIT_DIR="${SPLIT_DIR:-features/contact_labels/splits_mmseq30_tmk_no_len_limit}"
ID_LIST="${ID_LIST:-${SPLIT_DIR}/all_ids.txt}"
DEVICES_ARG="${DEVICES_ARG:-}"
MAX_ITEMS="${MAX_ITEMS:-}"

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

if [ ! -f "${ID_LIST}" ]; then
    python - <<'PY'
from pathlib import Path
split_dir = Path("features/contact_labels/splits_mmseq30_tmk_no_len_limit")
out = split_dir / "all_ids.txt"
seen = set()
ids = []
for name in ("train_ids.txt", "val_ids.txt", "test_ids.txt"):
    for line in (split_dir / name).read_text().splitlines():
        item = line.strip()
        if item and not item.startswith("#") and item not in seen:
            seen.add(item)
            ids.append(item)
out.write_text("\n".join(ids) + "\n")
print(f"wrote {out}")
PY
fi

CMD=(
    python scripts/preparations/extract_esm2_from_protein_features.py
    --features-root "${FEATURES_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --id-list "${ID_LIST}"
    --model esm2_t33_650M_UR50D
    --repr-layers 33
    --embedding-combine last
    --save-contacts
    --contact-top-k 16
    --contact-min-score 0.05
    --contact-min-seq-sep 6
    --max-len 1022
    --overlap 768
    --cpu-threads 4
)
if [ -n "${DEVICES_ARG}" ]; then
    CMD+=(--devices "${DEVICES_ARG}")
else
    CMD+=(--device cuda:0)
fi
if [ -n "${MAX_ITEMS}" ]; then
    CMD+=(--max "${MAX_ITEMS}")
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
