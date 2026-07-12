#!/usr/bin/env bash
#SBATCH --job-name=ppc_esmc_export
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ppc_esmc_export_%j.out
#SBATCH --error=logs/ppc_esmc_export_%j.err

set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${METHOD:?METHOD is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
SCRIPT="$PROJECT_DIR/scripts/analysis/export_esmc_checkpoint_scores.py"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

for split in val test; do
  "$PYTHON" "$SCRIPT" \
    --project-root "$PROJECT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --split "$split" \
    --method "$METHOD" \
    --output-npz "$OUTPUT_DIR/${METHOD}_${split}.npz" \
    --summary-json "$OUTPUT_DIR/${METHOD}_${split}.summary.json" \
    --device cuda:0 \
    --batch-size 64 \
    --max-batch-tokens 65536 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --payload-cache-size 8
done
