#!/usr/bin/env bash
#SBATCH --job-name=ppc_esmc_m6
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ppc_esmc_m6_%j.out
#SBATCH --error=logs/ppc_esmc_m6_%j.err

set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${SCORES_DIR:?SCORES_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

"$PYTHON" "$PROJECT_DIR/scripts/analysis/build_esmc_m6_ensemble.py" \
  --m0-val "$SCORES_DIR/m0_val.npz" \
  --m2-val "$SCORES_DIR/m2_val.npz" \
  --m0-test "$SCORES_DIR/m0_test.npz" \
  --m2-test "$SCORES_DIR/m2_test.npz" \
  --output-dir "$OUTPUT_DIR"
