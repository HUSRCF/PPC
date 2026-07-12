#!/usr/bin/env bash
#SBATCH --job-name=m6_ensemble
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${M0_VAL:?M0_VAL is required}"
: "${M2_VAL:?M2_VAL is required}"
: "${M0_TEST:?M0_TEST is required}"
: "${M2_TEST:?M2_TEST is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

PYTHON_BIN="${PYTHON_BIN:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" scripts/analysis/build_esmc_m6_ensemble.py \
  --m0-val "${M0_VAL}" \
  --m2-val "${M2_VAL}" \
  --m0-test "${M0_TEST}" \
  --m2-test "${M2_TEST}" \
  --output-dir "${OUTPUT_DIR}"
