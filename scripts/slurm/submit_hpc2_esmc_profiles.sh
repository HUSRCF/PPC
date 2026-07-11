#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
PROFILE_DIR="configs/experiments/20260711_esmc_matrix_hpc2/profiles"
LAUNCHER="scripts/slurm/hpc2_train_esm_site_profiled.slurm"
SUBMISSION_LOG="logs/esmc_profile_submissions_$(date +%Y%m%dT%H%M%S).tsv"

cd "${PROJECT_DIR}"
mkdir -p logs
printf 'job_id\tconfig\n' > "${SUBMISSION_LOG}"

profiles=(
    profile_m0_esm2_mlc_w4.yaml
    profile_m0_esm2_mlc_w8.yaml
    profile_m0_esm2_mlc_w8_b128_t131k.yaml
    profile_m2_esmc_mlc_concat_w8.yaml
    profile_m5_esm2_esmc_gated_residual_w8.yaml
    profile_m5_esm2_esmc_gated_residual_w8_b96_t98k.yaml
)

for profile in "${profiles[@]}"; do
    config="${PROFILE_DIR}/${profile}"
    job_id=$(
        /opt/slurm/bin/sbatch --parsable \
            --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${config}" \
            "${LAUNCHER}"
    )
    printf '%s\t%s\n' "${job_id}" "${config}" | tee -a "${SUBMISSION_LOG}"
done

echo "Submission manifest: ${SUBMISSION_LOG}"
