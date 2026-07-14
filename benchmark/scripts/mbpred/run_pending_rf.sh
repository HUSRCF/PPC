#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${1:?usage: run_pending_rf.sh RUN_ROOT}
WORKERS=${MBPRED_RF_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
R_BIN=${MBPRED_R_BIN:-/usr/bin/Rscript}
R_SCRIPT=${MBPRED_R_SCRIPT:-$RUN_ROOT/predict_mbpred_rf_base.R}
CHECKPOINT=${MBPRED_CHECKPOINT:-$RUN_ROOT/AllFit.rf}
FEATURE_DIR=$RUN_ROOT/features
PREDICTION_DIR=$RUN_ROOT/predictions
STATE_DIR=$RUN_ROOT/state

case "$WORKERS" in
    ''|*[!0-9]*) echo "MBPRED_RF_WORKERS must be a positive integer" >&2; exit 2 ;;
esac
if (( WORKERS < 1 || WORKERS > 32 )); then
    echo "MBPRED_RF_WORKERS must be in [1,32]" >&2
    exit 2
fi
for required in "$R_BIN" "$R_SCRIPT" "$CHECKPOINT" "$FEATURE_DIR"; do
    if [[ ! -e "$required" ]]; then
        echo "required path is missing: $required" >&2
        exit 2
    fi
done
mkdir -p "$PREDICTION_DIR" "$STATE_DIR"

exec 9>"$STATE_DIR/rf_batch.lock"
if ! flock -n 9; then
    echo "another MBPred RF batch holds $STATE_DIR/rf_batch.lock" >&2
    exit 2
fi

output_valid() {
    local feature_path=$1
    local output=$2
    local header
    local feature_rows
    local output_rows

    [[ -s "$output" ]] || return 1
    IFS= read -r header < "$output" || return 1
    [[ "$header" == '"","Res","AA","Score"' ]] || return 1
    feature_rows=$(wc -l < "$feature_path")
    output_rows=$(wc -l < "$output")
    [[ "$output_rows" -eq "$feature_rows" ]]
}

job_tag=${SLURM_JOB_ID:-manual_$$}.${SLURM_ARRAY_TASK_ID:-0}
discovered_tmp=$STATE_DIR/discovered_rf.${job_tag}.nul.tmp
discovered=$STATE_DIR/discovered_rf.${job_tag}.nul
pending_tmp=$STATE_DIR/pending_rf.${job_tag}.nul.tmp
pending=$STATE_DIR/pending_rf.${job_tag}.nul
find "$FEATURE_DIR" -maxdepth 1 -type f \
    -name 'dtm[0-9][0-9][0-9][0-9]A.features.tsv' -print0 \
    | sort -z > "$discovered_tmp"
mv "$discovered_tmp" "$discovered"
: > "$pending_tmp"
pending_count=0
while IFS= read -r -d '' feature_path; do
    filename=${feature_path##*/}
    run_chain=${filename%.features.tsv}
    output=$PREDICTION_DIR/${run_chain}.predictions.csv
    if ! output_valid "$feature_path" "$output"; then
        printf '%s\0' "$feature_path" >> "$pending_tmp"
        ((pending_count += 1))
    fi
done < "$discovered"
mv "$pending_tmp" "$pending"

run_one() {
    local feature_path=$1
    local filename=${feature_path##*/}
    local run_chain=${filename%.features.tsv}
    local output=$PREDICTION_DIR/${run_chain}.predictions.csv
    local tmp=$output.tmp.$BASHPID

    if output_valid "$feature_path" "$output"; then
        return 0
    fi
    "$R_BIN" "$R_SCRIPT" "$CHECKPOINT" "$feature_path" "$tmp"
    if ! output_valid "$feature_path" "$tmp"; then
        echo "$run_chain: RF output failed header/row-count validation" >&2
        return 3
    fi
    mv "$tmp" "$output"
}
export R_BIN R_SCRIPT CHECKPOINT PREDICTION_DIR
export -f output_valid run_one

printf 'job=%s workers=%s pending=%s\n' "$job_tag" "$WORKERS" "$pending_count"
if (( pending_count == 0 )); then
    exit 0
fi
xargs -0 -r -n 1 -P "$WORKERS" bash -c 'run_one "$1"' _ < "$pending"

completed=0
while IFS= read -r -d '' feature_path; do
    filename=${feature_path##*/}
    run_chain=${filename%.features.tsv}
    if output_valid "$feature_path" "$PREDICTION_DIR/${run_chain}.predictions.csv"; then
        ((completed += 1))
    fi
done < "$pending"
printf 'job=%s completed=%s pending=%s\n' "$job_tag" "$completed" "$pending_count"
if (( completed != pending_count )); then
    exit 3
fi
