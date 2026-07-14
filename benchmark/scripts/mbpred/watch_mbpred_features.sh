#!/usr/bin/env bash
set -euo pipefail

PY=${MBPRED_PYTHON:-/home/husrcf/anaconda3/envs/AIAA/bin/python}
DTM_ROOT=${DEEPTMINTER_RUN_ROOT:-/data/husrcf/ProtBind/PPC/benchmark/runs/chain_filtered_global_si30_20260709/deeptminter_official_full}
MB_ROOT=${MBPRED_MODEL_ROOT:-/data/husrcf/ProtBind/PPC/benchmark/models/mbpred}
OUT_ROOT=${MBPRED_RUN_ROOT:-/data/husrcf/ProtBind/PPC/benchmark/runs/chain_filtered_global_si30_20260709/mbpred_adapted}
POLL_SECONDS=${MBPRED_POLL_SECONDS:-300}
WORKERS=${MBPRED_FEATURE_WORKERS:-2}
EXPECTED_UNIQUE=${MBPRED_EXPECTED_UNIQUE:-724}
PRODUCER_TMUX=${DEEPTMINTER_TMUX_SESSION:-deeptminter_full_8x16_0714}
RUN_ONCE=${MBPRED_WATCH_ONCE:-0}
MAX_FAILURE_STREAK=${MBPRED_MAX_FAILURE_STREAK:-3}

case "$POLL_SECONDS" in
    ''|*[!0-9]*) echo "MBPRED_POLL_SECONDS must be a positive integer" >&2; exit 2 ;;
esac
case "$WORKERS" in
    ''|*[!0-9]*) echo "MBPRED_FEATURE_WORKERS must be a positive integer" >&2; exit 2 ;;
esac
case "$EXPECTED_UNIQUE" in
    ''|*[!0-9]*) echo "MBPRED_EXPECTED_UNIQUE must be a positive integer" >&2; exit 2 ;;
esac
case "$MAX_FAILURE_STREAK" in
    ''|*[!0-9]*) echo "MBPRED_MAX_FAILURE_STREAK must be a positive integer" >&2; exit 2 ;;
esac
if (( POLL_SECONDS < 1 || WORKERS < 1 || WORKERS > 2 \
    || EXPECTED_UNIQUE < 1 || MAX_FAILURE_STREAK < 1 || MAX_FAILURE_STREAK > 10 )); then
    echo "poll/expected counts must be positive and workers must be in [1,2]" >&2
    exit 2
fi
if [[ "$RUN_ONCE" != 0 && "$RUN_ONCE" != 1 ]]; then
    echo "MBPRED_WATCH_ONCE must be 0 or 1" >&2
    exit 2
fi

BUILDER="$MB_ROOT/adapters/build_mbpred_features.py"
AAINDEX="$MB_ROOT/source/AAindex/AAindex"
FEATURE_DIR="$DTM_ROOT/features"
OUTPUT_DIR="$OUT_ROOT/features"
QC_DIR="$OUT_ROOT/qc"
LOG_DIR="$OUT_ROOT/logs"
STATE_DIR="$OUT_ROOT/state"

for required in "$PY" "$BUILDER" "$AAINDEX" "$FEATURE_DIR"; do
    if [[ ! -e "$required" ]]; then
        echo "required path is missing: $required" >&2
        exit 2
    fi
done
mkdir -p "$OUTPUT_DIR" "$QC_DIR" "$LOG_DIR" "$STATE_DIR"

exec 9>"$STATE_DIR/feature_watch.lock"
if ! flock -n 9; then
    echo "another MBPred feature watcher holds $STATE_DIR/feature_watch.lock" >&2
    exit 2
fi

STATUS="$STATE_DIR/feature_watch.status.tsv"
STOP_FILE="$STATE_DIR/STOP_FEATURE_WATCH"

record_status() {
    local state=$1
    local detail=$2
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$state" "$detail" >> "$STATUS"
}

build_one() {
    local run_id=$1
    local prefix="$FEATURE_DIR/${run_id}A"
    "$PY" "$BUILDER" \
        --fasta "$prefix.fasta" \
        --alignment "$prefix.aln" \
        --couplings "$prefix.evfold" \
        --phobius "$prefix.jphobius" \
        --aaindex "$AAINDEX" \
        --output "$OUTPUT_DIR/${run_id}A.features.tsv" \
        --qc-json "$QC_DIR/${run_id}A.features.qc.json" \
        > "$LOG_DIR/${run_id}.feature.stdout.log" \
        2> "$LOG_DIR/${run_id}.feature.stderr.log"
}
export PY BUILDER AAINDEX FEATURE_DIR OUTPUT_DIR QC_DIR LOG_DIR
export -f build_one

record_status started \
    "workers=$WORKERS poll_seconds=$POLL_SECONDS expected=$EXPECTED_UNIQUE official_omissions=qc_zero_fill"
failure_streak=0
while true; do
    if [[ -e "$STOP_FILE" ]]; then
        record_status stopped "stop_file=$STOP_FILE"
        exit 0
    fi

    mapfile -t pending < <(
        for coupling in "$FEATURE_DIR"/dtm[0-9][0-9][0-9][0-9]A.evfold; do
            [[ -e "$coupling" ]] || continue
            filename=${coupling##*/}
            run_id=${filename%A.evfold}
            prefix="$FEATURE_DIR/${run_id}A"
            producer_marker="$DTM_ROOT/qc/$run_id/freecontact.complete"
            if [[ ! -f "$prefix.fasta" || ! -f "$prefix.aln" \
                || ! -f "$prefix.jphobius" || ! -f "$producer_marker" ]]; then
                continue
            fi
            if [[ ! -f "$OUTPUT_DIR/${run_id}A.features.tsv" \
                || ! -f "$QC_DIR/${run_id}A.features.qc.json" ]]; then
                printf '%s\n' "$run_id"
            fi
        done | sort -u
    )

    if (( ${#pending[@]} > 0 )); then
        record_status building "count=${#pending[@]} first=${pending[0]}"
        if ! printf '%s\0' "${pending[@]}" \
            | xargs -0 -r -n 1 -P "$WORKERS" bash -c 'build_one "$1"' _; then
            ((failure_streak += 1))
            record_status build_failed \
                "streak=$failure_streak max=$MAX_FAILURE_STREAK count=${#pending[@]}"
            if [[ "$RUN_ONCE" == 1 ]] || (( failure_streak >= MAX_FAILURE_STREAK )); then
                exit 3
            fi
            sleep "$POLL_SECONDS"
            continue
        fi
        failure_streak=0
    fi

    built=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'dtm[0-9][0-9][0-9][0-9]A.features.tsv' -printf '.' | wc -c)
    qc=$(find "$QC_DIR" -maxdepth 1 -type f -name 'dtm[0-9][0-9][0-9][0-9]A.features.qc.json' -printf '.' | wc -c)
    record_status idle "built=$built qc=$qc"
    if (( built == EXPECTED_UNIQUE && qc == EXPECTED_UNIQUE )); then
        record_status complete "built=$built qc=$qc"
        exit 0
    fi
    if (( built > EXPECTED_UNIQUE || qc > EXPECTED_UNIQUE )); then
        record_status failed "count_exceeds_contract built=$built qc=$qc"
        exit 3
    fi
    if [[ "$RUN_ONCE" == 1 ]]; then
        record_status complete_once "built=$built qc=$qc"
        exit 0
    fi
    if ! tmux has-session -t "$PRODUCER_TMUX" 2>/dev/null; then
        record_status failed "producer_tmux_missing=$PRODUCER_TMUX built=$built"
        exit 3
    fi
    sleep "$POLL_SECONDS"
done
