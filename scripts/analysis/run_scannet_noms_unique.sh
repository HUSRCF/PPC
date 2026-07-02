#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/husrcf/ProtBind/PPC/benchmark}"
RUN_ID="${RUN_ID:-strict_residue_effect_site_20260626}"
SCANNET_ROOT="${SCANNET_ROOT:-$ROOT/models/scannet/source/ScanNet}"
SCANNET_PY="${SCANNET_PY:-/home/husrcf/anaconda3/envs/scannet_tf114/bin/python}"
RUN_LIST="${RUN_LIST:-$ROOT/runs/$RUN_ID/scannet_noMSA_unique/unique_run_list.tsv}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/$RUN_ID/scannet_noMSA_unique_predictions}"
LOG_ROOT="${LOG_ROOT:-$ROOT/runs/$RUN_ID/logs/scannet_noMSA_unique}"
STATUS_TSV="${STATUS_TSV:-$ROOT/results/$RUN_ID/scannet_noMSA_unique_status.tsv}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-0}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$(dirname "$STATUS_TSV")"

if [[ ! -s "$RUN_LIST" ]]; then
  echo "[ERROR] missing RUN_LIST=$RUN_LIST" >&2
  exit 2
fi
if [[ ! -x "$SCANNET_PY" ]]; then
  echo "[ERROR] missing executable SCANNET_PY=$SCANNET_PY" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-5}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-5}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-5}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

if [[ ! -s "$STATUS_TSV" ]]; then
  printf "timestamp\tinput_id\tstatus\tcsv_path\tlog_path\n" > "$STATUS_TSV"
fi

idx=0
ran=0
tail -n +2 "$RUN_LIST" | while IFS=$'\t' read -r input_id pdb_path source length n_seq_ids_needed first_seq_id seq_ids; do
  if (( idx < START_INDEX )); then
    idx=$((idx + 1))
    continue
  fi
  if (( LIMIT > 0 && ran >= LIMIT )); then
    break
  fi
  idx=$((idx + 1))
  ran=$((ran + 1))

  out_dir="$OUT_ROOT/$input_id"
  log_path="$LOG_ROOT/${input_id}.log"
  csv_path="$(find "$out_dir" -type f -name "predictions_${input_id}.csv" -print -quit 2>/dev/null || true)"
  if [[ -n "$csv_path" && -s "$csv_path" ]]; then
    echo "[SKIP] $input_id $csv_path"
    printf "%s\t%s\tSKIP\t%s\t%s\n" "$(date '+%F %T %Z')" "$input_id" "$csv_path" "$log_path" >> "$STATUS_TSV"
    continue
  fi
  if [[ ! -s "$pdb_path" ]]; then
    echo "[ERROR] missing pdb for $input_id: $pdb_path" >&2
    printf "%s\t%s\tERROR_MISSING_PDB\t\t%s\n" "$(date '+%F %T %Z')" "$input_id" "$log_path" >> "$STATUS_TSV"
    exit 4
  fi

  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  echo "[RUN] $input_id len=$length pdb=$pdb_path"
  (
    cd "$SCANNET_ROOT"
    "$SCANNET_PY" predict_bindingsites.py \
      "$pdb_path" \
      --name "$input_id" \
      --noMSA \
      --mode interface \
      --pdb \
      --predictions_folder "$out_dir/"
  ) > "$log_path" 2>&1

  csv_path="$(find "$out_dir" -type f -name "predictions_${input_id}.csv" -print -quit 2>/dev/null || true)"
  if [[ -z "$csv_path" || ! -s "$csv_path" ]]; then
    echo "[ERROR] ScanNet output missing for $input_id; see $log_path" >&2
    printf "%s\t%s\tERROR_NO_CSV\t\t%s\n" "$(date '+%F %T %Z')" "$input_id" "$log_path" >> "$STATUS_TSV"
    exit 5
  fi
  printf "%s\t%s\tOK\t%s\t%s\n" "$(date '+%F %T %Z')" "$input_id" "$csv_path" "$log_path" >> "$STATUS_TSV"
done

echo "[OK] ScanNet noMSA batch finished or reached LIMIT"
