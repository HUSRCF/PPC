#!/usr/bin/env bash
#SBATCH --job-name=ppc_msa_context
#SBATCH --partition=acd_ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
ROOT=${PPC_REPO_ROOT:?}
EXPERIMENT=${PPC_EXPERIMENT_ROOT:?}
PY=${PPC_PYTHON:?}
TASK=${SLURM_ARRAY_TASK_ID:?}
read -r CONFIG EVAL_CONFIG RUN < <("$PY" -c 'import json,sys; j=json.load(open(sys.argv[1]))[int(sys.argv[2])]; print(j["config"],j["eval_config"],j["output"])' "$EXPERIMENT/jobs.json" "$TASK")
"$PY" -c 'import hashlib,json,pathlib,sys; j=json.load(open(sys.argv[1]))[int(sys.argv[2])]; p=pathlib.Path(j["config"]); assert hashlib.sha256(p.read_bytes()).hexdigest()==j["sha256"]; c=json.load(open(p)); s=json.load(open(pathlib.Path(c["data"]["prottrans_embedding_root"])/"validation_summary.json")); assert s["sequences"]==6667 and s["full_length"] and not s["labels_read"] and s["mode"]=="encoder" and s["dimension"]==768' "$EXPERIMENT/jobs.json" "$TASK"
export PYTHONPATH="$ROOT/src" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$EXPERIMENT/locks"
exec 8>"$EXPERIMENT/locks/${TASK}.lock"
flock -n 8 || exit 1
cd "$ROOT"
srun --unbuffered "$PY" -u -c 'import torch; assert torch.cuda.is_available(); free,total=torch.cuda.mem_get_info(); print(torch.cuda.get_device_name(0),free/2**30,total/2**30,flush=True); assert free>24*2**30, "Allocated GPU already occupied; never select unallocated GPUs"'
srun --unbuffered "$PY" -u scripts/preparations/m18_union/smoke_msa_context_training.py --config "$CONFIG"
srun --unbuffered "$PY" -u scripts/training/train_esm_site.py --config "$CONFIG"
"$PY" -c 'import hashlib,json,pathlib,sys; j=json.load(open(sys.argv[1]))[int(sys.argv[2])]; assert hashlib.sha256(pathlib.Path(j["eval_config"]).read_bytes()).hexdigest()==j["eval_sha256"]' "$EXPERIMENT/jobs.json" "$TASK"
srun --unbuffered "$PY" -u scripts/training/evaluate_esm_site.py --config "$EVAL_CONFIG" --checkpoint "$RUN/best.pt" --splits train,val --output-json "$RUN/train_val_best.json"
