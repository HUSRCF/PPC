#!/usr/bin/env bash
#SBATCH --job-name=ppc_geometry
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
CONFIG=$("$PY" -c 'import json,sys; j=json.load(open(sys.argv[1]))[int(sys.argv[2])]; print(j["config"])' "$EXPERIMENT/jobs.json" "$TASK")
"$PY" -c 'import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); s=json.load(open(r/"features/ready.json")); j=json.load(open(r/"jobs.json"))[int(sys.argv[2])]; assert s["sequences"]==6667 and not s["smoke"] and not s["labels_read"] and s["fit_split"]=="train"; assert s["source_code_sha256"]==hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest(); assert j["sha256"]==hashlib.sha256(pathlib.Path(j["config"]).read_bytes()).hexdigest()' "$EXPERIMENT" "$TASK" "$ROOT/scripts/preparations/m18_union/build_geometry_features.py"
export PYTHONPATH="$ROOT/src" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$EXPERIMENT/locks"
exec 8>"$EXPERIMENT/locks/${TASK}.lock"
flock -n 8 || exit 1
cd "$ROOT"
srun --unbuffered "$PY" -u -c 'import json,os,torch; assert torch.cuda.is_available(); free,total=torch.cuda.mem_get_info(); print(json.dumps({"device":torch.cuda.get_device_name(0),"free_GiB":free/2**30,"total_GiB":total/2**30,"visible":os.getenv("CUDA_VISIBLE_DEVICES"),"step_gpus":os.getenv("SLURM_STEP_GPUS")}),flush=True); assert free > 24*2**30, "Allocated GPU already occupied; do not select unallocated devices"'
srun --unbuffered "$PY" -u scripts/preparations/m18_union/smoke_geometry_training.py --config "$CONFIG"
srun --unbuffered "$PY" -u scripts/training/train_esm_site.py --config "$CONFIG"
