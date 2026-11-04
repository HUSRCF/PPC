#!/usr/bin/env bash
#SBATCH --job-name=ppc_C1_fp32
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
CONFIG=${PPC_CONFIG:?}
"$PY" -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]; c=json.load(open(p)); t=c["training"]; assert t["init_checkpoint"] is None and t["epochs"]==50 and t["validation_only"] and not t["eval_test_each_epoch"] and t["seed"]==49; assert c["model"]["linear_attention_layers"]==c["model"]["n_transformer_layers"]==1' "$CONFIG" "${PPC_CONFIG_SHA256:?}"
export PYTHONPATH="$ROOT/src" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 8>"$EXPERIMENT/training.lock"
flock -n 8 || exit 1
cd "$ROOT"
srun --unbuffered "$PY" -u -c 'import torch; assert torch.cuda.is_available(); free,total=torch.cuda.mem_get_info(); print(torch.cuda.get_device_name(0),free/2**30,total/2**30,flush=True); assert free>24*2**30, "Allocated GPU already occupied"'
srun --unbuffered "$PY" -u scripts/preparations/m18_union/smoke_linear_attention_precision.py --output-json "$EXPERIMENT/CUDA_precision_acceptance.json"
srun --unbuffered "$PY" -u scripts/preparations/m18_union/smoke_geometry_training.py --config "$CONFIG"
srun --unbuffered "$PY" -u scripts/training/train_esm_site.py --config "$CONFIG"
