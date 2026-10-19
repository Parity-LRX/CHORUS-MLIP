#!/usr/bin/env bash
#SBATCH --job-name=chorus-highl-mixed
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/highl_mixed_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v43"
export SOURCE="$ROOT/source_v23/chorus" HP_SOURCE="$ROOT/source_v23/ictc_o3_hp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
CASES=(l3 l3c192d3 l3c256d4)
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_mixed_sweep.py" \
  --root "$ROOT" --tools "$TOOLS" --case "${CASES[$SLURM_ARRAY_TASK_ID]}" \
  --atoms 1024 --result-set highl_mixed_screen_v1
printf 'DONE\n'
