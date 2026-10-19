#!/usr/bin/env bash
#SBATCH --job-name=chorus-large-training
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=1-5%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/large_training_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v40"
export SOURCE="$ROOT/source_v23/chorus" HP_SOURCE="$ROOT/source_v23/ictc_o3_hp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
CASES=(atom_scale l3 c256 d4 l3c192d3 l3c256d4)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_scale_matrix.py" \
  --root "$ROOT" --tools "$TOOLS" --case "$CASE" --result-set large_training_v2 --repeats 50 --warmup 5
printf 'DONE\n'
