#!/usr/bin/env bash
#SBATCH --job-name=chorus-cached-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/cached_pair_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v51"
export SOURCE="$ROOT/source_v29/chorus" HP_SOURCE="$ROOT/source_v29/ictc_o3_hp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
CASES=(l3 l3c192d3 l3c256d4)
case "$SLURM_ARRAY_TASK_ID" in
  0) SIZES=(1024 4096);;
  1) SIZES=(1024 2048);;
  2) SIZES=(1024 1536 1664);;
esac
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_scale_matrix.py" \
  --root "$ROOT" --tools "$TOOLS" --case "${CASES[$SLURM_ARRAY_TASK_ID]}" \
  --sizes "${SIZES[@]}" --result-set highl_cached_pair_v2 --repeats 50 --warmup 5 \
  --selection-screen highl_cached_screen_v3 --selection-qualification highl_cached_qual_v3
printf 'DONE\n'
