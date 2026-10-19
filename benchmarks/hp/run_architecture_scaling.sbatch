#!/usr/bin/env bash
#SBATCH --job-name=chorus-architecture-scale
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/logs/architecture_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017
SOURCE="${SOURCE:-$ROOT/source_v1/chorus}"
HP_SOURCE="${HP_SOURCE:-$ROOT/source_v1/ictc_o3_hp}"
CASES=(base l3 c192 c256 d3 d4 l3c192d3 l3c256d4)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/run_architecture_case.py" --case "$CASE" --root "$ROOT" \
  --source "$SOURCE" --hp-source "$HP_SOURCE" --result-set "${RESULT_SET:-results}" ${SIZE_FLAGS:-}
printf 'DONE\n'
