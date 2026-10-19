#!/usr/bin/env bash
#SBATCH --job-name=chorus-latest-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/latest_pair_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v59"
export SOURCE="$ROOT/source_v37/chorus" HP_SOURCE="$ROOT/source_v37/ictc_o3_hp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
CASES=(l3 l3c192d3 l3c256d4)
ORACLE=()
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  ORACLE=(--large-oracle-save-on-cpu --large-oracle-chunk-edges 1024)
fi
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_scale_matrix.py" \
  --root "$ROOT" --tools "$TOOLS" --case "${CASES[$SLURM_ARRAY_TASK_ID]}" \
  --sizes 1024 --result-set highl_latest_pair_v4 --repeats 50 --warmup 5 \
  --selection-screen first_cached_screen_v6 wide_cached_screen_v7 radial_tiles_screen_v8 fused_jvp_screen_v9 group_pilot_screen_v10 \
  --selection-qualification first_cached_qual_v6 wide_cached_qual_v7 radial_tiles_qual_v8 fused_jvp_qual_v9 group_pilot_qual_v10 \
  "${ORACLE[@]}"
printf 'DONE\n'
