#!/usr/bin/env bash
#SBATCH --job-name=chorus-first-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/first_pair_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v76"
export SOURCE="$ROOT/source_v40/chorus" HP_SOURCE="$ROOT/source_v40/ictc_o3_hp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
: "${CASE:?Specify l3 or l3c192d3}"
if [[ "$CASE" == l3 ]]; then ORACLE=1024; elif [[ "$CASE" == l3c192d3 ]]; then ORACLE=512; else exit 2; fi
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_scale_matrix.py" \
  --root "$ROOT" --tools "$TOOLS" --case "$CASE" --sizes 1024 \
  --result-set highl_parameter_matched_v6 --repeats 50 --warmup 5 \
  --parameter-matched-dpa-root "$ROOT/parameter_matched_v1" --native-baseline \
  --selection-screen first_cached_screen_v6 wide_cached_screen_v7 radial_tiles_screen_v8 fused_jvp_screen_v9 group_pilot_screen_v10 first_fusion_screen_v13 radial_first_screen_v14 first_scale_screen_v15 \
  --selection-qualification first_cached_qual_v6 wide_cached_qual_v7 radial_tiles_qual_v8 fused_jvp_qual_v9 group_pilot_qual_v10 first_fusion_qual_v13 radial_first_qual_v14 first_scale_qual_v15 \
  --oracle-atoms "$ORACLE" --large-oracle-save-on-cpu --large-oracle-chunk-edges 1024
printf 'DONE\n'
