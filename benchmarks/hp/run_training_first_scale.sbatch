#!/usr/bin/env bash
#SBATCH --job-name=chorus-first-scale
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/first_scale_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v72"
export SOURCE="$ROOT/source_v40/chorus" HP_SOURCE="$ROOT/source_v40/ictc_o3_hp"
CACHE="$ROOT/cache/native"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
for CASE in l3 l3c192d3; do
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_mixed_sweep.py" \
    --root "$ROOT" --tools "$TOOLS" --case "$CASE" --atoms 128 \
    --result-set first_scale_qual_v15 --qualify --repeats 1 --stop-on-error \
    --scopes persistent final --variants first_pruned first_grouped
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_mixed_sweep.py" \
    --root "$ROOT" --tools "$TOOLS" --case "$CASE" --atoms 1024 \
    --result-set first_scale_screen_v15 --repeats 16 \
    --scopes persistent final --variants first_pruned first_grouped
done
printf 'DONE\n'
