#!/usr/bin/env bash
#SBATCH --job-name=chorus-param-match
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/param_match_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v65"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
export SOURCE="$ROOT/source_v40/chorus"
export PYTHONPATH="$TOOLS:$OVERLAY:$SOURCE"
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:${LD_LIBRARY_PATH:-}"
for folder in "$OVERLAY"/nvidia/*/lib; do export LD_LIBRARY_PATH="$folder:$LD_LIBRARY_PATH"; done
export PYTHONPYCACHEPREFIX="$ROOT/cache/dpa/pycache" TMPDIR="$ROOT/cache/dpa/tmp"
export XDG_CACHE_HOME="$ROOT/cache/dpa/xdg" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/dpa/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/dpa/triton" CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python "$TOOLS/prepare_parameter_matched_dpa.py" \
  --root "$ROOT" --out "$ROOT/parameter_matched_v1"
printf 'DONE\n'
