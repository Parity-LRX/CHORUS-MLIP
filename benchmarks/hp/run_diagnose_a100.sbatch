#!/usr/bin/env bash
#SBATCH --job-name=chorus-hp-diagnose
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_o3_hp_20261017/logs/diagnose_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_o3_hp_20261017
export PYTHONPATH="$ROOT/source_v3/chorus:$ROOT/source/ictc_o3_hp"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XDG_CACHE_HOME="$ROOT/cache/checkpoint_1/xdg"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/checkpoint_1/ictd"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/checkpoint_1/extensions"
export TRITON_CACHE_DIR="$ROOT/cache/checkpoint_1/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/checkpoint_1/pycache"
export TMPDIR="$ROOT/cache/checkpoint_1/tmp"
cd "$ROOT"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$ROOT/tools/diagnose_checkpoint.py" \
 --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
 --out "$ROOT/results/diagnose_persistent_checkpoint.json"
