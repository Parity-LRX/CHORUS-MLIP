#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ylzhang/lrx/chorus_reference_sc_validation_20261018
export PYTHONPATH="$ROOT/source/tools:$ROOT/source/chorus:$ROOT/source/ictc_o3_hp"
export XDG_CACHE_HOME="$ROOT/cache/xdg" TORCH_EXTENSIONS_DIR="$ROOT/cache/extensions"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/ictd" TRITON_CACHE_DIR="$ROOT/cache/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/pycache" TMPDIR="$ROOT/cache/tmp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver" CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
mkdir -p "$ROOT/cache/tmp" "$ROOT/logs" "$ROOT/results"
printf '%s\n' "$$" > "$ROOT/unit.pid"
date -Is
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
set -x
/home/ylzhang/micromamba/envs/FSCETP/bin/python -m pytest -q \
  "$ROOT/source/tools/test_checkpoint_reference_tp.py" --maxfail=1 \
  --junitxml="$ROOT/results/reference_sc_unit.xml"
printf 'DONE\n'
