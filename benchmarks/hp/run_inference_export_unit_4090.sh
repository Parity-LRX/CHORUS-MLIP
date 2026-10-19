#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ylzhang/lrx/chorus_hp_inference_aoti_20261018
export CUDA_VISIBLE_DEVICES=0
export SOURCE="${SOURCE:-$ROOT/source_v1/chorus}" HP_SOURCE="${HP_SOURCE:-$ROOT/source_v1/ictc_o3_hp}"
export PYTHONPATH="$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export XDG_CACHE_HOME="$ROOT/cache/xdg" TORCH_EXTENSIONS_DIR="$ROOT/cache/extensions"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/ictd" TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" PYTHONPYCACHEPREFIX="$ROOT/cache/pycache"
export TMPDIR="$ROOT/cache/tmp" CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
cd "$ROOT"
printf '%s\n' "$$" > "$ROOT/prepared_export_unit.pid"
date -Is
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
set -x
if [[ "${DIAGNOSTIC:-0}" == 1 ]]; then
  /home/ylzhang/micromamba/envs/FSCETP/bin/python "$SOURCE/benchmarks/hp/diagnose_prepared_export.py"
else
  /home/ylzhang/micromamba/envs/FSCETP/bin/python -m pytest \
    "$SOURCE/chorus/test/test_hp_export.py" -q -x --junitxml="$ROOT/logs/${RUN:-prepared_export_unit}.xml"
fi
printf 'DONE\n'
