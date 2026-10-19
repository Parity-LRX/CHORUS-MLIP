#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ylzhang/lrx/chorus_o3_hp_20261017
SOURCE="$ROOT/source_v2/chorus"
cd "$ROOT"
printf '%s\n' "$$" > "$ROOT/run_4090.pid"
export PYTHONPATH="$SOURCE:$ROOT/source/ictc_o3_hp"
export TORCH_CUDA_ARCH_LIST=8.9
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 MAX_JOBS=4
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/ictd"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/extensions"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/pycache"
export TMPDIR="$ROOT/cache/tmp"
mkdir -p "$TMPDIR"
PYTHON=/home/ylzhang/micromamba/envs/FSCETP/bin/python
set -x
"$PYTHON" -m pytest -q "$SOURCE/chorus/test/test_hp_backend.py" --tb=short
"$PYTHON" "$SOURCE/benchmarks/hp/qualify_and_bench.py" \
  --channels 64 --lmax 2 --scope persistent --out results/4090_persistent_c64_l2.json
"$PYTHON" "$SOURCE/benchmarks/hp/qualify_and_bench.py" \
  --channels 64 --lmax 2 --scope final --out results/4090_final_c64_l2.json
printf 'DONE\n'
