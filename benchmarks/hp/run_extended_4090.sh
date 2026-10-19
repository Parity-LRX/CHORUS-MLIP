#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ylzhang/lrx/chorus_o3_hp_20261017
SOURCE="$ROOT/source_v3/chorus"
cd "$ROOT"
printf '%s\n' "$$" > "$ROOT/extended_4090.pid"
export PYTHONPATH="$SOURCE:$ROOT/source/ictc_o3_hp"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9
export CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH
export OMP_NUM_THREADS=1 MAX_JOBS=4
export XDG_CACHE_HOME="$ROOT/cache/xdg" FSCETP_ICTD_CACHE_DIR="$ROOT/cache/ictd"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/extensions" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/triton" PYTHONPYCACHEPREFIX="$ROOT/cache/pycache"
export TMPDIR="$ROOT/cache/tmp"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
set -x
/home/ylzhang/micromamba/envs/FSCETP/bin/python -m pytest -q "$ROOT/tools/test_extended.py" \
  "$SOURCE/chorus/test/test_phase_hermitian.py::test_hermitian_pair_formula_and_global_u1_invariance" \
  "$SOURCE/chorus/test/test_phase_hermitian.py::test_full_l_low_rank_density_u1_invariance_and_shapes" \
  --tb=short -x
printf 'DONE\n'
