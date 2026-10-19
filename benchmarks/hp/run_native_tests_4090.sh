#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ylzhang/lrx/chorus_hp_throughput_20261017
cd "$ROOT"
printf '%s\n' "$$" > "$ROOT/native_tests.pid"
export CUDA_VISIBLE_DEVICES=0
SOURCE="${SOURCE:-$ROOT/source/chorus}"
HP_SOURCE="${HP_SOURCE:-$ROOT/source/ictc_o3_hp}"
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
export CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XDG_CACHE_HOME="$ROOT/cache/native/xdg"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/native/extensions"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/native/ictd"
export TRITON_CACHE_DIR="$ROOT/cache/native/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
export TMPDIR="$ROOT/cache/native/tmp"
mkdir -p "$TMPDIR"
set -x
/home/ylzhang/micromamba/envs/FSCETP/bin/python -m pytest -q \
  "$SOURCE/test_coherent.py" "$SOURCE/test_capture.py" \
  "$SOURCE/test_native_checkpoint.py" "$SOURCE/test_phase_coefficients.py" \
  "$SOURCE/chorus/test/test_hp_backend.py" "$SOURCE/test_extended.py" \
  "$SOURCE/chorus/test/test_phase_hermitian.py::test_hermitian_pair_formula_and_global_u1_invariance" \
  "$SOURCE/chorus/test/test_phase_hermitian.py::test_full_l_low_rank_density_u1_invariance_and_shapes" --tb=short -x ${PYTEST_EXTRA:-}
printf 'DONE\n'
