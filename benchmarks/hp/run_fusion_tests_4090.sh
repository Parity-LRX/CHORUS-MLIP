#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ylzhang/lrx/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set immutable CHORUS snapshot}"
HP_SOURCE="${HP_SOURCE:?set immutable ICTC-O3-HP snapshot}"
cd "$ROOT"
printf '%s\n' "$$" > "$ROOT/fusion_tests.pid"
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$SOURCE:$HP_SOURCE"
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
  "$SOURCE/benchmarks/hp/test_coherent.py" \
  "$SOURCE/benchmarks/hp/test_coherent_training.py" \
  "$SOURCE/benchmarks/hp/test_radial_coherent.py" \
  "$SOURCE/benchmarks/hp/test_radial_jvp.py" \
  "$SOURCE/benchmarks/hp/test_radial_trunk.py" \
  "$SOURCE/benchmarks/hp/test_terminal_projection.py" \
  "$SOURCE/benchmarks/hp/test_species_gemm.py" \
  "$SOURCE/benchmarks/hp/test_capture.py" \
  "$SOURCE/benchmarks/hp/test_native_checkpoint.py" \
  "$SOURCE/benchmarks/hp/test_phase_coefficients.py" \
  "$SOURCE/benchmarks/hp/test_phase_trunk.py" \
  "$SOURCE/benchmarks/hp/test_layout_pruning.py" \
  "$SOURCE/benchmarks/hp/test_stable_reductions.py" \
  "$SOURCE/benchmarks/hp/test_affine_activation.py" \
  "$SOURCE/benchmarks/hp/test_phase_network.py" \
  "$SOURCE/benchmarks/hp/test_edge_geometry.py" \
  "$SOURCE/chorus/test/test_hp_backend.py" \
  "$SOURCE/benchmarks/hp/test_extended.py" \
  "$SOURCE/chorus/test/test_phase_hermitian.py::test_hermitian_pair_formula_and_global_u1_invariance" \
  "$SOURCE/chorus/test/test_phase_hermitian.py::test_full_l_low_rank_density_u1_invariance_and_shapes" \
  --tb=short -x ${PYTEST_EXTRA:-}
printf 'DONE\n'
