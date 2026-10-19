#!/usr/bin/env bash
#SBATCH --job-name=chorus-fusion-test
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/fusion_validation_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set immutable CHORUS snapshot}"
HP_SOURCE="${HP_SOURCE:?set immutable ICTC-O3-HP snapshot}"
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
CACHE="$ROOT/cache/native_v1_1"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
PY=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
set -x
"$PY" -m pytest -q \
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
  --tb=short -x
"$PY" "$SOURCE/benchmarks/hp/qualify_checkpoint.py" \
  --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
  --checkpoint-sc --symmetric-contraction on --frames 16 \
  ${RADIAL_FLAGS:-} \
  --data-h5 /public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728/transition1x/chorus_reaction_id_50k_seed20260616/processed_val.h5 \
  --out "$ROOT/results/${RESULT_SET:-fusion_validation_v15}/persistent_heldout.json"
printf 'DONE\n'
