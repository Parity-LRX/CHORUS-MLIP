#!/usr/bin/env bash
#SBATCH --job-name=chorus-large-regression
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/regression_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
SOURCE="$ROOT/source_v9/chorus"
HP_SOURCE="$ROOT/source_v9/ictc_o3_hp"
CACHE=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache/native_v1_1
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$SOURCE"
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q benchmarks/hp/test_*.py \
  chorus/test/test_hp_backend.py \
  chorus/test/test_phase_hermitian.py::test_hermitian_pair_formula_and_global_u1_invariance \
  chorus/test/test_phase_hermitian.py::test_full_l_low_rank_density_u1_invariance_and_shapes \
  --maxfail=1 --junitxml="$ROOT/results/tests_complete_v7.xml"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python benchmarks/hp/qualify_checkpoint.py \
  --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
  --checkpoint-sc --symmetric-contraction on --frames 16 \
  --radial-jvp --radial-gate-pipeline --radial-destination --radial-tile-edges 65536 \
  --data-h5 /public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728/transition1x/chorus_reaction_id_50k_seed20260616/processed_val.h5 \
  --out "$ROOT/results/public_v7/persistent_heldout.json"
printf 'DONE\n'
