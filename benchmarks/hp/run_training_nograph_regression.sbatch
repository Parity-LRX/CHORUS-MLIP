#!/usr/bin/env bash
#SBATCH --job-name=chorus-nograph-regression
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/regression_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
export SOURCE="$ROOT/source_v5/chorus" HP_SOURCE="$ROOT/source_v5/ictc_o3_hp"
CACHE="$ROOT/cache/native"
export PYTHONPATH="$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
cd "$SOURCE"
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q benchmarks/hp/test_*.py \
  chorus/test/test_hp_backend.py \
  chorus/test/test_phase_hermitian.py::test_hermitian_pair_formula_and_global_u1_invariance \
  chorus/test/test_phase_hermitian.py::test_full_l_low_rank_density_u1_invariance_and_shapes \
  --maxfail=1 --junitxml="$ROOT/results/full_regression_${SLURM_JOB_ID}.xml"
printf 'DONE\n'
