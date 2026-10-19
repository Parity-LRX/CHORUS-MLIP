#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-nograph-tests
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/tests_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
CANDIDATE="${CANDIDATE:-source_v5}"
export SOURCE="$ROOT/$CANDIDATE/chorus" HP_SOURCE="$ROOT/$CANDIDATE/ictc_o3_hp"
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
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  benchmarks/hp/test_phase_trunk.py benchmarks/hp/test_no_cuda_graph.py \
  benchmarks/hp/test_coherent_training.py chorus/test/test_hp_backend.py \
  --maxfail=1 --junitxml="$ROOT/results/training_nograph_tests_${SLURM_JOB_ID}.xml"
printf 'DONE\n'
