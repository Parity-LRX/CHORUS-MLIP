#!/usr/bin/env bash
#SBATCH --job-name=chorus-radial-test
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/radial_validation_%j.log
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
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  "$SOURCE/benchmarks/hp/test_radial_jvp.py" \
  "$SOURCE/benchmarks/hp/test_radial_coherent.py" ${PYTEST_EXTRA:-} --tb=short -x
printf 'DONE\n'
