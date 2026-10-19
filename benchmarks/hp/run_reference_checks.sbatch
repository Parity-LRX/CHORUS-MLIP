#!/usr/bin/env bash
#SBATCH --job-name=chorus-halo-reference-test
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/halo_reference_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set source snapshot}"
HP_SOURCE="${HP_SOURCE:?set ICTC snapshot}"
export PYTHONPATH="$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
CACHE="$ROOT/cache/native_v1_1"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  "${TEST_SOURCE:-$SOURCE}/benchmarks/hp/test_blocked_reference.py" --tb=short -x
printf 'DONE\n'
