#!/usr/bin/env bash
#SBATCH --job-name=chorus-virial-diagnostic
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/virial_diagnostic_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="$ROOT/native_v8/chorus"
export PYTHONPATH="$SOURCE:$ROOT/native_v8/ictc_o3_hp"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XDG_CACHE_HOME="$ROOT/cache/native_v1_1/xdg"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/native_v1_1/extensions"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/native_v1_1/ictd"
export TRITON_CACHE_DIR="$ROOT/cache/native_v1_1/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native_v1_1/pycache"
export TMPDIR="$ROOT/cache/native_v1_1/tmp"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$SOURCE/benchmarks/hp/diagnose_virial.py"
printf 'DONE\n'
