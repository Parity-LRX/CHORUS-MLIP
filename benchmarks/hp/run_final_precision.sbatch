#!/usr/bin/env bash
#SBATCH --job-name=chorus-final-precision-controls
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/final_precision_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
SOURCE="$ROOT/source_v6/chorus"
HP_SOURCE="$ROOT/source_v6/ictc_o3_hp"
CACHE=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache/native_v1_1
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$ROOT/benchmark_tools_v1/run_large_memory_case.py" --index 1 --root "$ROOT" \
  --source "$SOURCE" --hp-source "$HP_SOURCE" --scope final \
  --tag final_precision_v5 --tile-edges 65536
printf 'DONE\n'
