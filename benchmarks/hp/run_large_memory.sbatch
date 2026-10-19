#!/usr/bin/env bash
#SBATCH --job-name=chorus-large-memory
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/large_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
SOURCE="$ROOT/source_v2/chorus"
HP_SOURCE="$ROOT/source_v2/ictc_o3_hp"
CACHE=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache/native_v1_1
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/run_large_memory_case.py" --index "$SLURM_ARRAY_TASK_ID" \
  --root "$ROOT" --source "$SOURCE" --hp-source "$HP_SOURCE"
printf 'DONE\n'
