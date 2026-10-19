#!/usr/bin/env bash
#SBATCH --job-name=chorus-memory-tile-sweep
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/tiles_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
SOURCE="${SOURCE:-$ROOT/source_v3/chorus}"
HP_SOURCE="${HP_SOURCE:-$ROOT/source_v3/ictc_o3_hp}"
CACHE=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache/native_v1_1
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
read -ra SIZES <<< "${ATOM_COUNTS:-10000 32768}"
read -ra TILES <<< "${TILE_COUNTS:-32768 65536 131072 262144}"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/run_memory_tile_sweep.py" --atoms "${SIZES[$SLURM_ARRAY_TASK_ID]}" \
  --root "$ROOT" --source "$SOURCE" --tiles "${TILES[@]}" \
  --experiment "${EXECUTION_EXPERIMENT:-none}" --tag "${RESULT_TAG:-tile_v2}"
printf 'DONE\n'
