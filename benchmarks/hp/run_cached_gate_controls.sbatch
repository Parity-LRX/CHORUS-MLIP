#!/usr/bin/env bash
#SBATCH --job-name=chorus-cached-gate-controls
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/cached_gate_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
SOURCE="${SOURCE:?immutable source required}"
HP_SOURCE="${HP_SOURCE:?immutable mother source required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
SIZES=(10000 32768)
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/run_capacity_pair.py" --root "$ROOT" --source "$SOURCE" --hp-source "$HP_SOURCE" \
  --scope persistent --atoms "${SIZES[$SLURM_ARRAY_TASK_ID]}" --dpa-reference-atoms 1024 \
  --tiles 65536 262144 524288 --tag cached_gates_v12 \
  --experiments persistent_distance_cached persistent_distance_cached_deep
printf 'DONE\n'
