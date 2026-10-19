#!/usr/bin/env bash
#SBATCH --job-name=chorus-public-cached-gates
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/public_cached_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
SOURCE="${SOURCE:?immutable source required}"
HP_SOURCE="${HP_SOURCE:?immutable mother source required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
SIZES=(10000 32768 65536)
FLAGS=(--experiments public_cached --dpa-reference-atoms 1024)
if [[ "$SLURM_ARRAY_TASK_ID" == 2 ]]; then
  FLAGS=(--experiments public_cached public_destination --skip-dpa)
fi
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/run_capacity_pair.py" --root "$ROOT" --source "$SOURCE" --hp-source "$HP_SOURCE" \
  --scope persistent --atoms "${SIZES[$SLURM_ARRAY_TASK_ID]}" \
  --tiles 65536 262144 --tag "${RESULT_TAG:-cached_gates_public_v16}" "${FLAGS[@]}"
printf 'DONE\n'
