#!/usr/bin/env bash
#SBATCH --job-name=chorus-large-capacity-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/capacity_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
SOURCE="${SOURCE:?immutable CHORUS source required}"
HP_SOURCE="${HP_SOURCE:?immutable ICTC-O3-HP source required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
read -ra SCOPES <<< "${MODEL_SCOPES:-persistent final}"
read -ra SIZES <<< "${ATOM_COUNTS:-32768 32768}"
read -ra TILES <<< "${TILE_COUNTS:-65536 131072}"
FLAGS=()
if [[ -n "${DPA_REFERENCE_ATOMS:-}" ]]; then FLAGS+=(--dpa-reference-atoms "$DPA_REFERENCE_ATOMS"); fi
if [[ "${SKIP_DPA:-0}" == 1 ]]; then FLAGS+=(--skip-dpa); fi
if [[ -n "${EXECUTION_EXPERIMENTS:-}" ]]; then
  read -ra EXPERIMENTS <<< "$EXECUTION_EXPERIMENTS"
  FLAGS+=(--experiments "${EXPERIMENTS[@]}")
fi
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "${DRIVER:-$SOURCE/benchmarks/hp/run_capacity_pair.py}" --root "$ROOT" --source "$SOURCE" --hp-source "$HP_SOURCE" \
  --scope "${SCOPES[$SLURM_ARRAY_TASK_ID]}" --atoms "${SIZES[$SLURM_ARRAY_TASK_ID]}" \
  --tiles "${TILES[@]}" --tag "${RESULT_TAG:-capacity_v4}" \
  --experiment "${EXECUTION_EXPERIMENT:-persistent_distance_destination}" "${FLAGS[@]}"
printf 'DONE\n'
