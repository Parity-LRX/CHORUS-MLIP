#!/usr/bin/env bash
#SBATCH --job-name=chorus-u1-causal
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-79%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/slurm/%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/slurm/%A_%a.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-u1-causal-20260824}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/runs}"
task_id="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"

# Adjacent task IDs are the exact Full/Charge2 pair, so the two permitted A100s
# evaluate the primary intervention under the same dataset and seed first.
modes=(full charge2 diagonal off)
datasets=(mal bucky 3bpa zro)
seeds=(20260616 20260617 20260618 20260619 20260620)

mode="${modes[$((task_id % 4))]}"
dataset="${datasets[$(((task_id / 4) % 4))]}"
seed="${seeds[$((task_id / 16))]}"

mkdir -p "${RUN_ROOT}" /public/home/sps-xia/rxlin/chorus_u1_causal_20260824/slurm
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

env REPO="${REPO}" RUN_ROOT="${RUN_ROOT}" DATASET="${dataset}" \
  MODE="${mode}" SEED="${seed}" \
  bash "${REPO}/benchmarks/causal/u1_hermitian_20260824/run_one.sh"
