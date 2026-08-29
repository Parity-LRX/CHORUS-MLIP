#!/usr/bin/env bash
#SBATCH --job-name=phase-observe
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19%4
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/slurm/observe_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/slurm/observe_%A_%a.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-phase-causal-20260825}"
OBS_ROOT="${OBS_ROOT:-/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/observational_phase}"
datasets=(mal bucky 3bpa zro)
seeds=(20260616 20260617 20260618 20260619 20260620)
dataset_index=$((SLURM_ARRAY_TASK_ID / 5))
seed_index=$((SLURM_ARRAY_TASK_ID % 5))

mkdir -p "${OBS_ROOT}" /public/home/sps-xia/rxlin/chorus_phase_causal_20260825/slurm
env REPO="${REPO}" OBS_ROOT="${OBS_ROOT}" \
  DATASET="${datasets[$dataset_index]}" SEED="${seeds[$seed_index]}" \
  bash "${REPO}/benchmarks/causal/phase_specific_20260825/run_observational_phase_one.sh"
