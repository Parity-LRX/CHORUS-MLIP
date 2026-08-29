#!/usr/bin/env bash
#SBATCH --job-name=chorus-phase-aligned
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/slurm/aligned_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/slurm/aligned_%A_%a.err

set -euo pipefail
datasets=(mal bucky 3bpa zro)
seeds=(20260616 20260617 20260618 20260619 20260620)
dataset="${datasets[$((SLURM_ARRAY_TASK_ID / 5))]}"
seed="${seeds[$((SLURM_ARRAY_TASK_ID % 5))]}"
exec env DATASET="${dataset}" SEED="${seed}" REPO="${REPO}" \
  RUN_ROOT="${ALIGNED_ROOT}" PRIMARY_ROOT="${PRIMARY_ROOT}" MAX_STEPS=45000 \
  bash "${REPO}/benchmarks/causal/phase_specific_20260825/run_aligned_one.sh"
