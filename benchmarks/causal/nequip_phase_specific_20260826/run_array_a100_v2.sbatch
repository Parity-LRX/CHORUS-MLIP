#!/usr/bin/env bash
#SBATCH --job-name=nequip-causal-v2
#SBATCH --partition=GPU-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-79%4
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826/slurm/%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826/slurm/%A_%a.err

set -euo pipefail

export CAMPAIGN_ROOT=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826
CHORUS_REPO="${CHORUS_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
bash "${CHORUS_REPO}/benchmarks/causal/nequip_phase_specific_20260826/run_one.sh" \
  "${SLURM_ARRAY_TASK_ID}"
