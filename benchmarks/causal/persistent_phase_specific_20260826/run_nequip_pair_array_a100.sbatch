#!/usr/bin/env bash
#SBATCH --job-name=nequip-persist-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-19%1
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_persistent_phase_specific_20260826/slurm/%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_persistent_phase_specific_20260826/slurm/%A_%a.err

set -euo pipefail
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
ROOT="${ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_persistent_phase_specific_20260826}"
mkdir -p "${ROOT}/slurm"
exec env REPO="${REPO}" CAMPAIGN_ROOT="${ROOT}" \
  bash "${REPO}/benchmarks/causal/persistent_phase_specific_20260826/run_nequip_pair_one.sh" \
  "${SLURM_ARRAY_TASK_ID}"
