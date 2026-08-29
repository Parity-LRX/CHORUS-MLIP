#!/usr/bin/env bash
#SBATCH --job-name=chorus-serverchan
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826/serverchan/notifier-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826/serverchan/notifier-%j.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826}"
mkdir -p "${CAMPAIGN_ROOT}/serverchan"
exec env REPO="${REPO}" CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" INTERVAL_SECONDS=3600 \
  bash "${REPO}/benchmarks/causal/nequip_phase_specific_20260826/serverchan_hourly_loop.sh"
