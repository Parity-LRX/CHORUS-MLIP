#!/usr/bin/env bash
#SBATCH --job-name=chorus-phase-kimi
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/supervisor/supervisor-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/supervisor/supervisor-%j.err

set -euo pipefail
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-phase-causal-20260825}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/chorus_phase_causal_20260825}"
mkdir -p "${CAMPAIGN_ROOT}/supervisor"
exec env REPO="${REPO}" CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" POLL_SECONDS=300 \
  bash "${REPO}/benchmarks/causal/phase_specific_20260825/kimi_event_supervisor.sh"
