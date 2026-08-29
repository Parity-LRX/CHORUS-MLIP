#!/usr/bin/env bash
#SBATCH --job-name=chorus-kimi-operator
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826/kimi_supervisor/supervisor-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826/kimi_supervisor/supervisor-%j.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826}"
MACE_PERSISTENT_ROOT="${MACE_PERSISTENT_ROOT:-/public/home/sps-xia/rxlin/chorus_persistent_causal_20260826}"
NEQUIP_PERSISTENT_ROOT="${NEQUIP_PERSISTENT_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_persistent_phase_specific_20260826}"
mkdir -p "${CAMPAIGN_ROOT}/kimi_supervisor"
exec env REPO="${REPO}" CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  MACE_PERSISTENT_ROOT="${MACE_PERSISTENT_ROOT}" \
  NEQUIP_PERSISTENT_ROOT="${NEQUIP_PERSISTENT_ROOT}" \
  POLL_SECONDS=300 HEARTBEAT_SECONDS=21600 \
  bash "${REPO}/benchmarks/causal/nequip_phase_specific_20260826/kimi_cluster_supervisor.sh"
