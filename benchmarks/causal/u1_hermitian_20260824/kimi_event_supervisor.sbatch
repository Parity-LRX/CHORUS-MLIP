#!/usr/bin/env bash
#SBATCH --job-name=chorus-kimi-supervisor
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/supervisor/supervisor-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/supervisor/supervisor-%j.err

set -euo pipefail
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-u1-causal-20260824}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824}"
mkdir -p "${CAMPAIGN_ROOT}/supervisor"
exec env REPO="${REPO}" CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  POLL_SECONDS=300 ARRAY_THROTTLE=4 \
  bash "${REPO}/benchmarks/causal/u1_hermitian_20260824/kimi_event_supervisor.sh"
