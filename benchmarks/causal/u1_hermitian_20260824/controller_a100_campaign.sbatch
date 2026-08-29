#!/usr/bin/env bash
#SBATCH --job-name=chorus-u1-controller
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/controller/controller-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/controller/controller-%j.err

set -euo pipefail
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-u1-causal-20260824}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824}"
env REPO="${REPO}" CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  bash "${REPO}/benchmarks/causal/u1_hermitian_20260824/controller.sh"
