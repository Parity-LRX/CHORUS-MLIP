#!/usr/bin/env bash
#SBATCH --job-name=lips-reference-distances
#SBATCH --partition=CPU-AMD
#SBATCH --nodelist=XHPC-7B13-01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/lips_reference_distances_%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/lips_reference_distances_%j.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${ROOT}/source/CHORUS-MLIP"
PYTHON_BIN="/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python"
OUT="${ROOT}/diagnostics/lips_instability_20260926/reference_distances.json"

"${PYTHON_BIN}" \
  "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/summarize_lips_reference_distances.py" \
  --data-root "${ROOT}/data/prepared/lips/extxyz" --output "${OUT}"

echo "LIPS_REFERENCE_DISTANCES_DONE ${OUT} $(date -Is)"
