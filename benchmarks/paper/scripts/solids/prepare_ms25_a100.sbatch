#!/usr/bin/env bash
#SBATCH --job-name=ms25-prepare
# Data conversion is CPU-only.  Keeping it off the A100 partition avoids
# holding a scarce accelerator before any model training begins.
#SBATCH --partition=CPU-AMD
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm/prepare-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm/prepare-%j.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-ms25-20260818}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_solid_data/ms25_20260818}"

mkdir -p "${ROOT}" /public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

"${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/fetch_ms25_members.py" \
  --output-dir "${ROOT}/source" --set 1 \
  --datasets hea zr-o mgo-2x2 mgo-4x4

"${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/prepare_ms25.py" \
  --archive-root "${ROOT}/source" \
  --output-root "${ROOT}/prepared" \
  --set 1 --datasets hea zr-o mgo-2x2 --num-workers 8

touch "${ROOT}/DONE"
echo "MS25_A100_PREPARE_DONE $(date -Is)"
