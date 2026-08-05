#!/usr/bin/env bash
#SBATCH --job-name=nequip_d2_count
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2-width-count-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_scope_c64_r16_20260801}"
OUTPUT="${OUTPUT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_width_parameter_sweep_20260803.json}"

export PYTHONPATH="${REPO}:${OPERATOR_REPO}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0

"${PYTHON}" "${REPO}/scripts/count_depth2_widths.py" \
  --two-species-config "${RUN_ROOT}/md22_buckyball/e3nn_chorus_off/run/config.yaml" \
  --four-species-config "${RUN_ROOT}/transition1x/e3nn_chorus_off/run/config.yaml" \
  --widths 64 68 72 76 80 84 88 92 96 \
  --output "${OUTPUT}"
