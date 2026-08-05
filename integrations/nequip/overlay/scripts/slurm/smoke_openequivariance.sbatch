#!/usr/bin/env bash
#SBATCH --job-name=smoke_nequip_oeq
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/smoke-oeq-%j.log

set -euo pipefail

REPO="/public/home/sps-xia/rxlin/NequIP-CHORUS"
OPERATOR_REPO="/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730"
PYTHON="/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python"
OEQ_SITE="/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages"

test -f "/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/READY"
export PYTHONPATH="${OEQ_SITE}:${REPO}:${OPERATOR_REPO}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=8
cd "${REPO}"

"${PYTHON}" scripts/smoke_openequivariance_parity.py
