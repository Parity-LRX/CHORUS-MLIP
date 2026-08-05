#!/usr/bin/env bash
#SBATCH --job-name=setup_nequip_oeq
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/setup-oeq-%j.log

set -euo pipefail

BASE_PYTHON="/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python"
VENV="/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV}"
fi
"${VENV}/bin/python" -m pip install --upgrade "openequivariance>=0.3.0"
"${VENV}/bin/python" - <<'PY'
import openequivariance
import torch

assert torch.cuda.is_available()
print("torch", torch.__version__)
print("openequivariance", getattr(openequivariance, "__version__", "unknown"))
print("gpu", torch.cuda.get_device_name(0))
PY
touch "${VENV}/READY"
