#!/usr/bin/env bash
#SBATCH --job-name=nequip_chorus_setup
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/setup-%j.log

set -euo pipefail

BASE=/public/home/sps-xia/rxlin/.conda/envs/mff
REPO=/public/home/sps-xia/rxlin/NequIP-CHORUS
VENV=/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff

mkdir -p /public/home/sps-xia/rxlin/nequip_chorus_runs
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BASE}/bin/python" -m venv --system-site-packages "${VENV}"
fi
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install \
  "torch-runstats>=0.2.0" "torch-ema>=0.3.0" pytest
"${VENV}/bin/python" -m pip install --no-deps --editable "${REPO}"

"${VENV}/bin/python" - <<'PY'
import e3nn
import nequip
import torch
import torch_ema
import torch_runstats

print("torch", torch.__version__)
print("e3nn", e3nn.__version__)
print("nequip", nequip.__version__)
print("torch_runstats", torch_runstats.__file__)
print("torch_ema", torch_ema.__file__)
PY
