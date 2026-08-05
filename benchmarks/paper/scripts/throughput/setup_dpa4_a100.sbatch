#!/usr/bin/env bash
#SBATCH --job-name=dpa4_a100_setup
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs/dpa4-setup-%j.log

set -euo pipefail

BASE_PY="${BASE_PY:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
ASSETS="${ASSETS:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804}"
VENV="${VENV:-/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling}"
SOURCE="${SOURCE:-${ASSETS}/deepmd-kit-dpa4-master-source}"

mkdir -p /public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BASE_PY}" -m venv --system-site-packages "${VENV}"
fi
"${VENV}/bin/python" -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  'scikit-build-core>=1' dependency_groups ninja cmake \
  setuptools-scm hatch-fancy-pypi-readme 'vesin[torch]' 'e3nn>=0.5.9'
export DP_ENABLE_PYTORCH=1
export DP_ENABLE_TENSORFLOW=0
export DP_VARIANT="${DP_VARIANT:-cpu}"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-8}"
rm -rf "${SOURCE}/build"
"${VENV}/bin/python" -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-build-isolation --no-deps --force-reinstall "${SOURCE}"
"${VENV}/bin/python" - <<'PY'
import torch
import deepmd
from deepmd.pt.model.model import get_model

print("torch", torch.__version__)
print("deepmd", deepmd.__version__)
print("get_model", get_model)
PY
touch "${VENV}/READY"
