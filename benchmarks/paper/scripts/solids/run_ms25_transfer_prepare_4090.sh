#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-ms25-20260818}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
ROOT="${ROOT:-/home/ylzhang/chorus_data/ms25_20260818}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

if [[ ! -f "${ROOT}/source/Datasets/MgO-4x4/set1/test.traj" ]]; then
  "${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/fetch_ms25_members.py" \
    --output-dir "${ROOT}/source" --set 1 --datasets mgo-4x4
fi
"${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/prepare_ms25.py" \
  --archive-root "${ROOT}/source" --output-root "${ROOT}/prepared" \
  --set 1 --transfer-only --num-workers 8
bash "${REPO}/benchmarks/paper/scripts/solids/run_ms25_deepmd_prepare_4090.sh"

touch "${ROOT}/MGO4X4_TRANSFER_DONE"
echo "MS25_MGO4X4_TRANSFER_PREPARE_DONE $(date -Is)"
