#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-ms25-20260818}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
ROOT="${ROOT:-/home/ylzhang/chorus_data/ms25_20260818}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/source}"
PREPARED_ROOT="${PREPARED_ROOT:-${ROOT}/prepared}"
LOG="${LOG:-${ROOT}/prepare.log}"

mkdir -p "${ROOT}"
exec > >(tee -a "${LOG}") 2>&1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "PREPARE_START $(date -Is)"
"${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/fetch_ms25_members.py" \
  --output-dir "${SOURCE_ROOT}" \
  --set 1 \
  --datasets hea zr-o mgo-2x2 mgo-4x4

"${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/prepare_ms25.py" \
  --archive-root "${SOURCE_ROOT}" \
  --output-root "${PREPARED_ROOT}" \
  --set 1 \
  --datasets hea zr-o mgo-2x2 \
  --num-workers 8

touch "${ROOT}/DONE"
echo "PREPARE_DONE $(date -Is)"
