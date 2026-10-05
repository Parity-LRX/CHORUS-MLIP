#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ylzhang/lrx/chorus_transport_lips_water_20260922}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${LIPS_URL:-https://archive.materialscloud.org/records/ycbvx-knj69/files/lips.xyz?download=1}"
EXPECTED_MD5="${LIPS_MD5:-16f91b4aca0b5759026f8bd62790e5c6}"
OUTPUT="${ROOT}/download/lips.xyz"

mkdir -p "${ROOT}/download" "${ROOT}/logs"
if [[ ! -f "${OUTPUT}" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/download_resumable.py" \
    --url "${URL}" --output "${OUTPUT}.part"
  observed="$(md5sum "${OUTPUT}.part" | awk '{print $1}')"
  if [[ "${observed}" != "${EXPECTED_MD5}" ]]; then
    echo "LiPS MD5 mismatch: ${observed} != ${EXPECTED_MD5}" >&2
    exit 3
  fi
  mv "${OUTPUT}.part" "${OUTPUT}"
fi

observed="$(md5sum "${OUTPUT}" | awk '{print $1}')"
[[ "${observed}" == "${EXPECTED_MD5}" ]]
touch "${ROOT}/download/DONE"
echo "LIPS_DOWNLOAD_DONE bytes=$(stat -c %s "${OUTPUT}") $(date -Is)"
