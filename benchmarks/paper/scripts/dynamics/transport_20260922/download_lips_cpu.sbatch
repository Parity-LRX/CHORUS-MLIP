#!/usr/bin/env bash
#SBATCH --job-name=transport-download
#SBATCH --partition=CPU-AMD
#SBATCH --nodelist=XHPC-9654-01
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/download_%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/download_%j.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
URL="${LIPS_URL:-https://archive.materialscloud.org/records/ycbvx-knj69/files/lips.xyz?download=1}"
EXPECTED_MD5="${LIPS_MD5:-16f91b4aca0b5759026f8bd62790e5c6}"
OUTPUT="${ROOT}/data/raw/lips.xyz"

mkdir -p "${ROOT}/data/raw" "${ROOT}/slurm"
if [[ ! -f "${OUTPUT}" ]]; then
  "${PYTHON_BIN}" \
    "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/download_resumable.py" \
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
touch "${ROOT}/DOWNLOAD_READY"
echo "LIPS_DOWNLOAD_DONE bytes=$(stat -c %s "${OUTPUT}") $(date -Is)"
