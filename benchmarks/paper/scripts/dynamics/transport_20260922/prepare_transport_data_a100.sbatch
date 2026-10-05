#!/usr/bin/env bash
#SBATCH --job-name=transport-data
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/data_%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/data_%j.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
WATER_SOURCE="${WATER_SOURCE:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_ictd_public_water/cheng_water}"
LIPS_URL="${LIPS_URL:-https://archive.materialscloud.org/records/ycbvx-knj69/files/lips.xyz?download=1}"
LIPS_MD5="${LIPS_MD5:-16f91b4aca0b5759026f8bd62790e5c6}"
RAW_DIR="${ROOT}/data/raw"
PREPARED="${ROOT}/data/prepared"

mkdir -p "${ROOT}/slurm" "${RAW_DIR}" "${PREPARED}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

lips_source="${RAW_DIR}/lips.xyz"
if [[ ! -f "${lips_source}" ]]; then
  "${PYTHON_BIN}" \
    "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/download_resumable.py" \
    --url "${LIPS_URL}" --output "${lips_source}.part"
  mv "${lips_source}.part" "${lips_source}"
fi
observed_md5="$(md5sum "${lips_source}" | awk '{print $1}')"
if [[ "${observed_md5}" != "${LIPS_MD5}" ]]; then
  echo "LiPS MD5 mismatch: ${observed_md5} != ${LIPS_MD5}" >&2
  exit 3
fi

"${PYTHON_BIN}" \
  "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/prepare_transport_data.py" \
  --water-source "${WATER_SOURCE}" \
  --lips-source "${lips_source}" \
  --output-root "${PREPARED}" \
  --cutoff 5.0 --seed 20260616 --workers "${SLURM_CPUS_PER_TASK:-12}"

for dataset in water lips; do
  type_map="$(${PYTHON_BIN} - "${PREPARED}/${dataset}/metadata.json" <<'PY'
import json, sys
print(",".join(map(str, json.load(open(sys.argv[1]))["atomic_numbers"])))
PY
)"
  "${PYTHON_BIN}" \
    "${REPO}/benchmarks/paper/scripts/training/prepare_grouped_deepmd_npy_from_ictc_h5.py" \
    --source-dir "${PREPARED}/${dataset}/chorus" \
    --output-dir "${PREPARED}/${dataset}/deepmd" \
    --type-map "${type_map}" --splits train,val,test --periodic
done

touch "${ROOT}/DATA_READY"
echo "TRANSPORT_DATA_READY $(date -Is)"
