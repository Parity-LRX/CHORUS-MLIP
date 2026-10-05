#!/usr/bin/env bash
#SBATCH --job-name=transport-prepare
#SBATCH --partition=CPU-AMD
#SBATCH --nodelist=XHPC-9654-01
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/prepare_%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/prepare_%j.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
WATER_SOURCE="${WATER_SOURCE:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_ictd_public_water/cheng_water}"
LIPS_SOURCE="${LIPS_SOURCE:-${ROOT}/data/raw/lips.xyz}"
PREPARED="${ROOT}/data/prepared"
LIPS_EXPECTED_SIZE="${LIPS_EXPECTED_SIZE:-222930996}"
LIPS_EXPECTED_MD5="${LIPS_EXPECTED_MD5:-16f91b4aca0b5759026f8bd62790e5c6}"

[[ -f "${LIPS_SOURCE}" ]]
actual_size="$(stat -c %s "${LIPS_SOURCE}")"
actual_md5="$(md5sum "${LIPS_SOURCE}" | awk '{print $1}')"
if [[ "${actual_size}" != "${LIPS_EXPECTED_SIZE}" ]]; then
  echo "LiPS size mismatch: ${actual_size} != ${LIPS_EXPECTED_SIZE}" >&2
  exit 1
fi
if [[ "${actual_md5}" != "${LIPS_EXPECTED_MD5}" ]]; then
  echo "LiPS MD5 mismatch: ${actual_md5} != ${LIPS_EXPECTED_MD5}" >&2
  exit 1
fi
touch "${ROOT}/DOWNLOAD_READY"
mkdir -p "${ROOT}/slurm" "${PREPARED}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON_BIN}" \
  "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/prepare_transport_data.py" \
  --water-source "${WATER_SOURCE}" --lips-source "${LIPS_SOURCE}" \
  --output-root "${PREPARED}" --cutoff 5.0 --seed 20260616 \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

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
