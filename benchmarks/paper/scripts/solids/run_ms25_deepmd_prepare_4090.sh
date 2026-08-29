#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-ms25-20260818}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
ROOT="${ROOT:-/home/ylzhang/chorus_data/ms25_20260818}"
PREPARED_ROOT="${PREPARED_ROOT:-${ROOT}/prepared}"

while [[ ! -f "${ROOT}/DONE" ]]; do
  sleep 10
done

transfer_source="${PREPARED_ROOT}/mgo-4x4-transfer/set1/chorus"
if [[ -f "${transfer_source}/processed_test.h5" ]]; then
  "${PYTHON_BIN}" \
    "${REPO}/benchmarks/paper/scripts/training/prepare_grouped_deepmd_npy_from_ictc_h5.py" \
    --source-dir "${transfer_source}" \
    --output-dir "${PREPARED_ROOT}/mgo-4x4-transfer/set1/deepmd" \
    --type-map 8,12 --splits test --periodic
fi

for dataset in hea zr-o mgo-2x2; do
  source_dir="${PREPARED_ROOT}/${dataset}/set1/chorus"
  output_dir="${PREPARED_ROOT}/${dataset}/set1/deepmd"
  type_map="$("${PYTHON_BIN}" - "${PREPARED_ROOT}/${dataset}/set1/metadata.json" <<'PY'
import json, sys
print(",".join(map(str, json.load(open(sys.argv[1]))["atomic_numbers"])))
PY
)"
  "${PYTHON_BIN}" \
    "${REPO}/benchmarks/paper/scripts/training/prepare_grouped_deepmd_npy_from_ictc_h5.py" \
    --source-dir "${source_dir}" \
    --output-dir "${output_dir}" \
    --type-map "${type_map}" \
    --splits train,val,test \
    --periodic
done

touch "${ROOT}/DEEPMD_DONE"
echo "MS25_DEEPMD_PREPARE_DONE $(date -Is)"
