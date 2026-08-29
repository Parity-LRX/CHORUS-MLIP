#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-ms25-20260818}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/chorus_data/ms25_20260818/prepared}"
RUN_ROOT="${RUN_ROOT:-/home/ylzhang/chorus_runs/ms25_external_20260818/dpa4}"
SEED="${SEED:-20260616}"
MAX_STEPS="${MAX_STEPS:-45000}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "${RUN_ROOT}/driver_logs"
status="${RUN_ROOT}/status.log"
failures=0

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${status}"
}

run_stage() {
  local name="$1"
  shift
  mark "START_${name}"
  if "$@" >"${RUN_ROOT}/driver_logs/${name}.log" 2>&1; then
    mark "DONE_${name}"
  else
    local code=$?
    failures=$((failures + 1))
    mark "FAILED_${name}_exit${code}"
  fi
}

while [[ ! -f /home/ylzhang/chorus_data/ms25_20260818/DEEPMD_DONE ]]; do
  sleep 10
done

for dataset in hea zr-o mgo-2x2; do
  data="${DATA_ROOT}/${dataset}/set1/deepmd"
  cutoff="$("${DPA_ENV}/bin/python" - "${DATA_ROOT}/${dataset}/set1/metadata.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["cutoff_angstrom"])
PY
)"
  for channels in 32 48; do
    out="${RUN_ROOT}/${dataset}/c${channels}_mix3_fp32_compiled"
    run_stage "${dataset}_c${channels}_train" env \
      DATA="${data}" OUT="${out}" STEPS="${MAX_STEPS}" BATCH_SIZE="${BATCH_SIZE}" \
      CHANNELS="${channels}" MIXING_LAYERS=3 RCUT="${cutoff}" \
      USE_COMPILE=1 USE_AMP=0 SEED="${SEED}" REPO="${REPO}" DPA_ENV="${DPA_ENV}" \
      bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_fixed_xxmd.sh"
    if [[ -f "${out}/DONE" ]]; then
      extra_test=()
      if [[ "${dataset}" == "mgo-2x2" ]]; then
        while [[ ! -f /home/ylzhang/chorus_data/ms25_20260818/MGO4X4_TRANSFER_DONE ]]; do
          sleep 10
        done
        extra_test=(
          --extra-test-system
          "mgo_4x4_size_transfer=${DATA_ROOT}/mgo-4x4-transfer/set1/deepmd/test"
        )
      fi
      run_stage "${dataset}_c${channels}_select_test" \
        "${DPA_ENV}/bin/python" \
        "${REPO}/benchmarks/paper/scripts/training/evaluate_dpa4_xxmd_force_mae.py" \
        --run "${out}" --data "${data}" \
        --out "${out}/full_val_force_mae_eval" --steps "${MAX_STEPS}" \
        --checkpoint-every 1 --dp "${DPA_ENV}/bin/dp" \
        "${extra_test[@]}"
    fi
  done
done

if (( failures == 0 )); then
  touch "${RUN_ROOT}/DONE"
  mark "ALL_MS25_DPA4_DONE"
else
  touch "${RUN_ROOT}/COMPLETED_WITH_FAILURES"
  mark "ALL_MS25_DPA4_ATTEMPTED_failures${failures}"
  exit 1
fi
