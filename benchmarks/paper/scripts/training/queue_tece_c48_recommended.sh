#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
WAIT_SCREENS="${WAIT_SCREENS:-large_model_missing_queue}"
WAIT_FILE="${WAIT_FILE:-}"
SYSTEMS="${SYSTEMS:-mal dia sti t1x bucky}"
COMPLETION_FILE="${COMPLETION_FILE:-}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/tece_c48_recommended_20260725}"
XXMD_RUN_ROOT="${XXMD_RUN_ROOT:-/home/ylzhang/tace_chorus_runs/xxmd_tece_c48_seed20260616}"
OTHER_RUN_ROOT="${OTHER_RUN_ROOT:-/home/ylzhang/tace_chorus_runs/large_tece_c48_seed20260616}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"
MAX_BACKGROUND_GPU_MEMORY_MB="${MAX_BACKGROUND_GPU_MEMORY_MB:-4096}"
SEED="${SEED:-20260616}"
CHANNELS="${CHANNELS:-48}"

mkdir -p "${ROOT}/driver_logs"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

gpu_is_busy() {
  local used_mb
  used_mb="$(
    nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk '{sum += $1} END {print sum + 0}'
  )"
  (( used_mb > MAX_BACKGROUND_GPU_MEMORY_MB ))
}

wait_for_gpu() {
  mark "WAIT_GPU_MEMORY_LE_${MAX_BACKGROUND_GPU_MEMORY_MB}MB"
  while gpu_is_busy; do
    sleep 30
  done
  mark "GPU_MEMORY_THRESHOLD_READY"
}

stage_failures=0
run_stage() {
  local name="$1"
  shift
  wait_for_gpu
  mark "START_${name}"
  if "$@" >"${ROOT}/driver_logs/${name}.log" 2>&1; then
    mark "DONE_${name}"
  else
    local exit_code=$?
    stage_failures=$((stage_failures + 1))
    mark "FAILED_${name}_exit${exit_code}"
  fi
}

for wait_screen in ${WAIT_SCREENS}; do
  mark "WAIT_SCREEN_${wait_screen}"
  while screen_exists "${wait_screen}"; do
    sleep 30
  done
done

if [[ -n "${WAIT_FILE}" ]]; then
  mark "WAIT_FILE_${WAIT_FILE}"
  while [[ ! -f "${WAIT_FILE}" ]]; do
    sleep 30
  done
  mark "WAIT_FILE_READY_${WAIT_FILE}"
fi

for system in ${SYSTEMS}; do
  case "${system}" in
    mal|dia|sti)
      run_stage "tece_c48_${system}" \
        env SYSTEMS="${system}" CHANNELS="${CHANNELS}" SEED="${SEED}" \
        RUN_ROOT="${XXMD_RUN_ROOT}" \
        bash "${REPO}/benchmarks/paper/external/tece/run_tece_xxmd_queue.sh"
      ;;
    t1x|bucky)
      run_stage "tece_c48_${system}" \
        env SYSTEMS="${system}" CHANNELS="${CHANNELS}" SEED="${SEED}" \
        RUN_ROOT="${OTHER_RUN_ROOT}" \
        bash "${REPO}/benchmarks/paper/external/tece/run_tece_large_missing_queue.sh"
      ;;
    *)
      mark "FAILED_unknown_system_${system}"
      stage_failures=$((stage_failures + 1))
      ;;
  esac
done

if (( stage_failures == 0 )); then
  touch "${ROOT}/DONE"
  if [[ -n "${COMPLETION_FILE}" ]]; then
    mkdir -p "$(dirname "${COMPLETION_FILE}")"
    touch "${COMPLETION_FILE}"
  fi
  mark "ALL_TECE_C48_STAGES_DONE systems=${SYSTEMS}"
else
  touch "${ROOT}/COMPLETED_WITH_FAILURES"
  mark "ALL_TECE_C48_STAGES_ATTEMPTED failures=${stage_failures}"
  exit 1
fi
