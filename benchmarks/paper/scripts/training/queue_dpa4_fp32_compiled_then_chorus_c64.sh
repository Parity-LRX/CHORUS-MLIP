#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/dpa4_fp32_compiled_all_20260728}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

run_dpa_capacity() {
  local channels="$1"
  local out="${ROOT}/c${channels}_mix3"
  if [[ -f "${out}/DONE" ]]; then
    mark "SKIP_DPA4_C${channels}_DONE"
    return
  fi
  mark "START_DPA4_C${channels}_STRICT_FP32_COMPILED"
  env \
    REPO="${REPO}" ROOT="${out}" STATUS="${out}/queue_status.log" \
    CHANNELS="${channels}" MIXING_LAYERS=3 \
    USE_COMPILE=1 USE_AMP=0 \
    WAIT_SCREENS="" XXMD_SYSTEMS="mal sti" \
    bash "${REPO}/benchmarks/paper/scripts/training/queue_dpa4_c48_scaling.sh"
  mark "DONE_DPA4_C${channels}_STRICT_FP32_COMPILED"
}

run_dpa_capacity 32
run_dpa_capacity 48

mark "START_OR_RESUME_CHORUS_C64_RANK16"
bash "${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"
mark "DONE_CHORUS_C64_RANK16"

touch "${ROOT}/ALL_DONE"
mark "ALL_FP32_COMPILED_AND_C64_DONE"
