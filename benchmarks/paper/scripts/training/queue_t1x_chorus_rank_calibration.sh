#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x/train_only_energy_calibration_by_rank}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"
CALIBRATE="${REPO}/benchmarks/paper/scripts/training/calibrate_t1x_large_mae_checkpoints.sh"

RANK8_SOURCE="${RANK8_SOURCE:-/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x/chorus_c128_l2_corr3_rank8_mae_ckpts/checkpoints/t1x_c128_l2_corr3_rank8_mae_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs32.e20s65604.pth}"
RANK32_SOURCE="${RANK32_SOURCE:-/home/ylzhang/chorus_runs/large_scale_main_20260724/rank32_pilot/t1x_c128_l2_corr3_rank32/checkpoints/t1x_c128_l2_corr3_rank32_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs32.e20s65604.pth}"

mkdir -p "${ROOT}/driver_logs"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

run_rank() {
  local rank="$1"
  local source="$2"
  local out="${ROOT}/rank${rank}"
  if [[ -f "${out}/ALL_DONE" ]]; then
    mark "SKIP_rank${rank}_already_done"
    return
  fi
  mark "START_rank${rank}"
  if env \
    REPO="${REPO}" \
    OUT="${out}" \
    RUN_BASELINE=0 \
    RUN_CHORUS=1 \
    CHORUS_NAME="chorus_rank${rank}" \
    PHASE_DENSITY_RANK="${rank}" \
    CHORUS_SOURCE="${source}" \
    bash "${CALIBRATE}" \
    >"${ROOT}/driver_logs/rank${rank}.log" 2>&1; then
    mark "DONE_rank${rank}"
  else
    local exit_code=$?
    mark "FAILED_rank${rank}_exit${exit_code}"
  fi
}

run_rank 8 "${RANK8_SOURCE}"
run_rank 32 "${RANK32_SOURCE}"
mark "ALL_CHORUS_RANK_CALIBRATIONS_FINISHED"
