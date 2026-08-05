#!/usr/bin/env bash
set -euo pipefail

SCALE_PID="${SCALE_PID:-2240863}"
ORIGINAL_SCREEN_PID="${ORIGINAL_SCREEN_PID:-2240850}"
ROOT="/home/ylzhang/chorus_runs/dia_rank32_stability_controls_20260727"
DRIVER="/home/ylzhang/CHORUS-MLIP-attention-test/benchmarks/paper/scripts/training/run_dia_rank32_stability_controls.sh"

while kill -0 "${SCALE_PID}" 2>/dev/null; do
  sleep 20
done

scale_log="${ROOT}/rank32_scale0025/logs/train.log"
if ! grep -q "done. best loss" "${scale_log}"; then
  echo "SCALE_TRAIN_FAILED"
  exit 2
fi
touch "${ROOT}/rank32_scale0025/DONE"

screen_pgid="$(ps -o pgid= -p "${ORIGINAL_SCREEN_PID}" | tr -d " ")"
if [[ -n "${screen_pgid}" ]]; then
  kill -TERM -"${screen_pgid}" 2>/dev/null || true
fi

orth_eval_done="${ROOT}/rank32_orthogonal1e3/validation_force_mae_selected_eval/DONE"
while [[ ! -f "${orth_eval_done}" ]]; do
  sleep 20
done

CONTROL_FILTER=all "${DRIVER}"
