#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-ms25-20260818}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/chorus_data/ms25_20260818/prepared/zr-o/set1}"
RUN_ROOT="${RUN_ROOT:-/home/ylzhang/chorus_runs/ms25_external_20260818}"
DPA_PYTHON="${DPA_PYTHON:-/home/ylzhang/venvs/dpa4-master/bin/python}"
TECE_PYTHON="${TECE_PYTHON:-/home/ylzhang/tace_chorus_venv/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-30}"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE:-512}"

run_dpa() {
  local channels="$1"
  local run="${RUN_ROOT}/dpa4/zr-o/c${channels}_mix3_fp32_compiled"
  local raw="${run}/full_val_force_mae_eval/metrics.json"
  local out="${run}/train_only_energy_calibration"
  if [[ -f "${out}/DONE" ]]; then
    echo "SKIP_DPA_C${channels} $(date -Is)"
    return
  fi
  [[ -f "${raw}" ]]
  mkdir -p "${out}"
  local checkpoint
  checkpoint="$("${DPA_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "${raw}")"
  echo "START_DPA_C${channels} checkpoint=${checkpoint} $(date -Is)"
  "${DPA_PYTHON}" "${REPO}/benchmarks/paper/scripts/training/calibrate_dpa4_t1x_energy.py" \
    --checkpoint "${checkpoint}" --data "${DATA_ROOT}/deepmd" \
    --elements O,Zr --out "${out}" \
    >"${out}/run.log" 2>&1
  echo "DONE_DPA_C${channels} $(date -Is)"
}

run_tece() {
  local channels="$1"
  local run="${RUN_ROOT}/tece/zr-o/c${channels}"
  local raw="${run}/metrics.json"
  local out="${run}/train_only_energy_calibration"
  if [[ -f "${out}/DONE" ]]; then
    echo "SKIP_TECE_C${channels} $(date -Is)"
    return
  fi
  while [[ ! -f "${run}/DONE" || ! -f "${raw}" ]]; do
    sleep "${POLL_SECONDS}"
  done
  mkdir -p "${out}"
  local checkpoint
  checkpoint="$("${TECE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "${raw}")"
  if [[ "${checkpoint}" != /* ]]; then
    checkpoint="${run}/${checkpoint}"
  fi
  echo "START_TECE_C${channels} checkpoint=${checkpoint} $(date -Is)"
  "${TECE_PYTHON}" "${REPO}/benchmarks/paper/scripts/training/calibrate_tece_t1x_energy.py" \
    --checkpoint "${checkpoint}" --data "${DATA_ROOT}/extxyz" \
    --elements O,Zr --batch-size 16 --out "${out}" \
    >"${out}/run.log" 2>&1
  echo "DONE_TECE_C${channels} $(date -Is)"
}

run_dpa 32
run_dpa 48
run_tece 36
run_tece 48
echo "MS25_ZRO_EXTERNAL_CALIBRATION_ALL_DONE $(date -Is)"
