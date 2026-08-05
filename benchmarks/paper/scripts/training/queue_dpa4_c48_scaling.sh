#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/dpa4_c48_scaling_20260727}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"
WAIT_SCREENS="${WAIT_SCREENS:-3bpa_chorus_attention 3bpa_external}"
MAX_BACKGROUND_GPU_MEMORY_MB="${MAX_BACKGROUND_GPU_MEMORY_MB:-4096}"
CHANNELS="${CHANNELS:-48}"
MIXING_LAYERS="${MIXING_LAYERS:-3}"
SEED="${SEED:-20260616}"
USE_COMPILE="${USE_COMPILE:-0}"
USE_AMP="${USE_AMP:-1}"
XXMD_SYSTEMS="${XXMD_SYSTEMS:-mal sti dia}"

THREE_BPA_DATA=/home/ylzhang/lrx/3bpa/deepmd_standard_450_50_seed20260616_r5
BUCKY_DATA=/home/ylzhang/lrx/md22/dpa4_buckyball_matched_20260724
XXMD_DATA=/home/ylzhang/lrx/xxmd/deepmd_temporal_r5
T1X_DATA=/home/ylzhang/lrx/transition1x/deepmd_reaction_id_50k_seed20260616

mkdir -p "${ROOT}/driver_logs"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

gpu_memory_mb() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
    2>/dev/null | awk '{sum += $1} END {print sum + 0}'
}

wait_for_gpu() {
  local used_mb
  while true; do
    used_mb="$(gpu_memory_mb)"
    if (( used_mb <= MAX_BACKGROUND_GPU_MEMORY_MB )); then
      return
    fi
    mark "WAIT_GPU used_mb=${used_mb} threshold_mb=${MAX_BACKGROUND_GPU_MEMORY_MB}"
    sleep 30
  done
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
    local code=$?
    stage_failures=$((stage_failures + 1))
    mark "FAILED_${name}_exit${code}"
  fi
}

train_dpa() {
  local data="$1"
  local out="$2"
  local steps="$3"
  env \
    DATA="${data}" OUT="${out}" STEPS="${steps}" BATCH_SIZE=16 \
    CHANNELS="${CHANNELS}" MIXING_LAYERS="${MIXING_LAYERS}" RCUT=5.0 \
    USE_COMPILE="${USE_COMPILE}" USE_AMP="${USE_AMP}" \
    SEED="${SEED}" REPO="${REPO}" DPA_ENV="${DPA_ENV}" \
    bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_fixed_xxmd.sh"
}

evaluate_dpa() {
  local data="$1"
  local out="$2"
  local steps="$3"
  local checkpoint_every="${4:-1}"
  "${DPA_ENV}/bin/python" \
    "${REPO}/benchmarks/paper/scripts/training/evaluate_dpa4_xxmd_force_mae.py" \
    --run "${out}" --data "${data}" \
    --out "${out}/full_val_force_mae_eval" --steps "${steps}" \
    --checkpoint-every "${checkpoint_every}"
}

for wait_screen in ${WAIT_SCREENS}; do
  mark "WAIT_SCREEN_${wait_screen}"
  while screen_exists "${wait_screen}"; do
    sleep 30
  done
done
mark "PREREQUISITE_SCREENS_DONE"

# 3BPA first: it gives the fastest direct test of whether capacity improves
# the 300 K -> 1200 K extrapolation curve.
THREE_BPA_OUT="${ROOT}/3bpa_c48_mix3"
run_stage 3bpa_train \
  train_dpa "${THREE_BPA_DATA}/300K" "${THREE_BPA_OUT}" 45000
if [[ -f "${THREE_BPA_OUT}/DONE" ]]; then
  run_stage 3bpa_select_test300 \
    evaluate_dpa "${THREE_BPA_DATA}/300K" "${THREE_BPA_OUT}" 45000 10
fi
if [[ -f "${THREE_BPA_OUT}/full_val_force_mae_eval/metrics.json" ]]; then
  selected="$("${DPA_ENV}/bin/python" - \
    "${THREE_BPA_OUT}/full_val_force_mae_eval/metrics.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])
PY
)"
  for temperature in 600K 1200K; do
    run_stage "3bpa_test_${temperature}" \
      env NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
      "${DPA_ENV}/bin/dp" --pt test -m "${selected}" \
      -s "${THREE_BPA_DATA}/${temperature}/test" -n 0
  done
fi

# MD22 Buckyball uses the existing matched 600/train validation/test split.
BUCKY_OUT="${ROOT}/bucky_c48_mix3"
run_stage bucky_train train_dpa "${BUCKY_DATA}" "${BUCKY_OUT}" 45000
if [[ -f "${BUCKY_OUT}/DONE" ]]; then
  run_stage bucky_select_test \
    evaluate_dpa "${BUCKY_DATA}" "${BUCKY_OUT}" 45000 10
fi

# Official temporal xxMD splits.
for system in ${XXMD_SYSTEMS}; do
  out="${ROOT}/xxmd_${system}_c48_mix3"
  run_stage "xxmd_${system}_train" \
    train_dpa "${XXMD_DATA}/${system}" "${out}" 45000
  if [[ -f "${out}/DONE" ]]; then
    run_stage "xxmd_${system}_select_test" \
      evaluate_dpa "${XXMD_DATA}/${system}" "${out}" 45000 1
  fi
done

# Transition1x is last because its 100k-step budget is the longest.
T1X_OUT="${ROOT}/t1x_c48_mix3"
run_stage t1x_train train_dpa "${T1X_DATA}" "${T1X_OUT}" 100000
if [[ -f "${T1X_OUT}/DONE" ]]; then
  run_stage t1x_select_test evaluate_dpa "${T1X_DATA}" "${T1X_OUT}" 100000 1
fi

if (( stage_failures == 0 )); then
  touch "${ROOT}/DONE"
  mark "ALL_DPA4_C48_SCALING_DONE"
else
  touch "${ROOT}/COMPLETED_WITH_FAILURES"
  mark "ALL_DPA4_C48_STAGES_ATTEMPTED failures=${stage_failures}"
  exit 1
fi
