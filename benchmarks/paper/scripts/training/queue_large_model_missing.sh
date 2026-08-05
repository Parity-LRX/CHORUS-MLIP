#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
WAIT_SCREENS="${WAIT_SCREENS:-external_large_xxmd_queue native_mace_baselines_queue}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/large_model_missing_20260725}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"
MAX_BACKGROUND_GPU_MEMORY_MB="${MAX_BACKGROUND_GPU_MEMORY_MB:-4096}"

DPA_XXMD_ROOT=/home/ylzhang/chorus_runs/large_scale_main_20260724/external/dpa4_xxmd_large
DPA_XXMD_DATA=/home/ylzhang/lrx/xxmd/deepmd_temporal_r5
DPA_T1X_DATA=/home/ylzhang/lrx/transition1x/deepmd_reaction_id_50k_seed20260616
DPA_T1X_RUN=/home/ylzhang/chorus_runs/large_scale_main_20260724/external/dpa4_t1x_large/c32_mix3
T1X_SOURCE=/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616
BUCKY_H5=/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed/processed_test.h5
BUCKY_TECE_TEST=/home/ylzhang/tace_chorus_data/buckyball/test.extxyz

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

# Finish the missing strict full-validation scan first; its training is
# already complete and test remains isolated until validation selects a step.
run_stage dpa4_sti_full_validation \
  "${DPA_ENV}/bin/python" \
  "${REPO}/benchmarks/paper/scripts/training/evaluate_dpa4_xxmd_force_mae.py" \
  --run "${DPA_XXMD_ROOT}/sti_c32_mix3" \
  --data "${DPA_XXMD_DATA}/sti" \
  --out "${DPA_XXMD_ROOT}/sti_c32_mix3/full_val_force_mae_eval" \
  --steps 45000

# Existing T1x and MAL training is reused; only the validation-selected test
# evaluation is added.  The three actually missing rank-32 trainings follow.
for system in t1x mal sti dia bucky; do
  run_stage "chorus_rank32_${system}" \
    env SYSTEMS="${system}" \
    bash "${REPO}/benchmarks/paper/scripts/training/queue_chorus_rank32_pilot.sh"
done

# Transition1x contains 64 atom-order topologies, so it is represented as
# multiple fixed-topology DeepMD systems without changing split membership.
run_stage dpa4_t1x_prepare \
  "${PYTHON_BIN}" \
  "${REPO}/benchmarks/paper/scripts/training/prepare_grouped_deepmd_npy_from_ictc_h5.py" \
  --source-dir "${T1X_SOURCE}" \
  --output-dir "${DPA_T1X_DATA}" \
  --type-map 1,6,7,8

if [[ -f "${DPA_T1X_DATA}/DONE" ]]; then
  run_stage dpa4_t1x_train \
    env DATA="${DPA_T1X_DATA}" OUT="${DPA_T1X_RUN}" \
    STEPS=100000 BATCH_SIZE=16 CHANNELS=32 MIXING_LAYERS=3 RCUT=5.0 \
    SEED=20260616 \
    bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_fixed_xxmd.sh"
else
  mark "SKIP_dpa4_t1x_train_dependency_not_done"
fi

if [[ -f "${DPA_T1X_RUN}/DONE" ]]; then
  run_stage dpa4_t1x_full_validation \
    "${DPA_ENV}/bin/python" \
    "${REPO}/benchmarks/paper/scripts/training/evaluate_dpa4_xxmd_force_mae.py" \
    --run "${DPA_T1X_RUN}" \
    --data "${DPA_T1X_DATA}" \
    --out "${DPA_T1X_RUN}/full_val_force_mae_eval" \
    --steps 100000
else
  mark "SKIP_dpa4_t1x_full_validation_dependency_not_done"
fi

# The exact Bucky train/validation extxyz files already match the ICTC HDF5
# split.  Add only the held-out test conversion before TECE evaluation.
run_stage tece_bucky_test_prepare \
  "${PYTHON_BIN}" \
  "${REPO}/benchmarks/paper/scripts/training/prepare_extxyz_from_ictc_h5.py" \
  --source "${BUCKY_H5}" \
  --output "${BUCKY_TECE_TEST}" \
  --energy-key Energy \
  --forces-key force

run_stage tece_large_t1x \
  env SYSTEMS=t1x \
  bash "${REPO}/benchmarks/paper/external/tece/run_tece_large_missing_queue.sh"

if [[ -f "${BUCKY_TECE_TEST}.DONE" ]]; then
  run_stage tece_large_bucky \
    env SYSTEMS=bucky \
    bash "${REPO}/benchmarks/paper/external/tece/run_tece_large_missing_queue.sh"
else
  mark "SKIP_tece_large_bucky_dependency_not_done"
fi

if (( stage_failures == 0 )); then
  touch "${ROOT}/DONE"
  mark "ALL_MISSING_LARGE_MODEL_STAGES_DONE"
else
  touch "${ROOT}/COMPLETED_WITH_FAILURES"
  mark "ALL_STAGES_ATTEMPTED failures=${stage_failures}"
  exit 1
fi
