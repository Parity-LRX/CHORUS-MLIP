#!/usr/bin/env bash
set -euo pipefail

R="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PY="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
BASE="${OUT_ROOT:-${R}/benchmarks/paper/results/phase/xxmd_dft_temporal_steps45000_20260721}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5}"
MATRIX="${R}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
EVAL="${R}/benchmarks/paper/scripts/training/evaluate_xxmd_dft.sh"
MODES="ictc_bridge_u_makefx,ictc_phase_full_l_softplus_makefx,ictc_phase_diagonal_full_l_makefx,ictc_attention_makefx,ictc_attention_legacy_makefx"
SEED=20260616
MAX_STEPS=45000
BATCH_SIZE=16
FORCE_WEIGHT=100

metadata_value() {
  local file="$1"
  local expression="$2"
  "${PY}" -c "import json; d=json.load(open('${file}')); print(${expression})"
}

evaluate_one() {
  local molecule="$1"
  local avg_neighbors="$2"
  local root="${BASE}/${molecule}"
  # shellcheck disable=SC1090
  source "${root}/train/metadata/${molecule}.env"
  DATA_DIR="${DATA_ROOT}/${molecule}" \
  CHECKPOINT_DIR="${root}/train/checkpoints" \
  OUT_DIR="${root}/official_test" \
  AVG_NEIGHBORS="${avg_neighbors}" \
  E0_KEYS="${E0_KEYS}" E0_VALUES="${E0_VALS}" \
  BATCH_SIZE="${BATCH_SIZE}" FORCE_WEIGHT="${FORCE_WEIGHT}" \
  MACE_ICTC_REPO="${R}" PYTHON_BIN="${PY}" \
  bash "${EVAL}" >"${BASE}/${molecule}_test_recovery.log" 2>&1
}

dia_avg="$(metadata_value "${DATA_ROOT}/dia/metadata.json" "d['splits']['train']['mean_directed_neighbors']")"
if [[ ! -f "${BASE}/dia/.complete" ]]; then
  echo "RECOVER_DIA_TEST_START $(date)" | tee -a "${BASE}/recovery_status.log"
  evaluate_one dia "${dia_avg}"
  touch "${BASE}/dia/.complete"
  echo "END dia rc=0 recovered $(date)" | tee -a "${BASE}/status.log" "${BASE}/recovery_status.log"
fi

if [[ ! -f "${BASE}/mal/.complete" ]]; then
  metadata="${DATA_ROOT}/mal/metadata.json"
  frames="$(metadata_value "${metadata}" "d['splits']['train']['frames']")"
  avg="$(metadata_value "${metadata}" "d['splits']['train']['mean_directed_neighbors']")"
  steps_per_epoch=$(((frames + BATCH_SIZE - 1) / BATCH_SIZE))
  epochs=$(((MAX_STEPS + steps_per_epoch - 1) / steps_per_epoch))
  mkdir -p "${BASE}/mal"
  echo "START mal frames=${frames} avg_neighbors=${avg} epochs=${epochs} max_steps=${MAX_STEPS} recovered $(date)" \
    | tee -a "${BASE}/status.log" "${BASE}/recovery_status.log"
  DATA_ROOT="${DATA_ROOT}" DATASETS=mal AVG_NEIGHBORS="${avg}" \
  SEEDS="${SEED}" MODES="${MODES}" EPOCHS="${epochs}" MAX_STEPS="${MAX_STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" R_MAX=5.0 LR=0.001 MIN_LR=1e-6 LR_SCHEDULER=cosine \
  ENERGY_WEIGHT=1 FORCE_WEIGHT="${FORCE_WEIGHT}" TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS=4 MAKEFX_MAX_SLOTS=8 PARALLEL_JOBS=1 \
  MACE_ICTC_REPO="${R}" PYTHON_BIN="${PY}" OUT_ROOT="${BASE}/mal/train" \
  bash "${MATRIX}" >"${BASE}/mal_train_recovery.log" 2>&1
  evaluate_one mal "${avg}"
  touch "${BASE}/mal/.complete"
  echo "END mal rc=0 recovered $(date)" | tee -a "${BASE}/status.log" "${BASE}/recovery_status.log"
fi

echo "RECOVERY_ALL_OK $(date)" | tee -a "${BASE}/recovery_status.log"
