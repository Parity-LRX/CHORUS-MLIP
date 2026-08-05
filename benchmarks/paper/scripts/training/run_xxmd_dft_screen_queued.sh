#!/usr/bin/env bash
set -euo pipefail

# Phase-one xxMD-DFT screen.  Preserve the official temporal split and compare
# the five paper-table mechanisms with the same 45k optimizer-step budget used
# by the 300-epoch MD22 experiment.  Run a
# single seed on all four molecules before committing multiple seeds to the
# most discriminating systems.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
EVALUATOR="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/evaluate_xxmd_dft.sh"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5}"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/xxmd_dft_temporal_steps45000_20260721}"
WAIT_SCREENS="${WAIT_SCREENS:-chorus_md22_multiseed,chorus_md22_testwatch}"
MOLECULES="${MOLECULES:-azo,sti,dia,mal}"
SEED="${SEED:-20260616}"
MAX_STEPS="${MAX_STEPS:-45000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100}"
MODES="${MODES:-ictc_bridge_u_makefx,ictc_phase_full_l_softplus_makefx,ictc_phase_diagonal_full_l_makefx,ictc_attention_makefx,ictc_attention_legacy_makefx}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
SKIP_OFFICIAL_TEST="${SKIP_OFFICIAL_TEST:-0}"
EMA_DECAY="${EMA_DECAY:-0.0}"
EMA_START_STEP="${EMA_START_STEP:-0}"

mkdir -p "${OUT_ROOT}"
cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "protocol": "xxmd_dft_official_temporal_steps45000",
  "source": "https://doi.org/10.5281/zenodo.10393859",
  "molecules_in_execution_order": "${MOLECULES}",
  "seed": "${SEED}",
  "modes": "${MODES}",
  "split": "official temporal train/validation/test; no random resplitting",
  "optimizer_steps_per_run": ${MAX_STEPS},
  "examples_seen_per_run": $((MAX_STEPS * BATCH_SIZE)),
  "batch_size": ${BATCH_SIZE},
  "initial_learning_rate": 0.001,
  "minimum_learning_rate": 1e-6,
  "scheduler": "optimizer-step cosine",
  "loss": {"type": "mse", "energy_weight": 1, "force_weight": ${FORCE_WEIGHT}},
  "execution": "required make_fx, one GPU job at a time",
  "checkpoint_selection": "minimum validation loss",
  "reporting": "energy and force MAE on the official test split from the same selected checkpoint",
  "stage_two_policy": "add two seeds only after ranking full-U1 versus diagonal and attention across all four molecules"
}
EOF

screen_is_alive() {
  local target="$1"
  screen -ls 2>/dev/null | grep -Eq "[.]${target}[[:space:]]"
}

wait_for_gpu_queue() {
  local waiting=0
  local raw_screen
  IFS=',' read -r -a wait_array <<<"${WAIT_SCREENS}"
  while true; do
    waiting=0
    for raw_screen in "${wait_array[@]}"; do
      raw_screen="$(echo "${raw_screen}" | xargs)"
      if [[ -n "${raw_screen}" ]] && screen_is_alive "${raw_screen}"; then
        waiting=1
        break
      fi
    done
    if (( waiting == 0 )); then
      break
    fi
    echo "WAIT active_screen=${raw_screen} $(date)" | tee -a "${OUT_ROOT}/status.log"
    sleep 30
  done
  echo "GPU_QUEUE_CLEAR $(date)" | tee -a "${OUT_ROOT}/status.log"
}

metadata_value() {
  local metadata="$1"
  local expression="$2"
  "${PYTHON_BIN}" -c "import json; d=json.load(open('${metadata}')); print(${expression})"
}

run_molecule() {
  local molecule="$1"
  local data_dir="${DATA_ROOT}/${molecule}"
  local metadata="${data_dir}/metadata.json"
  local molecule_root="${OUT_ROOT}/${molecule}"
  local train_frames avg_neighbors steps_per_epoch epochs

  if [[ -f "${molecule_root}/.complete" ]]; then
    echo "SKIP_COMPLETE ${molecule} $(date)" | tee -a "${OUT_ROOT}/status.log"
    return
  fi
  for required in train.extxyz val.extxyz test.extxyz processed_train.h5 processed_val.h5 processed_test.h5 metadata.json; do
    if [[ ! -f "${data_dir}/${required}" ]]; then
      echo "missing prepared xxMD file ${data_dir}/${required}" >&2
      exit 3
    fi
  done

  train_frames="$(metadata_value "${metadata}" "d['splits']['train']['frames']")"
  avg_neighbors="$(metadata_value "${metadata}" "d['splits']['train']['mean_directed_neighbors']")"
  steps_per_epoch=$(((train_frames + BATCH_SIZE - 1) / BATCH_SIZE))
  epochs=$(((MAX_STEPS + steps_per_epoch - 1) / steps_per_epoch))

  echo "START ${molecule} frames=${train_frames} avg_neighbors=${avg_neighbors} epochs=${epochs} max_steps=${MAX_STEPS} $(date)" \
    | tee -a "${OUT_ROOT}/status.log"
  DATA_ROOT="${DATA_ROOT}" \
  DATASETS="${molecule}" \
  AVG_NEIGHBORS="${avg_neighbors}" \
  SEEDS="${SEED}" \
  MODES="${MODES}" \
  EPOCHS="${epochs}" \
  MAX_STEPS="${MAX_STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  R_MAX=5.0 \
  LR=0.001 \
  MIN_LR=1e-6 \
  LR_SCHEDULER=cosine \
  ENERGY_WEIGHT=1 \
  FORCE_WEIGHT="${FORCE_WEIGHT}" \
  TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS=4 \
  MAKEFX_MAX_SLOTS=8 \
  PARALLEL_JOBS=1 \
  KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS}" \
  EMA_DECAY="${EMA_DECAY}" \
  EMA_START_STEP="${EMA_START_STEP}" \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${molecule_root}/train" \
  bash "${RUN_MATRIX}" >"${OUT_ROOT}/${molecule}_train_driver.log" 2>&1

  if [[ "${SKIP_OFFICIAL_TEST}" == "1" ]]; then
    echo "SKIP_OFFICIAL_TEST ${molecule} $(date)" | tee -a "${OUT_ROOT}/status.log"
    return
  fi

  # The matrix runner records the minimum-norm fixed-composition E0 values.
  # Reuse those exact values for the untouched official temporal test split.
  # shellcheck disable=SC1090
  source "${molecule_root}/train/metadata/${molecule}.env"
  DATA_DIR="${data_dir}" \
  CHECKPOINT_DIR="${molecule_root}/train/checkpoints" \
  OUT_DIR="${molecule_root}/official_test" \
  AVG_NEIGHBORS="${avg_neighbors}" \
  E0_KEYS="${E0_KEYS}" \
  E0_VALUES="${E0_VALS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  FORCE_WEIGHT="${FORCE_WEIGHT}" \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVALUATOR}" >"${OUT_ROOT}/${molecule}_test_driver.log" 2>&1

  touch "${molecule_root}/.complete"
  echo "END ${molecule} rc=0 $(date)" | tee -a "${OUT_ROOT}/status.log"
}

wait_for_gpu_queue
IFS=',' read -r -a molecule_array <<<"${MOLECULES}"
for raw_molecule in "${molecule_array[@]}"; do
  run_molecule "$(echo "${raw_molecule}" | xargs)"
done

echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
