#!/usr/bin/env bash
set -euo pipefail

# Fill only the missing cells in the matched 500-epoch, three-seed molecular
# comparison of the ordinary ICTC trunk, full coherent Hermitian residual, and
# j=k diagonal control.  The density-preserving attention cells already exist
# and are intentionally not repeated here.  Water is excluded from this
# campaign; its historical results remain untouched.
#
# By default this campaign waits until both MD22 training and held-out testing
# screen sessions have exited, so the jobs never contend for the RTX 4090.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/four_system_missing_cosine500_20260721}"
WAIT_SCREENS="${WAIT_SCREENS:-chorus_md22_multiseed,chorus_md22_testwatch,chorus_xxmd_screen}"

mkdir -p "${OUT_ROOT}"
cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "protocol": "fill_only_missing_cells_molecular3_cosine500",
  "datasets": ["revised_benzene", "revised_ethanol", "revised_aspirin"],
  "epochs": 500,
  "initial_learning_rate": 0.001,
  "minimum_learning_rate": 1e-6,
  "scheduler": "optimizer-step cosine",
  "execution": "required make_fx compilation, strictly serial",
  "models_filled": ["ordinary ICTC", "full coherent Hermitian residual", "diagonal j=k control", "DPA-4-style legacy softmax attention"],
  "attention_policy": "reuse completed density-preserving final-attention runs; additionally train legacy-softmax attention under the matched 500-epoch protocol",
  "checkpoint_selection": "minimum validation loss",
  "primary_reporting": "force and energy MAE from the same selected checkpoint",
  "wait_screens": "${WAIT_SCREENS}",
  "missing_run_count": 32
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

run_chunk() {
  local label="$1"
  local dataset="$2"
  local data_root="$3"
  local avg_neighbors="$4"
  local seeds="$5"
  local modes="$6"
  local chunk_root="${OUT_ROOT}/${label}"

  if [[ -f "${chunk_root}/.complete" ]]; then
    echo "SKIP_COMPLETE ${label} $(date)" | tee -a "${OUT_ROOT}/status.log"
    return
  fi

  mkdir -p "${chunk_root}"
  echo "START ${label} dataset=${dataset} seeds=${seeds} modes=${modes} $(date)" \
    | tee -a "${OUT_ROOT}/status.log"
  DATA_ROOT="${data_root}" \
  DATASETS="${dataset}" \
  AVG_NEIGHBORS="${avg_neighbors}" \
  SEEDS="${seeds}" \
  MODES="${modes}" \
  EPOCHS=500 \
  LR=0.001 \
  MIN_LR=1e-6 \
  LR_SCHEDULER=cosine \
  TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS=4 \
  MAKEFX_MAX_SLOTS=8 \
  PARALLEL_JOBS=1 \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${chunk_root}" \
  bash "${RUN_MATRIX}" >"${OUT_ROOT}/${label}_driver.log" 2>&1
  touch "${chunk_root}/.complete"
  echo "END ${label} rc=0 $(date)" | tee -a "${OUT_ROOT}/status.log"
}

BASELINE=ictc_bridge_u_makefx
FULL_U1=ictc_phase_full_l_softplus_makefx
DIAGONAL=ictc_phase_diagonal_full_l_makefx
LEGACY_ATTN=ictc_attention_legacy_makefx
ALL_THREE="${BASELINE},${FULL_U1},${DIAGONAL}"

wait_for_gpu_queue

# Finish the near-complete aspirin table first.
run_chunk aspirin_seed17 revised_aspirin /tmp/mace_ictd_public_md17 8.0 \
  20260617 "${FULL_U1},${DIAGONAL}"
run_chunk aspirin_seed18 revised_aspirin /tmp/mace_ictd_public_md17 8.0 \
  20260618 "${ALL_THREE}"

# Complete the two inexpensive molecular systems.
run_chunk benzene_all_seeds revised_benzene /tmp/mace_ictd_public_md17 8.0 \
  20260616,20260617,20260618 "${ALL_THREE}"
run_chunk ethanol_all_seeds revised_ethanol /tmp/mace_ictd_public_md17 8.0 \
  20260616,20260617,20260618 "${ALL_THREE}"

# Matched DPA-4-style control. Historical 300-epoch eager runs are not mixed
# with this 500-epoch cosine/MakeFX campaign.
run_chunk aspirin_legacy_all_seeds revised_aspirin /tmp/mace_ictd_public_md17 8.0 \
  20260616,20260617,20260618 "${LEGACY_ATTN}"
run_chunk benzene_legacy_all_seeds revised_benzene /tmp/mace_ictd_public_md17 8.0 \
  20260616,20260617,20260618 "${LEGACY_ATTN}"
run_chunk ethanol_legacy_all_seeds revised_ethanol /tmp/mace_ictd_public_md17 8.0 \
  20260616,20260617,20260618 "${LEGACY_ATTN}"

echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
