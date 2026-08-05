#!/usr/bin/env bash
set -euo pipefail

# Exact pair-count-balanced CHORUS screen, serialized after an optional running
# campaign. Tests, water, and aspirin all execute one at a time on the same GPU.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-pair-balanced}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/pair_balance_cosine500_20260720}"
WAIT_PID="${WAIT_PID:-}"
SEED="${SEED:-20260616}"
MODE="ictc_phase_full_l_pair_balanced_makefx"

mkdir -p "${OUT_ROOT}"
if [[ -n "${WAIT_PID}" ]]; then
  echo "WAIT pid=${WAIT_PID} $(date)" | tee -a "${OUT_ROOT}/status.log"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
fi

cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "protocol": "pair_count_balanced_confirmatory_cosine500",
  "datasets": ["cheng_water", "revised_aspirin"],
  "seed": "${SEED}",
  "mode": "${MODE}",
  "epochs": 500,
  "optimizer": "AdamW with MACE parameter groups",
  "scheduler": {"type": "cosine", "initial_lr": 0.001, "minimum_lr": 1e-6, "horizon_epochs": 500},
  "execution": "required make_fx compilation, serial tests then water then aspirin",
  "operator": "D scaled as n_ref/n_eff; C scaled as (n_ref(n_ref-1)+1)/(n_eff(n_eff-1)+1)",
  "effective_coordination": "smooth cutoff participation ratio (sum w)^2/sum w^2"
}
EOF

export PYTHONPATH="${MACE_ICTC_REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}"
echo "START tests $(date)" | tee -a "${OUT_ROOT}/status.log"
"${PYTHON_BIN}" -m pytest -q "${MACE_ICTC_REPO}/chorus/test/test_phase_hermitian.py" \
  >"${OUT_ROOT}/tests.log" 2>&1
echo "END tests rc=0 $(date)" | tee -a "${OUT_ROOT}/status.log"

run_one() {
  local dataset="$1"
  local data_root="$2"
  local avg_neighbors="$3"
  local output="$4"
  DATA_ROOT="${data_root}" \
  DATASETS="${dataset}" \
  AVG_NEIGHBORS="${avg_neighbors}" \
  SEEDS="${SEED}" \
  MODES="${MODE}" \
  EPOCHS=500 \
  LR=0.001 \
  MIN_LR=1e-6 \
  LR_SCHEDULER=cosine \
  TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS=4 \
  MAKEFX_MAX_SLOTS=8 \
  PARALLEL_JOBS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${output}" \
  bash "${RUN_MATRIX}"
}

echo "START water $(date)" | tee -a "${OUT_ROOT}/status.log"
run_one cheng_water /tmp/mace_ictd_public_water 34.0 "${OUT_ROOT}/water" \
  >"${OUT_ROOT}/water_driver.log" 2>&1
echo "END water rc=0 $(date)" | tee -a "${OUT_ROOT}/status.log"

echo "START aspirin $(date)" | tee -a "${OUT_ROOT}/status.log"
run_one revised_aspirin /tmp/mace_ictd_public_md17 8.0 "${OUT_ROOT}/aspirin" \
  >"${OUT_ROOT}/aspirin_driver.log" 2>&1
echo "END aspirin rc=0 $(date)" | tee -a "${OUT_ROOT}/status.log"
echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
