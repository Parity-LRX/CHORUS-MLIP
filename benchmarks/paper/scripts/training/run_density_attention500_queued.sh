#!/usr/bin/env bash
set -euo pipefail

# Serial 500-epoch validation of the scope-matched density-preserving attention
# control. When WAIT_PID is provided, do not start until that complete screen
# session exits; this prevents overlap with an existing multi-stage campaign.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/density_attention_final_cosine500_20260720}"
WAIT_PID="${WAIT_PID:-}"
SEEDS="${SEEDS:-20260616,20260617,20260618}"

mkdir -p "${OUT_ROOT}"
cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "protocol": "density_preserving_final_attention_cosine500",
  "operator": {
    "attention_heads": 4,
    "attention_mode": "density-preserving",
    "attention_scope": "final",
    "phase_mode": "none"
  },
  "dataset_order": [
    "revised_aspirin",
    "revised_benzene",
    "revised_ethanol",
    "cheng_water"
  ],
  "seeds": "${SEEDS}",
  "epochs": 500,
  "initial_learning_rate": 0.001,
  "minimum_learning_rate": 1e-6,
  "scheduler": "optimizer-step cosine",
  "execution": "required make_fx compilation, strictly serial",
  "checkpoint_selection": "minimum validation loss",
  "primary_reporting": "energy and force MAE from the same selected checkpoint"
}
EOF

if [[ -n "${WAIT_PID}" ]]; then
  echo "WAIT pid=${WAIT_PID} $(date)" | tee -a "${OUT_ROOT}/status.log"
  while ps -p "${WAIT_PID}" -o args= 2>/dev/null | grep -q "chorus_pair_balance"; do
    sleep 30
  done
fi

run_one() {
  local dataset="$1"
  local data_root="$2"
  local avg_neighbors="$3"
  echo "START ${dataset} $(date)" | tee -a "${OUT_ROOT}/status.log"
  DATA_ROOT="${data_root}" \
  DATASETS="${dataset}" \
  AVG_NEIGHBORS="${avg_neighbors}" \
  SEEDS="${SEEDS}" \
  MODES=ictc_attention_makefx \
  EPOCHS=500 \
  LR=0.001 \
  MIN_LR=1e-6 \
  LR_SCHEDULER=cosine \
  TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS=4 \
  MAKEFX_MAX_SLOTS=8 \
  PARALLEL_JOBS=1 \
  ATTN_HEADS=4 \
  ATTN_MODE=density-preserving \
  ATTN_SCOPE=final \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${OUT_ROOT}/${dataset}" \
  bash "${RUN_MATRIX}" \
    >"${OUT_ROOT}/${dataset}_driver.log" 2>&1
  echo "END ${dataset} rc=0 $(date)" | tee -a "${OUT_ROOT}/status.log"
}

run_one revised_aspirin /tmp/mace_ictd_public_md17 8.0
run_one revised_benzene /tmp/mace_ictd_public_md17 8.0
run_one revised_ethanol /tmp/mace_ictd_public_md17 8.0
run_one cheng_water /tmp/mace_ictd_public_water 34.0

echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
