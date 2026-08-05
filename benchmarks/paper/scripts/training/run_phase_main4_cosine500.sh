#!/usr/bin/env bash
set -euo pipefail

# From-scratch 500-epoch campaign for the four paper-table models:
# ordinary MACE-ICTC, CHORUS-Final, diagonal j=k, and final-layer
# density-preserving positive QK attention.
# Every run uses the same make_fx execution path, initial LR, and
# optimizer-step cosine schedule.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/mechanism_main4_makefx_cosine500_20260720}"
SEEDS="${SEEDS:-20260616,20260617,20260618}"
MODES="${MODES:-ictc_bridge_u_makefx,ictc_phase_full_l_softplus_makefx,ictc_phase_diagonal_full_l_makefx,ictc_attention_makefx}"
INITIAL_LR="${INITIAL_LR:-0.001}"
MIN_LR="${MIN_LR:-1e-6}"
PARALLEL_CAMPAIGNS="${PARALLEL_CAMPAIGNS:-2}"
MAKEFX_BUCKETS="${MAKEFX_BUCKETS:-4}"
MAKEFX_MAX_SLOTS="${MAKEFX_MAX_SLOTS:-8}"

mkdir -p "${OUT_ROOT}"
cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "protocol": "main4_from_scratch_cosine500",
  "datasets": ["revised_aspirin", "revised_benzene", "revised_ethanol", "cheng_water"],
  "seeds": "${SEEDS}",
  "modes": "${MODES}",
  "epochs": 500,
  "initial_learning_rate": ${INITIAL_LR},
  "minimum_learning_rate": ${MIN_LR},
  "scheduler": "optimizer-step cosine",
  "execution": "required make_fx compilation",
  "attention": {"mode": "density-preserving", "scope": "final"},
  "makefx_buckets": "${MAKEFX_BUCKETS}",
  "makefx_max_slots": ${MAKEFX_MAX_SLOTS},
  "padding": ["nodes-to-bucket-max", "edges-to-bucket-max"],
  "checkpoint_selection": "minimum validation loss",
  "primary_reporting": "force and energy MAE from the same selected checkpoint",
  "from_scratch": true,
  "parallel_campaigns": ${PARALLEL_CAMPAIGNS}
}
EOF

run_molecules() {
  DATA_ROOT=/tmp/mace_ictd_public_md17 \
  DATASETS=revised_aspirin,revised_benzene,revised_ethanol \
  AVG_NEIGHBORS=8.0 \
  SEEDS="${SEEDS}" \
  MODES="${MODES}" \
  EPOCHS=500 \
  LR="${INITIAL_LR}" \
  MIN_LR="${MIN_LR}" \
  LR_SCHEDULER=cosine \
  TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS="${MAKEFX_BUCKETS}" \
  MAKEFX_MAX_SLOTS="${MAKEFX_MAX_SLOTS}" \
  PARALLEL_JOBS=1 \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${OUT_ROOT}/molecules" \
  bash "${RUN_MATRIX}"
}

run_water() {
  DATA_ROOT=/tmp/mace_ictd_public_water \
  DATASETS=cheng_water \
  AVG_NEIGHBORS=34.0 \
  SEEDS="${SEEDS}" \
  MODES="${MODES}" \
  EPOCHS=500 \
  LR="${INITIAL_LR}" \
  MIN_LR="${MIN_LR}" \
  LR_SCHEDULER=cosine \
  TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS="${MAKEFX_BUCKETS}" \
  MAKEFX_MAX_SLOTS="${MAKEFX_MAX_SLOTS}" \
  PARALLEL_JOBS=1 \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${OUT_ROOT}/water" \
  bash "${RUN_MATRIX}"
}

if [[ "${PARALLEL_CAMPAIGNS}" == "2" ]]; then
  run_molecules >"${OUT_ROOT}/molecules_driver.log" 2>&1 &
  molecules_pid=$!
  run_water >"${OUT_ROOT}/water_driver.log" 2>&1 &
  water_pid=$!
  echo "START molecules pid=${molecules_pid} $(date)" | tee -a "${OUT_ROOT}/status.log"
  echo "START water pid=${water_pid} $(date)" | tee -a "${OUT_ROOT}/status.log"
  rc=0
  wait "${molecules_pid}" || rc=1
  wait "${water_pid}" || rc=1
  if [[ "${rc}" != "0" ]]; then
    echo "FAIL child campaign $(date)" | tee -a "${OUT_ROOT}/status.log"
    exit 1
  fi
else
  echo "START molecules $(date)" | tee -a "${OUT_ROOT}/status.log"
  run_molecules
  echo "START water $(date)" | tee -a "${OUT_ROOT}/status.log"
  run_water
fi

"${PYTHON_BIN}" \
  "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/analyze_md17_convergence.py" \
  "${OUT_ROOT}/molecules/logs" "${OUT_ROOT}/water/logs" \
  --out-dir "${OUT_ROOT}/analysis" \
  --target-epoch 500 \
  --plots

echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
echo "results: ${OUT_ROOT}"
