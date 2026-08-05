#!/usr/bin/env bash
set -euo pipefail

# Mechanism-improvement triage for CHORUS-Final.  ForceTrainer uses max_steps
# as the cosine horizon, so this deliberately compressed schedule is only for
# ranking candidates; it is not a same-epoch comparison with cosine500 runs.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/coherence_improvements_screen_20260720}"
SEED="${SEED:-20260616}"
MAX_STEPS="${MAX_STEPS:-9000}"
MODES="${MODES:-ictc_phase_full_l_local_makefx,ictc_phase_full_l_gated_makefx,ictc_phase_full_l_gated_local_makefx}"

mkdir -p "${OUT_ROOT}"
cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "protocol": "coherence_improvements_screen",
  "datasets": ["revised_aspirin", "cheng_water"],
  "seed": "${SEED}",
  "modes": "${MODES}",
  "nominal_epochs": 500,
  "early_stop_optimizer_steps": ${MAX_STEPS},
  "scheduler": {"type": "cosine", "initial_lr": 0.001, "minimum_lr": 1e-6, "horizon_optimizer_steps": ${MAX_STEPS}},
  "execution": "required make_fx compilation",
  "selection_rule": "heuristic candidate ranking only; any promoted variant must be rerun with the complete cosine500 protocol",
  "normalization": {
    "avg-neighbors": "legacy 1/avg_n charged normalization",
    "local-effective": "1/sqrt(avg_n*n_eff(i)), n_eff=(sum envelope)^2/sum envelope^2"
  },
  "coherence_gate": "rho_L = D_L + gamma_L (rho_full_L-D_L), gamma_L initialized to 1"
}
EOF

run_aspirin() {
  DATA_ROOT=/tmp/mace_ictd_public_md17 \
  DATASETS=revised_aspirin \
  AVG_NEIGHBORS=8.0 \
  SEEDS="${SEED}" \
  MODES="${MODES}" \
  EPOCHS=500 \
  MAX_STEPS="${MAX_STEPS}" \
  LR=0.001 \
  MIN_LR=1e-6 \
  LR_SCHEDULER=cosine \
  TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS=4 \
  MAKEFX_MAX_SLOTS=8 \
  PARALLEL_JOBS=1 \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${OUT_ROOT}/aspirin" \
  bash "${RUN_MATRIX}"
}

run_water() {
  DATA_ROOT=/tmp/mace_ictd_public_water \
  DATASETS=cheng_water \
  AVG_NEIGHBORS=34.0 \
  SEEDS="${SEED}" \
  MODES="${MODES}" \
  EPOCHS=500 \
  MAX_STEPS="${MAX_STEPS}" \
  LR=0.001 \
  MIN_LR=1e-6 \
  LR_SCHEDULER=cosine \
  TRAIN_MAKEFX_COMPILE=1 \
  MAKEFX_BUCKETS=4 \
  MAKEFX_MAX_SLOTS=8 \
  PARALLEL_JOBS=1 \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${OUT_ROOT}/water" \
  bash "${RUN_MATRIX}"
}

run_aspirin >"${OUT_ROOT}/aspirin_driver.log" 2>&1 &
aspirin_pid=$!
run_water >"${OUT_ROOT}/water_driver.log" 2>&1 &
water_pid=$!
echo "START aspirin pid=${aspirin_pid} $(date)" | tee -a "${OUT_ROOT}/status.log"
echo "START water pid=${water_pid} $(date)" | tee -a "${OUT_ROOT}/status.log"

rc=0
wait "${aspirin_pid}" || rc=1
wait "${water_pid}" || rc=1
if [[ "${rc}" != "0" ]]; then
  echo "FAIL child campaign $(date)" | tee -a "${OUT_ROOT}/status.log"
  exit 1
fi
echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
