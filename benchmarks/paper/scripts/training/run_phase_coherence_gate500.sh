#!/usr/bin/env bash
set -euo pipefail

# Full-length confirmatory run for the three-parameter coherence gate.
# Water and aspirin are deliberately sequential so the confirmatory timings and
# optimization trajectories are not contaminated by GPU contention.  The exact
# diagonal density uses the factorized |c_e|^2 (b_e tensor b_e) implementation.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/coherence_gate_cosine500_20260720}"
SEED="${SEED:-20260616}"
MODE="ictc_phase_full_l_gated_makefx"

mkdir -p "${OUT_ROOT}"
cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "protocol": "coherence_gate_confirmatory_cosine500",
  "datasets": ["cheng_water", "revised_aspirin"],
  "seed": "${SEED}",
  "mode": "${MODE}",
  "epochs": 500,
  "max_steps": null,
  "optimizer": "AdamW with MACE parameter groups",
  "scheduler": {"type": "cosine", "initial_lr": 0.001, "minimum_lr": 1e-6, "horizon_epochs": 500},
  "execution": "required make_fx compilation, sequential datasets, factorized exact diagonal density",
  "batch_size": 16,
  "cuda_allocator": "expandable_segments:True",
  "comparison": "existing seed-matched Full-U1 cosine500 checkpoint and log",
  "operator": "rho_L = D_L + gamma_L (rho_full_L-D_L), gamma_L initialized to 1"
}
EOF

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

rc=0
echo "START water $(date)" | tee -a "${OUT_ROOT}/status.log"
run_one cheng_water /tmp/mace_ictd_public_water 34.0 "${OUT_ROOT}/water" \
  >"${OUT_ROOT}/water_driver.log" 2>&1 || rc=1
echo "END water rc=${rc} $(date)" | tee -a "${OUT_ROOT}/status.log"

aspirin_rc=0
echo "START aspirin $(date)" | tee -a "${OUT_ROOT}/status.log"
run_one revised_aspirin /tmp/mace_ictd_public_md17 8.0 "${OUT_ROOT}/aspirin" \
  >"${OUT_ROOT}/aspirin_driver.log" 2>&1 || aspirin_rc=1
echo "END aspirin rc=${aspirin_rc} $(date)" | tee -a "${OUT_ROOT}/status.log"

if [[ "${rc}" != "0" || "${aspirin_rc}" != "0" ]]; then
  exit 1
fi
echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
