#!/usr/bin/env bash
set -euo pipefail

# Extend the CHORUS-Final versus CHORUS-Persistent scope comparison.
# The rMD17 Final runs already exist under the archived three-seed campaign;
# this queue adds matched Persistent runs for aspirin and benzene. It then
# adds a three-seed Persistent campaign on MD22 Buckyball Catcher, whose
# matched Final runs are also already archived.

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/persistent_scope_multisystem_20260730}"
RMD17_SEEDS="${RMD17_SEEDS:-20260616,20260617,20260618}"
MD22_SEEDS="${MD22_SEEDS:-20260616 20260617 20260618}"
STATUS="${ROOT}/queue_status.log"
RMD17_RUNNER="${REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
MD22_RUNNER="${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
MD22_EVALUATOR="${REPO}/benchmarks/paper/scripts/training/evaluate_md22_heldout.sh"

mkdir -p "${ROOT}"

echo "START_RMD17 $(date --iso-8601=seconds)" | tee -a "${STATUS}"
PYTHON_BIN="${PYTHON_BIN}" \
MACE_ICTC_REPO="${REPO}" \
DATA_ROOT=/tmp/mace_ictd_public_md17 \
DATASETS=revised_aspirin,revised_benzene \
SEEDS="${RMD17_SEEDS}" \
MODES=ictc_phase_full_l_persistent_softplus_eager \
EPOCHS=300 \
BATCH_SIZE=16 \
CHANNELS=64 \
HIDDEN_LMAX=1 \
MAX_ELL=2 \
NUM_INTERACTIONS=2 \
CORRELATION=2 \
R_MAX=4.5 \
PHASE_DENSITY_RANK=8 \
LR_SCHEDULER=exp \
LR_GAMMA=0.9993 \
TRAIN_MAKEFX_COMPILE=0 \
PARALLEL_JOBS=2 \
NUM_WORKERS=1 \
OUT_ROOT="${ROOT}/rmd17_aspirin_benzene_persistent_r8_three_seed" \
bash "${RMD17_RUNNER}" \
  >"${ROOT}/rmd17_driver.log" 2>&1
echo "DONE_RMD17 $(date --iso-8601=seconds)" | tee -a "${STATUS}"

MD22_DATA=/home/ylzhang/lrx/md22/all_lowdata600_test1000_20260723/buckyball_catcher/processed
# shellcheck disable=SC1091
source /home/ylzhang/lrx/md22/all_lowdata600_test1000_20260723/buckyball_catcher/training.env

for seed in ${MD22_SEEDS}; do
  out="${ROOT}/md22_buckyball_persistent_r8/seed${seed}"
  if grep -q "ALL_OK" "${out}/heldout_test/status.log" 2>/dev/null; then
    echo "REUSE_MD22 seed=${seed} $(date --iso-8601=seconds)" | tee -a "${STATUS}"
    continue
  fi
  mkdir -p "${out}"
  echo "START_MD22 seed=${seed} $(date --iso-8601=seconds)" | tee -a "${STATUS}"
  MACE_ICTC_REPO="${REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_TAG=buckyball_catcher \
  DATA_DIR="${MD22_DATA}" \
  OUT_ROOT="${out}" \
  MODES=ictc_phase_full_l_persistent_softplus_makefx \
  SEED="${seed}" \
  EPOCHS=300 \
  BATCH_SIZE=4 \
  CHANNELS=64 \
  HIDDEN_LMAX=1 \
  MAX_ELL=2 \
  NUM_INTERACTIONS=2 \
  CORRELATION=2 \
  PHASE_DENSITY_RANK=8 \
  AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
  E0_KEYS="${E0_KEYS}" \
  E0_VALUES="${E0_VALUES}" \
  bash "${MD22_RUNNER}" >"${out}/driver.log" 2>&1

  MACE_ICTC_REPO="${REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CHECKPOINT_DIR="${out}/checkpoints" \
  OUT_DIR="${out}/heldout_test" \
  DATA_DIR="${MD22_DATA}" \
  BATCH_SIZE=4 \
  AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
  E0_KEYS="${E0_KEYS}" \
  E0_VALUES="${E0_VALUES}" \
  SEED="${seed}" \
  bash "${MD22_EVALUATOR}" >"${out}/heldout_test_driver.log" 2>&1
  echo "DONE_MD22 seed=${seed} $(date --iso-8601=seconds)" | tee -a "${STATUS}"
done

echo "ALL_OK $(date --iso-8601=seconds)" | tee -a "${STATUS}"
