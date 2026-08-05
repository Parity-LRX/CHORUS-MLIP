#!/usr/bin/env bash
#SBATCH --job-name=chorus_scope_md22
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_persistent_runs/slurm-%x-%j.log

set -euo pipefail

SYSTEM="${SYSTEM:?submit with SYSTEM=Ac_Ala3_NHMe or SYSTEM=DHA}"
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-operator}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/chorus_persistent_data/md22}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/chorus_persistent_runs/md22_final_persistent_c64_r8_20260730}"
SEED="${SEED:-20260616}"

case "${SYSTEM}" in
  Ac_Ala3_NHMe|DHA) ;;
  *) echo "unsupported SYSTEM=${SYSTEM}" >&2; exit 2 ;;
esac

DATA_DIR="${DATA_ROOT}/${SYSTEM}/processed"
OUT_ROOT="${RUN_ROOT}/${SYSTEM}/seed${SEED}"
TRAIN_ENV="${DATA_ROOT}/${SYSTEM}/training.env"
RUNNER="${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
EVALUATOR="${REPO}/benchmarks/paper/scripts/training/evaluate_md22_heldout.sh"

test -f "${DATA_DIR}/processed_train.h5"
test -f "${DATA_DIR}/processed_val.h5"
test -f "${DATA_DIR}/processed_test.h5"
# shellcheck disable=SC1090
source "${TRAIN_ENV}"
mkdir -p "${OUT_ROOT}" /public/home/sps-xia/rxlin/chorus_persistent_runs

{
  echo "START $(date --iso-8601=seconds)"
  echo "HOST $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "SYSTEM=${SYSTEM}"
  echo "CONFIG channels=64 hidden_lmax=1 max_ell=2 correlation=2 interactions=2 rank=8 seed=${SEED}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "${OUT_ROOT}/a100_protocol.log"

MACE_ICTC_REPO="${REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
DATASET_TAG="${SYSTEM}" \
DATA_DIR="${DATA_DIR}" \
OUT_ROOT="${OUT_ROOT}" \
MODES=ictc_phase_full_l_softplus_makefx,ictc_phase_full_l_persistent_softplus_makefx \
SEED="${SEED}" \
EPOCHS=300 \
BATCH_SIZE=4 \
CHANNELS=64 \
HIDDEN_LMAX=1 \
MAX_ELL=2 \
NUM_INTERACTIONS=2 \
CORRELATION=2 \
READOUT_HIDDEN=64 \
PHASE_DENSITY_RANK=8 \
AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
E0_KEYS="${E0_KEYS}" \
E0_VALUES="${E0_VALUES}" \
bash "${RUNNER}" >"${OUT_ROOT}/driver.log" 2>&1

MACE_ICTC_REPO="${REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
CHECKPOINT_DIR="${OUT_ROOT}/checkpoints" \
OUT_DIR="${OUT_ROOT}/heldout_test" \
DATA_DIR="${DATA_DIR}" \
BATCH_SIZE=4 \
AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
E0_KEYS="${E0_KEYS}" \
E0_VALUES="${E0_VALUES}" \
SEED="${SEED}" \
bash "${EVALUATOR}" >"${OUT_ROOT}/heldout_test_driver.log" 2>&1

touch "${OUT_ROOT}/ALL_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)" | tee -a "${OUT_ROOT}/a100_protocol.log"
