#!/usr/bin/env bash
#SBATCH --job-name=chorus_diag_persist
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_persistent_runs/slurm-%x-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/chorus_persistent_data/md22}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/chorus_persistent_runs/persistent_diagonal_c64_r8_20260802}"
SEED="${SEED:-20260616}"
RUNNER="${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
EVALUATOR="${REPO}/benchmarks/paper/scripts/training/evaluate_md22_heldout.sh"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${RUN_ROOT}"

for system in Ac_Ala3_NHMe DHA; do
  data_dir="${DATA_ROOT}/${system}/processed"
  out="${RUN_ROOT}/${system}/seed${SEED}"
  train_env="${DATA_ROOT}/${system}/training.env"
  test -f "${data_dir}/processed_train.h5"
  test -f "${data_dir}/processed_val.h5"
  test -f "${data_dir}/processed_test.h5"
  # shellcheck disable=SC1090
  source "${train_env}"
  mkdir -p "${out}"

  if [[ ! -f "${out}/ALL_DONE" ]]; then
    MACE_ICTC_REPO="${REPO}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
    DATASET_TAG="${system}" \
    DATA_DIR="${data_dir}" \
    OUT_ROOT="${out}" \
    MODES=ictc_phase_diagonal_full_l_all_layers_softplus_makefx \
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
    bash "${RUNNER}" >"${out}/driver.log" 2>&1

    MACE_ICTC_REPO="${REPO}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
    CHECKPOINT_DIR="${out}/checkpoints" \
    OUT_DIR="${out}/heldout_test" \
    DATA_DIR="${data_dir}" \
    BATCH_SIZE=4 \
    AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
    E0_KEYS="${E0_KEYS}" \
    E0_VALUES="${E0_VALUES}" \
    SEED="${SEED}" \
    bash "${EVALUATOR}" >"${out}/heldout_test_driver.log" 2>&1
    touch "${out}/ALL_DONE"
  fi
done

touch "${RUN_ROOT}/ALL_DONE"
