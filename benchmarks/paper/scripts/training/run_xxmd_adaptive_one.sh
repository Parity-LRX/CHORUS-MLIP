#!/usr/bin/env bash
set -euo pipefail

# One official-temporal xxMD adaptive-coherence run.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
EVALUATOR="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/evaluate_xxmd_dft.sh"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5}"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/xxmd_adaptive_steps45000_20260721}"
MOLECULE="${MOLECULE:?set MOLECULE to azo, dia, or mal}"
SEED="${SEED:-20260616}"
MAX_STEPS="${MAX_STEPS:-45000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100}"
COHERENCE_INIT="${COHERENCE_INIT:-0.1}"
MODE="${MODE:-ictc_phase_full_l_adaptive_makefx}"

case "${MODE}" in
  ictc_phase_full_l_gated_makefx)
    OPERATOR_DESCRIPTION="D + gamma_L * (rho_full - D); gamma_L initialized to 1"
    ;;
  ictc_phase_full_l_adaptive_env_makefx)
    OPERATOR_DESCRIPTION="D_i + sigmoid(base_logit_L + f_L(h_i^0)) * (rho_full_i - D_i)"
    ;;
  *)
    OPERATOR_DESCRIPTION="D + sigmoid(logit_gamma_L) * (rho_full - D)"
    ;;
esac

case "${MOLECULE}" in
  azo|dia|mal) ;;
  *) echo "MOLECULE must be azo, dia, or mal, got ${MOLECULE}" >&2; exit 2 ;;
esac

data_dir="${DATA_ROOT}/${MOLECULE}"
metadata="${data_dir}/metadata.json"
molecule_root="${OUT_ROOT}/${MOLECULE}"
mkdir -p "${molecule_root}"

metadata_value() {
  "${PYTHON_BIN}" -c "import json; d=json.load(open('${metadata}')); print($1)"
}

train_frames="$(metadata_value "d['splits']['train']['frames']")"
avg_neighbors="$(metadata_value "d['splits']['train']['mean_directed_neighbors']")"
steps_per_epoch=$(((train_frames + BATCH_SIZE - 1) / BATCH_SIZE))
epochs=$(((MAX_STEPS + steps_per_epoch - 1) / steps_per_epoch))

cat >"${molecule_root}/protocol.json" <<EOF
{
  "protocol": "xxmd_adaptive_coherence_official_temporal",
  "molecule": "${MOLECULE}",
  "seed": ${SEED},
  "max_steps": ${MAX_STEPS},
  "batch_size": ${BATCH_SIZE},
  "scheduler": "cosine_1e-3_to_1e-6",
  "loss": {"type": "mse", "energy_weight": 1, "force_weight": ${FORCE_WEIGHT}},
  "operator": "${OPERATOR_DESCRIPTION}",
  "mode": "${MODE}",
  "coherence_init": ${COHERENCE_INIT},
  "checkpoint_selection": "minimum_validation_loss",
  "execution": "required_makefx; one GPU job at a time"
}
EOF

echo "START ${MOLECULE} frames=${train_frames} avg_neighbors=${avg_neighbors} epochs=${epochs} max_steps=${MAX_STEPS} $(date)" \
  | tee -a "${molecule_root}/status.log"

DATA_ROOT="${DATA_ROOT}" \
DATASETS="${MOLECULE}" \
AVG_NEIGHBORS="${avg_neighbors}" \
SEEDS="${SEED}" \
MODES="${MODE}" \
EPOCHS="${epochs}" \
MAX_STEPS="${MAX_STEPS}" \
BATCH_SIZE="${BATCH_SIZE}" \
R_MAX=5.0 \
LR=0.001 \
MIN_LR=1e-6 \
LR_SCHEDULER=cosine \
ENERGY_WEIGHT=1 \
FORCE_WEIGHT="${FORCE_WEIGHT}" \
PHASE_COHERENCE_INIT="${COHERENCE_INIT}" \
TRAIN_MAKEFX_COMPILE=1 \
MAKEFX_BUCKETS=4 \
MAKEFX_MAX_SLOTS=8 \
PARALLEL_JOBS=1 \
MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUT_ROOT="${molecule_root}/train" \
bash "${RUN_MATRIX}" >"${molecule_root}/train_driver.log" 2>&1

# shellcheck disable=SC1090
source "${molecule_root}/train/metadata/${MOLECULE}.env"
DATA_DIR="${data_dir}" \
CHECKPOINT_DIR="${molecule_root}/train/checkpoints" \
OUT_DIR="${molecule_root}/official_test" \
AVG_NEIGHBORS="${avg_neighbors}" \
E0_KEYS="${E0_KEYS}" \
E0_VALUES="${E0_VALS}" \
BATCH_SIZE="${BATCH_SIZE}" \
FORCE_WEIGHT="${FORCE_WEIGHT}" \
PHASE_COHERENCE_INIT="${COHERENCE_INIT}" \
MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${EVALUATOR}" >"${molecule_root}/test_driver.log" 2>&1

touch "${molecule_root}/.complete"
echo "END ${MOLECULE} rc=0 $(date)" | tee -a "${molecule_root}/status.log"
