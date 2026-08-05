#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
RESULTS="${RESULTS:-${REPO}/benchmarks/paper/results/phase}"
PREP_ROOT="${PREP_ROOT:-/home/ylzhang/lrx/md22/all_lowdata600_test1000_20260723}"
RUNNER="${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
EVALUATOR="${REPO}/benchmarks/paper/scripts/training/evaluate_md22_heldout.sh"
STATUS="${RESULTS}/md22_all_multiseed_priority_20260723.status.log"
MODES="ictc_phase_full_l_softplus_makefx,ictc_phase_diagonal_full_l_makefx,ictc_attention_makefx,ictc_bridge_u_makefx"
WAIT_SCREEN="${WAIT_SCREEN:-chorus_after_md22}"

while screen -ls 2>/dev/null | grep -q "[.]${WAIT_SCREEN}[[:space:]]"; do
  sleep 5
done

# Complete the existing Buckyball Catcher campaign with the manuscript's
# standard ungated full-pair operator. The other three modes already have
# seeds 16--18 under the identical 600/600 training protocol.
BCC_DATA="/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed"
BCC_ROOT="${RESULTS}/md22_buckyball_standard_full_multiseed_20260723"
mkdir -p "${BCC_ROOT}"
for seed in 20260616 20260617 20260618; do
  out="${BCC_ROOT}/seed${seed}"
  if grep -q "ALL_OK" "${out}/heldout_test/status.log" 2>/dev/null; then
    echo "SKIP buckyball_catcher full seed=${seed}" | tee -a "${STATUS}"
    continue
  fi
  echo "START buckyball_catcher full seed=${seed} $(date)" | tee -a "${STATUS}"
  DATASET_TAG=buckyball_catcher DATA_DIR="${BCC_DATA}" OUT_ROOT="${out}" \
    MODES=ictc_phase_full_l_softplus_makefx SEED="${seed}" EPOCHS=300 \
    BATCH_SIZE=4 AVG_NEIGHBORS=30.3929 \
    E0_KEYS=1,6 E0_VALUES=-230.09867339,-986.13717166 \
    bash "${RUNNER}" >"${out}.driver.log" 2>&1
  CHECKPOINT_DIR="${out}/checkpoints" OUT_DIR="${out}/heldout_test" \
    DATA_DIR="${BCC_DATA}" BATCH_SIZE=4 AVG_NEIGHBORS=30.3929 \
    E0_KEYS=1,6 E0_VALUES=-230.09867339,-986.13717166 SEED="${seed}" \
    bash "${EVALUATOR}" >"${out}.test_driver.log" 2>&1
  echo "END buckyball_catcher full seed=${seed} $(date)" | tee -a "${STATUS}"
done

while screen -ls 2>/dev/null | grep -q '[.]chorus_md22_prepare[[:space:]]'; do
  echo "WAIT md22 preparation $(date)" >>"${STATUS}"
  sleep 30
done

systems=(Ac_Ala3_NHMe DHA AT_AT AT_AT_CG_CG stachyose double_walled_nanotube)
CAMPAIGN="${RESULTS}/md22_all_systems_multiseed_20260723"
for system in "${systems[@]}"; do
  data_root="${PREP_ROOT}/${system}"
  data_dir="${data_root}/processed"
  if [[ ! -f "${data_root}/.complete" ]]; then
    echo "ERROR preparation missing for ${system}" | tee -a "${STATUS}"
    exit 3
  fi
  # shellcheck disable=SC1090
  source "${data_root}/training.env"
  for seed in 20260616 20260617 20260618; do
    out="${CAMPAIGN}/${system}/seed${seed}"
    mkdir -p "$(dirname "${out}")"
    if grep -q "ALL_OK" "${out}/heldout_test/status.log" 2>/dev/null; then
      echo "SKIP ${system} seed=${seed}" | tee -a "${STATUS}"
      continue
    fi
    echo "START ${system} seed=${seed} $(date)" | tee -a "${STATUS}"
    DATASET_TAG="${system}" DATA_DIR="${data_dir}" OUT_ROOT="${out}" \
      MODES="${MODES}" SEED="${seed}" EPOCHS=300 BATCH_SIZE=4 \
      AVG_NEIGHBORS="${AVG_NEIGHBORS}" E0_KEYS="${E0_KEYS}" E0_VALUES="${E0_VALUES}" \
      bash "${RUNNER}" >"${out}.driver.log" 2>&1
    CHECKPOINT_DIR="${out}/checkpoints" OUT_DIR="${out}/heldout_test" \
      DATA_DIR="${data_dir}" BATCH_SIZE=4 AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
      E0_KEYS="${E0_KEYS}" E0_VALUES="${E0_VALUES}" SEED="${seed}" \
      bash "${EVALUATOR}" >"${out}.test_driver.log" 2>&1
    echo "END ${system} seed=${seed} $(date)" | tee -a "${STATUS}"
  done
done

echo "ALL_MD22_OK $(date)" | tee -a "${STATUS}"
