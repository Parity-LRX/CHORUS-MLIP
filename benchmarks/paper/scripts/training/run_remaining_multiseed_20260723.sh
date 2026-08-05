#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
RESULTS="${RESULTS:-${REPO}/benchmarks/paper/results/phase}"
WAIT_SCREEN="${WAIT_SCREEN:-chorus_transition1x_multiseed}"
MD22_RUNNER="${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
MD22_EVALUATOR="${REPO}/benchmarks/paper/scripts/training/evaluate_md22_heldout.sh"
XXMD_RUNNER="${REPO}/benchmarks/paper/scripts/training/run_xxmd_dft_screen_queued.sh"
MD22_DATA="/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed"
MODES="ictc_bridge_u_makefx,ictc_phase_full_l_softplus_makefx,ictc_phase_diagonal_full_l_makefx,ictc_attention_makefx"
STATUS="${RESULTS}/remaining_multiseed_20260723.status.log"

while screen -ls 2>/dev/null | grep -q "[.]${WAIT_SCREEN}[[:space:]]"; do
  echo "WAIT ${WAIT_SCREEN} $(date)" >>"${STATUS}"
  sleep 30
done

# MD22 already has three seeds for baseline, diagonal density, and attention.
# Complete the matched main table with the standard (ungated) full-pair CHORUS
# operator used by the manuscript and Transition1x campaign.
MD22_ROOT="${RESULTS}/md22_buckyball_standard_full_multiseed_20260723"
mkdir -p "${MD22_ROOT}"
for seed in 20260616 20260617 20260618; do
  out="${MD22_ROOT}/seed${seed}"
  echo "START md22 full seed=${seed} $(date)" | tee -a "${STATUS}"
  DATA_DIR="${MD22_DATA}" OUT_ROOT="${out}" \
    MODES=ictc_phase_full_l_softplus_makefx SEED="${seed}" \
    EPOCHS=300 BATCH_SIZE=4 AVG_NEIGHBORS=30.3929 \
    E0_KEYS=1,6 E0_VALUES=-230.09867339,-986.13717166 \
    bash "${MD22_RUNNER}" >"${out}.driver.log" 2>&1
  CHECKPOINT_DIR="${out}/checkpoints" OUT_DIR="${out}/heldout_test" \
    DATA_DIR="${MD22_DATA}" BATCH_SIZE=4 AVG_NEIGHBORS=30.3929 \
    E0_KEYS=1,6 E0_VALUES=-230.09867339,-986.13717166 \
    bash "${MD22_EVALUATOR}" >"${out}.test_driver.log" 2>&1
  echo "END md22 full seed=${seed} $(date)" | tee -a "${STATUS}"
done

run_xxmd() {
  local seed="$1"
  local molecules="$2"
  local steps="$3"
  local tag="$4"
  local out="${RESULTS}/xxmd_multiseed_main_20260723/seed${seed}_${tag}_steps${steps}"
  mkdir -p "$(dirname "${out}")"
  echo "START xxmd seed=${seed} molecules=${molecules} steps=${steps} $(date)" | tee -a "${STATUS}"
  OUT_ROOT="${out}" MOLECULES="${molecules}" SEED="${seed}" \
    MAX_STEPS="${steps}" MODES="${MODES}" WAIT_SCREENS=__none__ \
    bash "${XXMD_RUNNER}" >"${out}.driver.log" 2>&1
  echo "END xxmd seed=${seed} molecules=${molecules} steps=${steps} $(date)" | tee -a "${STATUS}"
}

# Preserve the final seed-16 budgets: Azo/STI use 45k steps; the harder
# Dia/Mal tasks use their completed 150k-step protocol.
run_xxmd 20260617 azo,sti 45000 azo_sti
run_xxmd 20260618 azo,sti 45000 azo_sti
run_xxmd 20260617 dia,mal 150000 dia_mal
run_xxmd 20260618 dia,mal 150000 dia_mal

echo "ALL_OK $(date)" | tee -a "${STATUS}"
