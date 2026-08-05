#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
RESULTS="${RESULTS:-${REPO}/benchmarks/paper/results/phase}"
TRANSITION_RUNNER="${REPO}/benchmarks/paper/scripts/training/run_transition1x_reaction_id.sh"
XXMD_RUNNER="${REPO}/benchmarks/paper/scripts/training/run_xxmd_dft_screen_queued.sh"
STATUS="${RESULTS}/after_md22_multiseed_20260723.status.log"
MAIN_MODES="ictc_bridge_u_makefx,ictc_phase_full_l_softplus_makefx,ictc_phase_diagonal_full_l_makefx,ictc_attention_makefx"

while screen -ls 2>/dev/null | grep -q '[.]chorus_md22_priority[[:space:]]'; do
  sleep 30
done

# Seed 17 baseline completed before MD22 was promoted. Resume only its missing
# mechanisms; then run all four mechanisms for seed 18.
OUT17="${RESULTS}/transition1x_reaction_id_50k_steps100000_seed20260617_20260723"
OUT18="${RESULTS}/transition1x_reaction_id_50k_steps100000_seed20260618_20260723"
echo "START transition1x seed17 remainder $(date)" | tee -a "${STATUS}"
OUT_ROOT="${OUT17}" SEED=20260617 \
  MODES=ictc_phase_full_l_softplus_makefx,ictc_phase_diagonal_full_l_makefx,ictc_attention_makefx \
  bash "${TRANSITION_RUNNER}" >"${OUT17}.resume_driver.log" 2>&1
echo "END transition1x seed17 $(date)" | tee -a "${STATUS}"
echo "START transition1x seed18 $(date)" | tee -a "${STATUS}"
OUT_ROOT="${OUT18}" SEED=20260618 MODES="${MAIN_MODES}" \
  bash "${TRANSITION_RUNNER}" >"${OUT18}.driver.log" 2>&1
echo "END transition1x seed18 $(date)" | tee -a "${STATUS}"

run_xxmd() {
  local seed="$1" molecules="$2" steps="$3" tag="$4"
  local out="${RESULTS}/xxmd_multiseed_main_20260723/seed${seed}_${tag}_steps${steps}"
  mkdir -p "$(dirname "${out}")"
  echo "START xxmd seed=${seed} molecules=${molecules} steps=${steps} $(date)" | tee -a "${STATUS}"
  OUT_ROOT="${out}" MOLECULES="${molecules}" SEED="${seed}" \
    MAX_STEPS="${steps}" MODES="${MAIN_MODES}" WAIT_SCREENS=__none__ \
    bash "${XXMD_RUNNER}" >"${out}.driver.log" 2>&1
  echo "END xxmd seed=${seed} molecules=${molecules} steps=${steps} $(date)" | tee -a "${STATUS}"
}

run_xxmd 20260617 azo,sti 45000 azo_sti
run_xxmd 20260618 azo,sti 45000 azo_sti
run_xxmd 20260617 dia,mal 150000 dia_mal
run_xxmd 20260618 dia,mal 150000 dia_mal
echo "ALL_OK $(date)" | tee -a "${STATUS}"
