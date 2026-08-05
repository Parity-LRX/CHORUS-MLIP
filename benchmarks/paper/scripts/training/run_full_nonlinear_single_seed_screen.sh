#!/usr/bin/env bash
set -euo pipefail

# Strict single-seed screen for deciding whether full-nonlinear should replace
# the plain Full Hermitian density as the default CHORUS configuration.
#
# Existing plain-Full runs are reused.  Every new non-Transition1x run changes
# only --phase-density-pairs full -> full-nonlinear and the output path.

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
OUT="${OUT:-/home/ylzhang/chorus_runs/full_nonlinear_single_seed_screen_20260724}"
SEED="${SEED:-20260616}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
mkdir -p "${OUT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${OUT}/status.log"
}

run_saved_full_as_nonlinear() {
  local dataset="$1"
  local source_command="$2"
  local run_dir="${OUT}/${dataset}"
  local checkpoint="${run_dir}/${dataset}_full_nonlinear_seed${SEED}.pth"
  local log="${run_dir}/train.log"
  local command

  mkdir -p "${run_dir}"
  command="$(tail -n 1 "${source_command}")"
  if [[ "${command}" != *"--phase-density-pairs full "* ]]; then
    echo "source command is not a plain-Full run: ${source_command}" >&2
    exit 2
  fi
  command="${command/--phase-density-pairs full /--phase-density-pairs full-nonlinear }"
  command="$(printf '%s\n' "${command}" |
    sed -E "s#--checkpoint [^ ]+#--checkpoint ${checkpoint}#")"
  printf '%s\n' "${command}" >"${run_dir}/command.sh"

  mark "START ${dataset} full-nonlinear"
  eval "${command}" >"${log}" 2>&1
  mark "DONE ${dataset} full-nonlinear"
}

# 1. The decisive Transition1x pair.  This reproduces the enhanced model's
# C48/L2/rms-forces configuration and changes only full-nonlinear -> full.
mark "START transition1x matched plain-full"
env \
  REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
  MODE=full RANK=8 CHANNELS=48 LMAX=2 MAX_ELL=2 CORRELATION=2 \
  PHASE_HEADS=1 READOUT_HIDDEN_CHANNELS=64 \
  ELEMENT_ENERGY_CORRECTION=1 \
  SCALING=rms_forces_scaling ATOMIC_INTER_SCALE=0.7642790079116821 \
  NO_ATOMIC_INTER_SHIFT=1 PHASE_SCALE_INIT=0.05 \
  MAX_STEPS=100000 EPOCHS=33 SEED="${SEED}" \
  OUT_ROOT="${OUT}/transition1x_plain_full_c48_l2" \
  bash "${REPO}/benchmarks/paper/scripts/training/run_transition1x_chorus_enhanced.sh"
mark "DONE transition1x matched plain-full"

# 2. A strong plain-Full system: detect degradation before expanding.
mark "START buckyball full-nonlinear"
env \
  MACE_ICTC_REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" \
  MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
  DATA_DIR=/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
  OUT_ROOT="${OUT}/buckyball" \
  DATASET_TAG=buckyball_catcher \
  MODES=ictc_phase_full_l_nonlinear_makefx \
  SEED="${SEED}" EPOCHS=300 BATCH_SIZE=4 \
  AVG_NEIGHBORS=30.3929 \
  E0_KEYS=1,6 E0_VALUES=-230.09867339,-986.13717166 \
  bash "${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
mark "DONE buckyball full-nonlinear"

# 3. The main negative-transfer system.
run_saved_full_as_nonlinear \
  sti \
  "${REPO}/benchmarks/paper/results/phase/xxmd_dft_temporal_steps45000_20260721/sti/train/commands/sti_ictc_phase_full_l_softplus_makefx_seed20260616_epochs57.sh"

# 4. A completed three-seed MD22 system with a small Full-vs-diagonal gap.
mark "START ac_ala3_nhme full-nonlinear"
env \
  MACE_ICTC_REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" \
  MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
  DATA_DIR=/home/ylzhang/lrx/md22/all_lowdata600_test1000_20260723/Ac_Ala3_NHMe/processed \
  OUT_ROOT="${OUT}/ac_ala3_nhme" \
  DATASET_TAG=Ac_Ala3_NHMe JOB_PREFIX=md22_Ac_Ala3_NHMe \
  MODES=ictc_phase_full_l_nonlinear_makefx \
  SEED="${SEED}" EPOCHS=300 BATCH_SIZE=4 \
  AVG_NEIGHBORS=21.4033 \
  E0_KEYS=1,6,7,8 \
  E0_VALUES=-897.1504208695164,-489.3547750197357,-163.1182583399127,-163.11825833991253 \
  bash "${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
mark "DONE ac_ala3_nhme full-nonlinear"

# 5. The converged 150k-step xxMD case is intentionally last so that the
# higher-value short screens finish first if the queue is interrupted.
run_saved_full_as_nonlinear \
  dia \
  "${REPO}/benchmarks/paper/results/phase/xxmd_dia_mal_steps150000_v2_20260722/dia/train/commands/dia_ictc_phase_full_l_softplus_makefx_seed20260616_epochs194.sh"

mark "ALL_DONE"
