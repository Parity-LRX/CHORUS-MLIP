#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/buckyball_fair_r5_noema_20260724}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"

mkdir -p "${ROOT}"
mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

gpu_is_busy() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[0-9]+$'
}

mark "WAIT_GPU_IDLE"
while gpu_is_busy; do
  sleep 30
done

export NVIDIA_TF32_OVERRIDE=0

mark "START_DPA4_C13_MIX2_R5_NOEMA"
OUT="${ROOT}/dpa4_c13_mix2" \
STEPS=45000 \
CHANNELS=13 \
MIXING_LAYERS=2 \
RCUT=5.0 \
ENABLE_EMA=false \
REPO="${REPO}" \
DPA_ENV="${DPA_ENV}" \
  bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_master_buckyball.sh"
mark "DONE_DPA4_C13_MIX2_R5_NOEMA"

mark "START_DPA4_C32_MIX3_R5_NOEMA"
OUT="${ROOT}/dpa4_c32_mix3" \
STEPS=45000 \
CHANNELS=32 \
MIXING_LAYERS=3 \
RCUT=5.0 \
ENABLE_EMA=false \
REPO="${REPO}" \
DPA_ENV="${DPA_ENV}" \
  bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_master_buckyball.sh"
mark "DONE_DPA4_C32_MIX3_R5_NOEMA"

mark "START_CHORUS_C64_L2_CORR3_RANK8_R5_NOEMA"
MACE_ICTC_REPO="${REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
DATA_DIR=/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
OUT_ROOT="${ROOT}/chorus_c64_l2_corr3_rank8" \
DATASET_TAG=buckyball_catcher \
JOB_PREFIX=md22_buckyball_catcher_c64_l2_corr3_rank8 \
MODES=ictc_phase_full_l_nonlinear_makefx \
SEED=20260616 \
EPOCHS=300 \
MAX_STEPS=45000 \
BATCH_SIZE=4 \
CHANNELS=64 \
HIDDEN_LMAX=2 \
MAX_ELL=2 \
NUM_INTERACTIONS=2 \
CORRELATION=3 \
PHASE_DENSITY_RANK=8 \
AVG_NEIGHBORS=30.3929 \
READOUT_HIDDEN=64 \
E0_KEYS=1,6 \
E0_VALUES=-230.09867339,-986.13717166 \
  bash "${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
mark "DONE_CHORUS_C64_L2_CORR3_RANK8_R5_NOEMA"

mark "START_CHORUS_C128_L2_CORR3_RANK8_R5_NOEMA"
MACE_ICTC_REPO="${REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
DATA_DIR=/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
OUT_ROOT="${ROOT}/chorus_c128_l2_corr3_rank8" \
DATASET_TAG=buckyball_catcher \
JOB_PREFIX=md22_buckyball_catcher_c128_l2_corr3_rank8 \
MODES=ictc_phase_full_l_nonlinear_makefx \
SEED=20260616 \
EPOCHS=300 \
MAX_STEPS=45000 \
BATCH_SIZE=4 \
CHANNELS=128 \
HIDDEN_LMAX=2 \
MAX_ELL=2 \
NUM_INTERACTIONS=2 \
CORRELATION=3 \
PHASE_DENSITY_RANK=8 \
AVG_NEIGHBORS=30.3929 \
READOUT_HIDDEN=64 \
E0_KEYS=1,6 \
E0_VALUES=-230.09867339,-986.13717166 \
  bash "${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
mark "DONE_CHORUS_C128_L2_CORR3_RANK8_R5_NOEMA"
mark "ALL_DONE"
