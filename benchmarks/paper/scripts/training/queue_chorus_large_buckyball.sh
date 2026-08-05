#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
OUT="${OUT:-/home/ylzhang/chorus_runs/chorus_large_c128_l2_corr3_buckyball_20260724}"
STATUS="${STATUS:-${OUT}/queue_status.log}"

mkdir -p "${OUT}"
mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

gpu_is_busy() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[0-9]+$'
}

mark "WAIT_DPA4_MIX2"
while screen -list | grep -q '\.dpa4_mix2'; do
  sleep 30
done

mark "WAIT_GPU_IDLE"
while gpu_is_busy; do
  sleep 30
done

# Force strict IEEE FP32 matmul behavior for the capacity-matched comparison.
export NVIDIA_TF32_OVERRIDE=0

mark "START_CHORUS_C128_L2_CORR3"
MACE_ICTC_REPO="${REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
DATA_DIR=/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
OUT_ROOT="${OUT}" \
DATASET_TAG=buckyball_catcher \
JOB_PREFIX=md22_buckyball_catcher_large_c128_l2_corr3 \
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
AVG_NEIGHBORS=30.3929 \
READOUT_HIDDEN=64 \
E0_KEYS=1,6 \
E0_VALUES=-230.09867339,-986.13717166 \
  bash "${REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
mark "DONE_CHORUS_C128_L2_CORR3"
