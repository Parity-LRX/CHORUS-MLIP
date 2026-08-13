#!/usr/bin/env bash
set -euo pipefail

# Table 4 angular-resolution controls.  Relative to the published C128/R16,
# two-interaction protocol, only hidden_lmax=max_ell=L changes.  L=2 is the
# existing reference and is deliberately not repeated.

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
RUNNER="${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/table4_lmax_controls_c128_2layer_r16_20260813}"
LMAX_VALUES="${LMAX_VALUES:-1 3}"
MODES="${MODES:-phaseoff chorus persistent}"
SYSTEMS="${SYSTEMS:-t1x,mal,sti,bucky,3bpa}"
STATUS="${ROOT}/queue_status.log"

mkdir -p "${ROOT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

mark() {
  printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${STATUS}"
}

gpu_is_busy() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[0-9]+$'
}

mark "WAIT_GPU_IDLE"
while gpu_is_busy; do
  sleep 30
done

for lmax in ${LMAX_VALUES}; do
  for mode in ${MODES}; do
    run_root="${ROOT}/L${lmax}/${mode}"
    mkdir -p "$(dirname "${run_root}")"
    if [[ -f "${run_root}/ALL_DONE" ]]; then
      mark "REUSE_L${lmax}_${mode}"
      continue
    fi
    mark "START_L${lmax}_${mode}"
    REPO="${REPO}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
    ROOT="${run_root}" \
    SEED=20260616 CHANNELS=128 RANK=16 NUM_INTERACTIONS=2 \
    HIDDEN_LMAX="${lmax}" MAX_ELL="${lmax}" \
    MODE="${mode}" SYSTEMS="${SYSTEMS}" PHASE_CONTEXT=content \
      bash "${RUNNER}" >"${run_root}.driver.log" 2>&1
    mark "DONE_L${lmax}_${mode}"
  done
done

touch "${ROOT}/ALL_DONE"
mark "ALL_TABLE4_LMAX_CONTROLS_DONE"
