#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/t1x_diagonal_multiseed_c128_r16_20260731}"
RUNNER="${REPO}/benchmarks/paper/scripts/training/run_t1x_diagonal_large.sh"
SEEDS="${SEEDS:-20260617 20260618}"

mkdir -p "${ROOT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

for seed in ${SEEDS}; do
  out="${ROOT}/seed${seed}"
  if [[ -f "${out}/DONE" ]]; then
    echo "REUSE seed=${seed} $(date -Is)" | tee -a "${ROOT}/queue_status.log"
    continue
  fi
  echo "START seed=${seed} $(date -Is)" | tee -a "${ROOT}/queue_status.log"
  REPO="${REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
  OUT_ROOT="${out}" \
  SEED="${seed}" \
  bash "${RUNNER}" >"${out}.driver.log" 2>&1
  echo "DONE seed=${seed} $(date -Is)" | tee -a "${ROOT}/queue_status.log"
done

touch "${ROOT}/ALL_DONE"
echo "ALL_DONE $(date -Is)" | tee -a "${ROOT}/queue_status.log"
