#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/external_multiseed_formal_20260731}"
SEEDS="${SEEDS:-20260617 20260618}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
FSCETP="${FSCETP:-/home/ylzhang/micromamba/envs/FSCETP}"
TACE_REPO="${TACE_REPO:-/home/ylzhang/tace_chorus_benchmark}"
TACE_BIN="${TACE_BIN:-/home/ylzhang/tace_chorus_venv/bin}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${ROOT}/queue_status.log"
}

run_stage() {
  local marker="$1"
  shift
  local done="${ROOT}/markers/${marker}.DONE"
  mkdir -p "$(dirname "${done}")"
  if [[ -f "${done}" ]]; then
    mark "REUSE_${marker}"
    return
  fi
  mark "START_${marker}"
  "$@"
  touch "${done}"
  mark "DONE_${marker}"
}

for seed in ${SEEDS}; do
  seed_root="${ROOT}/seed${seed}"
  mkdir -p "${seed_root}"

  for channels in 32 48; do
    run_stage "seed${seed}_dpa4_c${channels}" \
      env REPO="${REPO}" DPA_ENV="${DPA_ENV}" \
      ROOT="${seed_root}/dpa4_c${channels}" STATUS="${seed_root}/dpa4_c${channels}/status.log" \
      CHANNELS="${channels}" MIXING_LAYERS=3 SEED="${seed}" \
      USE_COMPILE=1 USE_AMP=0 XXMD_SYSTEMS="mal sti" \
      WAIT_SCREENS="__none__" MAX_BACKGROUND_GPU_MEMORY_MB=4096 \
      bash "${REPO}/benchmarks/paper/scripts/training/queue_dpa4_c48_scaling.sh"
  done

  for channels in 36 48; do
    run_stage "seed${seed}_tece_c${channels}_main" \
      env REPO="${REPO}" SEED="${seed}" CHANNELS="${channels}" \
      SYSTEMS="mal sti t1x bucky" WAIT_SCREENS="__none__" \
      ROOT="${seed_root}/tece_c${channels}_driver" \
      XXMD_RUN_ROOT="${seed_root}/tece_c${channels}/xxmd" \
      OTHER_RUN_ROOT="${seed_root}/tece_c${channels}/other" \
      bash "${REPO}/benchmarks/paper/scripts/training/queue_tece_c48_recommended.sh"
  done

  run_stage "seed${seed}_3bpa_tece_and_native" \
    env REPO="${REPO}" FSCETP="${FSCETP}" DPA_ENV="${DPA_ENV}" \
    TACE_REPO="${TACE_REPO}" TACE_BIN="${TACE_BIN}" \
    MACE_TORCH_PATH="${MACE_TORCH_PATH}" SEED="${seed}" \
    ROOT="${seed_root}/3bpa_external" WAIT_SCREEN="__none__" \
    MODELS="tece36 tece48 native_mace" \
    bash "${REPO}/benchmarks/paper/scripts/training/queue_3bpa_external_models.sh"

  run_stage "seed${seed}_native_mace_main" \
    env REPO="${REPO}" PYTHON_BIN="${FSCETP}/bin/python" \
    MACE_TORCH_PATH="${MACE_TORCH_PATH}" SEED="${seed}" \
    ROOT="${seed_root}/native_mace" WAIT_SCREENS="__none__" \
    SYSTEMS="t1x xxmd_mal xxmd_sti buckyball" \
    bash "${REPO}/benchmarks/paper/scripts/training/queue_native_mace_and_ictc_baselines.sh"

  touch "${seed_root}/ALL_DONE"
  mark "ALL_SEED_${seed}_DONE"
done

touch "${ROOT}/ALL_DONE"
mark "ALL_EXTERNAL_MULTISEED_DONE"
