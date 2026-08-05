#!/usr/bin/env bash
#SBATCH --job-name=nequip_d2_off_final
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2-off-final-multiseed-%j.log

set -euo pipefail

SEED="${SEED:?submit with the required seed}"
REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
BASE_ROOT="${BASE_ROOT:-/public/home/sps-xia/rxlin/chorus_multiseed_followup_20260731/seed${SEED}/nequip}"
DATASETS="${DATASETS:-transition1x xxmd_mal xxmd_sti md22_buckyball 3bpa}"
MODES="${MODES:-off final}"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

run_variant() {
  local label="$1"
  local enabled="$2"
  local scope="$3"
  local run_root="${BASE_ROOT}/e3nn_c64_chorus_${label}_depth2"

  if [[ -f "${run_root}/ALL_DONE" ]]; then
    echo "REUSE ${label} seed=${SEED} ${run_root}"
    return
  fi

  mkdir -p "${run_root}"
  echo "START ${label} seed=${SEED} $(date --iso-8601=seconds)"
  env \
    REPO="${REPO}" \
    OPERATOR_REPO="${OPERATOR_REPO}" \
    DATA_ROOT="${DATA_ROOT}" \
    RUN_ROOT="${run_root}" \
    DATASETS="${DATASETS}" \
    CHANNELS=64 \
    CHORUS_RANK=16 \
    CHORUS_SCOPE="${scope}" \
    NUM_LAYERS=2 \
    SEED="${SEED}" \
    EVAL_BATCH_SIZE=25 \
    bash "${REPO}/scripts/run_extended_variant.sh" \
      e3nn "${enabled}" "${CUDA_VISIBLE_DEVICES}" \
      >"${run_root}/driver.log" 2>&1
  touch "${run_root}/ALL_DONE"
  echo "DONE ${label} seed=${SEED} $(date --iso-8601=seconds)"
}

for mode in ${MODES}; do
  case "${mode}" in
    off)
      run_variant off off final
      ;;
    final)
      run_variant final on final
      ;;
    *)
      echo "Unsupported mode: ${mode}" >&2
      exit 2
      ;;
  esac
done
