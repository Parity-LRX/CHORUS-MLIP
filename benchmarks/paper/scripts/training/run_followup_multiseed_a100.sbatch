#!/usr/bin/env bash
#SBATCH --job-name=chorus_ms_follow
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_multiseed_followup_20260731/slurm-%x-%j.log

set -euo pipefail

SEED="${SEED:?submit with SEED=20260617 or SEED=20260618}"
CHORUS_REPO="${CHORUS_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
NEQUIP_REPO="${NEQUIP_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_multiseed_followup_20260731/seed${SEED}}"
CHORUS_RUNNER="${CHORUS_REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"
NEQUIP_RUNNER="${NEQUIP_REPO}/scripts/run_extended_variant.sh"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${ROOT}/queue_status.log"
}

run_widened_ictc() {
  local out="${ROOT}/mace_ictc_c140_phaseoff"
  if [[ -f "${out}/ALL_DONE" ]]; then
    mark "REUSE_MACE_ICTC_C140"
    return
  fi
  mark "START_MACE_ICTC_C140"
  env REPO="${CHORUS_REPO}" PYTHON_BIN="${PYTHON_BIN}" \
    MACE_TORCH_PATH="${MACE_TORCH_PATH}" ROOT="${out}" \
    MODE=phaseoff SEED="${SEED}" CHANNELS=140 RANK=16 \
    SYSTEMS=t1x,mal,sti,bucky,3bpa \
    T1X_DATA_DIR="${DATA_ROOT}/transition1x/chorus_reaction_id_50k_seed20260616" \
    MAL_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal" \
    STI_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti" \
    BUCKY_DATA_DIR="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed" \
    THREE_BPA_DATA_DIR="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5" \
    bash "${CHORUS_RUNNER}"
  mark "DONE_MACE_ICTC_C140"
}

run_nequip_variant() {
  local backend="$1" chorus="$2" channels="$3"
  local tag="${backend}_c${channels}_chorus_${chorus}"
  local out="${ROOT}/nequip/${tag}"
  if [[ -f "${out}/ALL_DONE" ]]; then
    mark "REUSE_NEQUIP_${tag}"
    return
  fi
  mkdir -p "${out}"
  mark "START_NEQUIP_${tag}"
  env REPO="${NEQUIP_REPO}" OPERATOR_REPO="${CHORUS_REPO}" RUN_ROOT="${out}" \
    DATA_ROOT="${DATA_ROOT}" DATASETS="transition1x xxmd_mal xxmd_sti md22_buckyball 3bpa" \
    CHANNELS="${channels}" CHORUS_RANK=16 SEED="${SEED}" \
    bash "${NEQUIP_RUNNER}" "${backend}" "${chorus}" "${CUDA_VISIBLE_DEVICES}" \
    >"${out}/driver.log" 2>&1
  touch "${out}/ALL_DONE"
  mark "DONE_NEQUIP_${tag}"
}

mark "START seed=${SEED}"
run_widened_ictc
for backend in e3nn ictc; do
  run_nequip_variant "${backend}" off 64
  run_nequip_variant "${backend}" on 64
  run_nequip_variant "${backend}" off 72
done
touch "${ROOT}/ALL_DONE"
mark "ALL_FOLLOWUP_DONE seed=${SEED}"
