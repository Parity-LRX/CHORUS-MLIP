#!/usr/bin/env bash
#SBATCH --job-name=chorus_ms_formal
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_multiseed_formal_20260731/slurm-%x-%j.log

set -euo pipefail

SEED="${SEED:?submit with SEED=20260617 or SEED=20260618}"
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
ROOT_BASE="${ROOT_BASE:-/public/home/sps-xia/rxlin/chorus_multiseed_formal_20260731}"
STATUS="${ROOT_BASE}/seed${SEED}/queue_status.log"
RUNNER="${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"

mkdir -p "${ROOT_BASE}/seed${SEED}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

{
  echo "START $(date --iso-8601=seconds)"
  echo "HOST $(hostname)"
  echo "JOB_ID=${SLURM_JOB_ID:-}"
  echo "SEED=${SEED}"
  echo "MATRIX=phaseoff-r16,final-r8,final-r16,final-r32,persistent-r16"
  echo "SYSTEMS=t1x-50k,mal,sti,bucky,3bpa"
  echo "CONFIG=C128 lmax2 maxL2 interactions2 correlation3 strict-float32"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee -a "${STATUS}"

run_block() {
  local mode="$1"
  local rank="$2"
  local tag="$3"
  local root="${ROOT_BASE}/seed${SEED}/${tag}"

  if [[ -f "${root}/ALL_DONE" ]]; then
    echo "REUSE ${tag} $(date --iso-8601=seconds)" | tee -a "${STATUS}"
    return
  fi

  echo "START ${tag} $(date --iso-8601=seconds)" | tee -a "${STATUS}"
  REPO="${REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
  ROOT="${root}" \
  STATUS="${root}/queue_status.log" \
  MODE="${mode}" \
  SEED="${SEED}" \
  CHANNELS=128 \
  RANK="${rank}" \
  SYSTEMS=t1x,mal,sti,bucky,3bpa \
  T1X_DATA_DIR="${DATA_ROOT}/transition1x/chorus_reaction_id_50k_seed20260616" \
  MAL_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal" \
  STI_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti" \
  BUCKY_DATA_DIR="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed" \
  THREE_BPA_DATA_DIR="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5" \
  bash "${RUNNER}"
  echo "DONE ${tag} $(date --iso-8601=seconds)" | tee -a "${STATUS}"
}

run_block phaseoff 16 phaseoff_r16
run_block chorus 8 final_r8
run_block chorus 16 final_r16
run_block chorus 32 final_r32
run_block persistent 16 persistent_r16

touch "${ROOT_BASE}/seed${SEED}/ALL_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)" | tee -a "${STATUS}"
