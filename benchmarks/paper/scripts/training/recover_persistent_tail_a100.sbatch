#!/usr/bin/env bash
#SBATCH --job-name=chorus_persist_tail
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_multiseed_formal_20260731/slurm-%x-%j.log

set -euo pipefail

SEED="${SEED:?submit with SEED=20260617 or SEED=20260618}"
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
ROOT_BASE="${ROOT_BASE:-/public/home/sps-xia/rxlin/chorus_multiseed_formal_20260731}"
ROOT="${ROOT_BASE}/seed${SEED}/persistent_r16"
RUNNER="${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

env REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
  ROOT="${ROOT}" STATUS="${ROOT}/queue_status.log" MODE=persistent \
  SEED="${SEED}" CHANNELS=128 RANK=16 NUM_INTERACTIONS=2 \
  SYSTEMS=bucky,3bpa \
  T1X_DATA_DIR="${DATA_ROOT}/transition1x/chorus_reaction_id_50k_seed20260616" \
  MAL_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal" \
  STI_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti" \
  BUCKY_DATA_DIR="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed" \
  THREE_BPA_DATA_DIR="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5" \
  bash "${RUNNER}"

touch "${ROOT}/ALL_DONE" "${ROOT_BASE}/seed${SEED}/ALL_DONE"
echo "RECOVERY_DONE seed=${SEED} $(date --iso-8601=seconds)"
