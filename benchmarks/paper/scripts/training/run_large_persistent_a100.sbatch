#!/usr/bin/env bash
#SBATCH --job-name=chorus_persist_lg
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_persistent_runs/large_c128_r16_20260730/slurm-%x-%j.log

set -euo pipefail

GROUP="${GROUP:?submit with GROUP=mal_bucky or sti_3bpa}"
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
ROOT_BASE="${ROOT_BASE:-/public/home/sps-xia/rxlin/chorus_persistent_runs/large_c128_r16_20260730}"

case "${GROUP}" in
  mal_bucky)
    SYSTEMS="mal,bucky"
    ;;
  sti_3bpa)
    SYSTEMS="sti,3bpa"
    ;;
  *)
    echo "unsupported GROUP=${GROUP}" >&2
    exit 2
    ;;
esac

mkdir -p "${ROOT_BASE}/${GROUP}"
{
  echo "START $(date --iso-8601=seconds)"
  echo "HOST $(hostname)"
  echo "JOB_ID=${SLURM_JOB_ID:-}"
  echo "GROUP=${GROUP}"
  echo "SYSTEMS=${SYSTEMS}"
  echo "CONFIG=C128 lmax2 maxL2 interactions2 correlation3 rank16 persistent"
  echo "TF32=disabled EMA=disabled dtype=float32"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "${ROOT_BASE}/${GROUP}/protocol.log"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

REPO="${REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
ROOT="${ROOT_BASE}/${GROUP}" \
MODE=persistent \
CHANNELS=128 \
RANK=16 \
SYSTEMS="${SYSTEMS}" \
MAL_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal" \
STI_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti" \
BUCKY_DATA_DIR="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed" \
THREE_BPA_DATA_DIR="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5" \
bash "${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"

touch "${ROOT_BASE}/${GROUP}/SBATCH_ALL_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)" | tee -a "${ROOT_BASE}/${GROUP}/protocol.log"
