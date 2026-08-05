#!/usr/bin/env bash
#SBATCH --job-name=chorus_t1x_d3
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_depth3_scope_20260801/slurm-%x-%j.log

set -euo pipefail

MODE="${MODE:?submit with MODE=phaseoff, chorus, or persistent}"
case "${MODE}" in
  phaseoff|chorus|persistent) ;;
  *) echo "MODE must be phaseoff, chorus, or persistent, got ${MODE}" >&2; exit 2 ;;
esac

SEED="${SEED:-20260616}"
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
ROOT_BASE="${ROOT_BASE:-/public/home/sps-xia/rxlin/chorus_depth3_scope_20260801}"
SYSTEMS="${SYSTEMS:-t1x}"
case "${MODE}" in
  phaseoff)
    TAG="off"
    ROOT="${ROOT_BASE}/${TAG}_c128_l2_corr3_depth3_seed${SEED}"
    ;;
  chorus)
    TAG="final"
    ROOT="${ROOT_BASE}/${TAG}_c128_l2_corr3_r16_depth3_seed${SEED}"
    ;;
  persistent)
    TAG="persistent"
    ROOT="${ROOT_BASE}/${TAG}_c128_l2_corr3_r16_depth3_seed${SEED}"
    ;;
esac
STATUS="${ROOT}/queue_status.log"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT_BASE}" "${ROOT}"

{
  echo "START $(date --iso-8601=seconds)"
  echo "JOB_ID=${SLURM_JOB_ID:-}"
  echo "MODE=${MODE}"
  echo "SEED=${SEED}"
  echo "SYSTEMS=${SYSTEMS}"
  echo "CONFIG=C128 lmax2 maxL2 interactions3 correlation3 rank16 strict-float32"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee -a "${STATUS}"

REPO="${REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
ROOT="${ROOT}" \
STATUS="${STATUS}" \
MODE="${MODE}" \
SEED="${SEED}" \
CHANNELS=128 \
RANK=16 \
NUM_INTERACTIONS=3 \
SYSTEMS="${SYSTEMS}" \
T1X_DATA_DIR="${DATA_ROOT}/transition1x/chorus_reaction_id_50k_seed20260616" \
MAL_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal" \
STI_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti" \
BUCKY_DATA_DIR="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed" \
THREE_BPA_DATA_DIR="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5" \
bash "${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"

touch "${ROOT}/ALL_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)" | tee -a "${STATUS}"
