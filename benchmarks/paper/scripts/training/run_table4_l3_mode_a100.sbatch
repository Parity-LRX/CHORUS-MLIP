#!/usr/bin/env bash
#SBATCH --job-name=ch-l3-mode
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

set -euo pipefail

MODE="${MODE:?submit with MODE=chorus or MODE=persistent}"
case "${MODE}" in
  chorus|persistent) ;;
  *) echo "unsupported MODE=${MODE}" >&2; exit 2 ;;
esac

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-table4-l3-20260814}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
ROOT_BASE="${ROOT_BASE:-/public/home/sps-xia/rxlin/chorus_table4_lmax_controls_c128_2layer_r16_20260814/L3}"
RUN_ROOT="${ROOT_BASE}/${MODE}"
RUNNER="${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/slurm-${SLURM_JOB_ID:-manual}.log") 2>&1

echo "START $(date --iso-8601=seconds)"
echo "HOST=$(hostname) JOB_ID=${SLURM_JOB_ID:-} MODE=${MODE}"
echo "CONFIG=C128 lmax3 maxL3 interactions2 correlation3 rank16 ${MODE}"
echo "PROTOCOL=strict-float32 no-EMA no-TF32 MakeFX seed20260616"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

REPO="${REPO}" \
PYTHON_BIN="${PYTHON_BIN}" \
MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
ROOT="${RUN_ROOT}" \
MODE="${MODE}" \
CHANNELS=128 \
RANK=16 \
NUM_INTERACTIONS=2 \
HIDDEN_LMAX=3 \
MAX_ELL=3 \
PHASE_CONTEXT=content \
SEED=20260616 \
SYSTEMS=t1x,mal,sti,bucky,3bpa \
T1X_DATA_DIR="${DATA_ROOT}/transition1x/chorus_reaction_id_50k_seed20260616" \
MAL_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal" \
STI_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti" \
BUCKY_DATA_DIR="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed" \
THREE_BPA_DATA_DIR="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5" \
bash "${RUNNER}"

touch "${RUN_ROOT}/SBATCH_ALL_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)"
