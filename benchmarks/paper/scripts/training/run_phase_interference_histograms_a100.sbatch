#!/usr/bin/env bash
#SBATCH --job-name=chorus_phase_hist
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_phase_interference_20260804/logs/%A_%a.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/chorus_persistent_runs/large_c128_r16_20260730}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/home/sps-xia/rxlin/chorus_phase_interference_20260804}"
SCRIPT="${REPO}/benchmarks/paper/scripts/training/diagnose_phase_interference.py"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    LABEL="MD22 Buckyball"
    NAME="buckyball"
    ELEMENTS="H,C"
    DATA_DIR="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed"
    CHECKPOINT="${RUN_ROOT}/mal_bucky/bucky/checkpoints/bucky_persistent_c128_l2_corr3_rank16_final.e295s44400.pth"
    ;;
  1)
    LABEL="xxMD MAL"
    NAME="mal"
    ELEMENTS="H,C,O"
    DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal"
    CHECKPOINT="${RUN_ROOT}/mal_bucky/mal/checkpoints/mal_persistent_c128_l2_corr3_rank16_final.e45s40250.pth"
    ;;
  2)
    LABEL="xxMD STI"
    NAME="sti"
    ELEMENTS="H,C"
    DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti"
    CHECKPOINT="${RUN_ROOT}/sti_3bpa/sti/checkpoints/sti_persistent_c128_l2_corr3_rank16_final.e17s14400.pth"
    ;;
  3)
    LABEL="3BPA 300 K"
    NAME="3bpa"
    ELEMENTS="H,C,N,O"
    DATA_DIR="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5"
    CHECKPOINT="${RUN_ROOT}/sti_3bpa/3bpa/3bpa_persistent_c128_l2_corr3_rank16/checkpoints/3bpa_persistent_c128_l2_corr3_rank16_final.e1407s39424.pth"
    ;;
  *)
    echo "unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/${NAME}"
test -f "${CHECKPOINT}"
test -f "${DATA_DIR}/processed_val.h5"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

{
  echo "START $(date --iso-8601=seconds)"
  echo "HOST $(hostname)"
  echo "DATASET=${LABEL}"
  echo "CHECKPOINT=${CHECKPOINT}"
  echo "DATA_DIR=${DATA_DIR}"
  echo "STRICT_FP32=true TF32=false"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}

"${PYTHON_BIN}" "${SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --elements "${ELEMENTS}" \
  --split val \
  --dataset-label "${LABEL}" \
  --batch-size 16 \
  --max-batches 1000000 \
  --intervention-batches 16 \
  --bins 72 \
  --device cuda \
  --output-dir "${OUTPUT_ROOT}/${NAME}"

echo "DONE $(date --iso-8601=seconds)"
