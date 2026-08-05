#!/usr/bin/env bash
#SBATCH --job-name=nequip_d2_persist
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2-persistent-multiseed-%j.log

set -euo pipefail

SEED="${SEED:?submit with the required seed}"
REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/chorus_multiseed_followup_20260731/seed${SEED}/nequip/e3nn_c64_chorus_persistent}"

mkdir -p "${RUN_ROOT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

env \
  REPO="${REPO}" \
  OPERATOR_REPO="${OPERATOR_REPO}" \
  DATA_ROOT="${DATA_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  DATASETS="transition1x xxmd_mal xxmd_sti md22_buckyball 3bpa" \
  CHANNELS=64 \
  CHORUS_RANK=16 \
  CHORUS_SCOPE=persistent \
  NUM_LAYERS=2 \
  SEED="${SEED}" \
  EVAL_BATCH_SIZE=25 \
  bash "${REPO}/scripts/run_extended_variant.sh" \
    e3nn on "${CUDA_VISIBLE_DEVICES}" \
    >"${RUN_ROOT}/driver.log" 2>&1

touch "${RUN_ROOT}/ALL_DONE"
