#!/usr/bin/env bash
#SBATCH --job-name=nequip_d2_off
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2-off-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_scope_c64_r16_20260801}"
DATASETS="${DATASETS:-transition1x xxmd_mal xxmd_sti md22_buckyball 3bpa}"
CHANNELS="${CHANNELS:-64}"
SEED="${SEED:-20260616}"

mkdir -p "${RUN_ROOT}"
export OPERATOR_REPO
exec env \
  REPO="${REPO}" \
  RUN_ROOT="${RUN_ROOT}" \
  DATASETS="${DATASETS}" \
  CHANNELS="${CHANNELS}" \
  NUM_LAYERS=2 \
  SEED="${SEED}" \
  EVAL_BATCH_SIZE=25 \
  bash "${REPO}/scripts/run_extended_variant.sh" \
    e3nn off "${CUDA_VISIBLE_DEVICES}"
