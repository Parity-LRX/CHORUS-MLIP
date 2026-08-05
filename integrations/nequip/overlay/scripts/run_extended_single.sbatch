#!/usr/bin/env bash
#SBATCH --job-name=nequip_ext_recovery
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/extended-recovery-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
BACKEND="${BACKEND:?set BACKEND=e3nn or ictc}"
CHORUS="${CHORUS:?set CHORUS=off or on}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to the existing benchmark root}"
DATASETS="${DATASETS:?set DATASETS to the datasets to recover}"
CHANNELS="${CHANNELS:-64}"
CHORUS_RANK="${CHORUS_RANK:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
REUSE_COMPLETED_TRAIN="${REUSE_COMPLETED_TRAIN:-1}"

case "${BACKEND}" in
  e3nn|ictc) ;;
  *) echo "unsupported BACKEND=${BACKEND}" >&2; exit 2 ;;
esac
case "${CHORUS}" in
  off|on) ;;
  *) echo "unsupported CHORUS=${CHORUS}" >&2; exit 2 ;;
esac

exec env \
  REPO="${REPO}" \
  RUN_ROOT="${RUN_ROOT}" \
  DATASETS="${DATASETS}" \
  CHANNELS="${CHANNELS}" \
  CHORUS_RANK="${CHORUS_RANK}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  REUSE_COMPLETED_TRAIN="${REUSE_COMPLETED_TRAIN}" \
  bash "${REPO}/scripts/run_extended_variant.sh" \
    "${BACKEND}" "${CHORUS}" "${CUDA_VISIBLE_DEVICES}"
