#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/Users/sara/Desktop/code/CHORUS-MLIP/benchmarks/paper/results/external/ms25_solids_20260818/raw}"
A100_HOST="${A100_HOST:-rxlin@10.10.3.20}"
GPU4090_HOST="${GPU4090_HOST:-ylzhang@10.10.3.21}"
SSH_PORT="${SSH_PORT:-18022}"

mkdir -p "${LOCAL_ROOT}/a100" "${LOCAL_ROOT}/gpu4090"
common_filters=(
  --include='*/'
  --include='*.json'
  --include='*.csv'
  --include='*.yaml'
  --include='*.log'
  --include='*.txt'
  --include='DONE'
  --include='TRAIN_DONE'
  --include='RECOVERED_CUEQ_EXPORT_FAILURE'
  --exclude='*'
)

rsync -az --prune-empty-dirs -e "ssh -p ${SSH_PORT}" \
  "${common_filters[@]}" \
  "${A100_HOST}:/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/" \
  "${LOCAL_ROOT}/a100/"
rsync -az --prune-empty-dirs -e "ssh -p ${SSH_PORT}" \
  "${common_filters[@]}" \
  "${GPU4090_HOST}:/home/ylzhang/chorus_runs/ms25_external_20260818/" \
  "${LOCAL_ROOT}/gpu4090/"

echo "MS25_RESULT_SYNC_DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
