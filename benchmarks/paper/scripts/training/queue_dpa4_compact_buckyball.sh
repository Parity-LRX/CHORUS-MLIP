#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/dpa4_compact_buckyball_20260724}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"

mkdir -p "${ROOT}"
mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

gpu_is_busy() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[0-9]+$'
}

mark "WAIT_PRIMARY_DPA_QUEUE"
while screen -list | grep -q '\.dpa4_queue'; do
  sleep 30
done

mark "WAIT_GPU_IDLE"
while gpu_is_busy; do
  sleep 30
done

mark "START_C13_MIX2_45000"
OUT="${ROOT}/c13_mix2" \
STEPS=45000 \
CHANNELS=13 \
MIXING_LAYERS=2 \
REPO="${REPO}" \
DPA_ENV="${DPA_ENV}" \
  bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_master_buckyball.sh"
mark "DONE_C13_MIX2_45000"
mark "ALL_DONE"
