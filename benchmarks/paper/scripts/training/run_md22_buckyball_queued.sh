#!/usr/bin/env bash
set -euo pipefail

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
RUNNER="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/md22_buckyball_chorus30_20260720}"
WAIT_SCREEN="${WAIT_SCREEN:-chorus_density_attention}"

mkdir -p "${OUT_ROOT}"
echo "QUEUE $(date) wait_screen=${WAIT_SCREEN}" | tee -a "${OUT_ROOT}/queue_status.log"

while screen -ls 2>/dev/null | grep -q "[.]${WAIT_SCREEN}[[:space:]]"; do
  echo "WAIT $(date) screen=${WAIT_SCREEN}" >>"${OUT_ROOT}/queue_status.log"
  sleep 60
done

echo "START $(date)" | tee -a "${OUT_ROOT}/queue_status.log"
set +e
MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
OUT_ROOT="${OUT_ROOT}" \
MODES="ictc_phase_full_l_gated_makefx,ictc_phase_diagonal_full_l_makefx,ictc_bridge_u_makefx" \
SEED=20260616 \
EPOCHS=30 \
BATCH_SIZE=4 \
MAKEFX_BUCKETS=4 \
MAKEFX_MAX_SLOTS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash "${RUNNER}" >>"${OUT_ROOT}/driver.log" 2>&1
rc=$?
set -e
echo "END rc=${rc} $(date)" | tee -a "${OUT_ROOT}/queue_status.log"
exit "${rc}"
