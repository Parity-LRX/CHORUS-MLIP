#!/usr/bin/env bash
set -euo pipefail

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
CAMPAIGN="${CAMPAIGN:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/md22_buckyball_lowdata600_multiseed_cosine300_20260721}"
WAIT_SCREEN="${WAIT_SCREEN:-chorus_md22_multiseed}"
EVALUATOR="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/evaluate_md22_heldout.sh"

echo "WAIT ${WAIT_SCREEN} $(date)" | tee -a "${CAMPAIGN}/heldout_watcher.log"
while screen -ls 2>/dev/null | grep -q "[.]${WAIT_SCREEN}"; do
  sleep 30
done

for run in seed20260617 seed20260618 seed20260616_attention; do
  checkpoint_dir="${CAMPAIGN}/${run}/checkpoints"
  out_dir="${CAMPAIGN}/${run}/heldout_test"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "MISSING ${checkpoint_dir}" | tee -a "${CAMPAIGN}/heldout_watcher.log"
    exit 2
  fi
  CHECKPOINT_DIR="${checkpoint_dir}" OUT_DIR="${out_dir}" BATCH_SIZE=4 \
    bash "${EVALUATOR}" | tee -a "${CAMPAIGN}/heldout_watcher.log"
done

echo "ALL_HELDOUT_OK $(date)" | tee -a "${CAMPAIGN}/heldout_watcher.log"
