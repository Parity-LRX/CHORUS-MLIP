#!/usr/bin/env bash
set -euo pipefail

# Recover the seed-20260616 density-preserving attention run that was omitted
# when the original multiseed screen exited after seed 20260618.  Keep the
# historical chorus_md22_multiseed screen name so downstream serial queues do
# not start until training and every held-out evaluation are complete.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed}"
BASE_CAMPAIGN="${BASE_CAMPAIGN:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/md22_buckyball_lowdata600_multiseed_cosine300_20260721}"
LEGACY_CAMPAIGN="${LEGACY_CAMPAIGN:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/md22_buckyball_legacy_softmax_multiseed_cosine300_20260721}"
WAIT_SCREEN="${WAIT_SCREEN:-chorus_md22_testwatch}"
RUNNER="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
EVALUATOR="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/evaluate_md22_heldout.sh"
STATUS="${BASE_CAMPAIGN}/seed20260616_attention_recovery.log"

echo "WAIT ${WAIT_SCREEN} $(date)" | tee -a "${STATUS}"
while screen -ls 2>/dev/null | grep -q "[.]${WAIT_SCREEN}[[:space:]]"; do
  sleep 30
done

seed_root="${BASE_CAMPAIGN}/seed20260616_attention"
checkpoint="${seed_root}/checkpoints/md22_buckyball_ictc_attention_makefx_seed20260616_epochs300.pth"
if [[ ! -f "${checkpoint}" ]]; then
  echo "START_TRAIN seed=20260616 density_attention $(date)" | tee -a "${STATUS}"
  DATA_DIR="${DATA_DIR}" \
  OUT_ROOT="${seed_root}" \
  MODES=ictc_attention_makefx \
  SEED=20260616 \
  EPOCHS=300 \
  BATCH_SIZE=4 \
  READOUT_HIDDEN=64 \
  E0_KEYS=1,6 \
  E0_VALUES=-230.09867339,-986.13717166 \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${RUNNER}" >"${BASE_CAMPAIGN}/seed20260616_attention_driver.log" 2>&1
  echo "END_TRAIN seed=20260616 density_attention $(date)" | tee -a "${STATUS}"
fi

evaluate_if_missing() {
  local checkpoint_dir="$1"
  local out_dir="$2"
  if [[ -f "${out_dir}/status.log" ]] && grep -q '^ALL_OK ' "${out_dir}/status.log"; then
    echo "SKIP_HELDOUT ${out_dir} $(date)" | tee -a "${STATUS}"
    return
  fi
  CHECKPOINT_DIR="${checkpoint_dir}" OUT_DIR="${out_dir}" BATCH_SIZE=4 \
    MACE_ICTC_REPO="${MACE_ICTC_REPO}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${EVALUATOR}" | tee -a "${STATUS}"
}

for run in seed20260617 seed20260618 seed20260616_attention; do
  evaluate_if_missing "${BASE_CAMPAIGN}/${run}/checkpoints" \
    "${BASE_CAMPAIGN}/${run}/heldout_test"
done

for seed in 20260616 20260617 20260618; do
  evaluate_if_missing "${LEGACY_CAMPAIGN}/seed${seed}/checkpoints" \
    "${LEGACY_CAMPAIGN}/seed${seed}/heldout_test"
done

echo "ALL_RECOVERED $(date)" | tee -a "${STATUS}"
