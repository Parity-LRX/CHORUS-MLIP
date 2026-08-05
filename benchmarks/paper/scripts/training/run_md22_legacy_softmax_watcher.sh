#!/usr/bin/env bash
set -euo pipefail

# Extend the active MD22 campaign with the existing DPA-4-style legacy
# softmax attention, then evaluate all original and added checkpoints on the
# same untouched held-out pool.  This script intentionally keeps the existing
# chorus_md22_testwatch screen dependency used by downstream queues.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed}"
BASE_CAMPAIGN="${BASE_CAMPAIGN:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/md22_buckyball_lowdata600_multiseed_cosine300_20260721}"
LEGACY_CAMPAIGN="${LEGACY_CAMPAIGN:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/md22_buckyball_legacy_softmax_multiseed_cosine300_20260721}"
WAIT_SCREEN="${WAIT_SCREEN:-chorus_md22_multiseed}"
RUNNER="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_md22_buckyball_chorus.sh"
EVALUATOR="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/evaluate_md22_heldout.sh"
SEEDS="${SEEDS:-20260616,20260617,20260618}"

mkdir -p "${LEGACY_CAMPAIGN}"
cat >"${LEGACY_CAMPAIGN}/protocol.json" <<EOF
{
  "protocol": "md22_buckyball_legacy_softmax_multiseed_cosine300",
  "operator": "four-head DPA-4-style env^2-gated null softmax",
  "attention_scope": "all interaction layers",
  "seeds": "${SEEDS}",
  "epochs": 300,
  "loss": {"energy_weight": 1, "force_weight": 100},
  "scheduler": "optimizer-step cosine, 1e-3 to 1e-6",
  "execution": "required make_fx, strictly serial",
  "test": "same 4902-frame disjoint held-out pool as the base MD22 campaign"
}
EOF

echo "WAIT ${WAIT_SCREEN} $(date)" | tee -a "${LEGACY_CAMPAIGN}/status.log"
while screen -ls 2>/dev/null | grep -q "[.]${WAIT_SCREEN}[[:space:]]"; do
  sleep 30
done

IFS=',' read -r -a seed_array <<<"${SEEDS}"
for raw_seed in "${seed_array[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  run_root="${LEGACY_CAMPAIGN}/seed${seed}"
  if [[ -f "${run_root}/.complete" ]]; then
    echo "SKIP_COMPLETE seed=${seed} $(date)" | tee -a "${LEGACY_CAMPAIGN}/status.log"
    continue
  fi
  echo "START_TRAIN seed=${seed} $(date)" | tee -a "${LEGACY_CAMPAIGN}/status.log"
  DATA_DIR="${DATA_DIR}" \
  OUT_ROOT="${run_root}" \
  MODES=ictc_attention_legacy_makefx \
  SEED="${seed}" \
  EPOCHS=300 \
  BATCH_SIZE=4 \
  READOUT_HIDDEN=64 \
  E0_KEYS=1,6 \
  E0_VALUES=-230.09867339,-986.13717166 \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${RUNNER}" >"${LEGACY_CAMPAIGN}/seed${seed}_driver.log" 2>&1
  touch "${run_root}/.complete"
  echo "END_TRAIN seed=${seed} $(date)" | tee -a "${LEGACY_CAMPAIGN}/status.log"
done

# Evaluate the original runs exactly as the original watcher would have done.
for run in seed20260617 seed20260618 seed20260616_attention; do
  checkpoint_dir="${BASE_CAMPAIGN}/${run}/checkpoints"
  out_dir="${BASE_CAMPAIGN}/${run}/heldout_test"
  CHECKPOINT_DIR="${checkpoint_dir}" OUT_DIR="${out_dir}" BATCH_SIZE=4 \
    MACE_ICTC_REPO="${MACE_ICTC_REPO}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${EVALUATOR}" | tee -a "${LEGACY_CAMPAIGN}/status.log"
done

for raw_seed in "${seed_array[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  checkpoint_dir="${LEGACY_CAMPAIGN}/seed${seed}/checkpoints"
  out_dir="${LEGACY_CAMPAIGN}/seed${seed}/heldout_test"
  CHECKPOINT_DIR="${checkpoint_dir}" OUT_DIR="${out_dir}" BATCH_SIZE=4 \
    MACE_ICTC_REPO="${MACE_ICTC_REPO}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${EVALUATOR}" | tee -a "${LEGACY_CAMPAIGN}/status.log"
done

echo "ALL_OK $(date)" | tee -a "${LEGACY_CAMPAIGN}/status.log"
