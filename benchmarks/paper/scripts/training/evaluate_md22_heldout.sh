#!/usr/bin/env bash
set -euo pipefail

# Evaluate minimum-validation-loss checkpoints on the disjoint MD22 test pool.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to a completed run checkpoint directory}"
OUT_DIR="${OUT_DIR:?set OUT_DIR for held-out evaluation logs}"
BATCH_SIZE="${BATCH_SIZE:-4}"
AVG_NEIGHBORS="${AVG_NEIGHBORS:-30.3929}"
E0_KEYS="${E0_KEYS:-1,6}"
E0_VALUES="${E0_VALUES:--230.09867339,-986.13717166}"
SEED="${SEED:-20260616}"

export PYTHONPATH="${MACE_ICTC_REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

shopt -s nullglob
checkpoints=("${CHECKPOINT_DIR}"/*.pth)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "no checkpoints found in ${CHECKPOINT_DIR}" >&2
  exit 2
fi

for checkpoint in "${checkpoints[@]}"; do
  name="$(basename "${checkpoint}" .pth)"
  log="${OUT_DIR}/${name}_heldout_test.log"
  echo "START ${name} $(date)" | tee -a "${OUT_DIR}/status.log"
  "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir "${DATA_DIR}" --train-prefix train --val-prefix test \
    --channels 64 --lmax 1 --max-ell 2 --num-interaction 2 --correlation 2 \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --readout-hidden-channels 64 \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}" \
    --atomic-energy-keys "${E0_KEYS}" \
    --atomic-energy-values="${E0_VALUES}" \
    --scaling std_scaling --batch-size "${BATCH_SIZE}" \
    --dtype float32 --device cuda --num-workers 0 \
    --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
    --epochs 1 --seed "${SEED}" \
    --resume-checkpoint "${checkpoint}" --eval-only \
    --checkpoint "${OUT_DIR}/${name}_unused.pth" \
    >"${log}" 2>&1
  grep -A3 "\[EVAL-ONLY\]" "${log}" | tee -a "${OUT_DIR}/status.log"
  echo "END ${name} rc=0 $(date)" | tee -a "${OUT_DIR}/status.log"
done

echo "ALL_OK $(date)" | tee -a "${OUT_DIR}/status.log"
