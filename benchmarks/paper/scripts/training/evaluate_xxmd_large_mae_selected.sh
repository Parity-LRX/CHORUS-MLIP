#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_DIR="${DATA_DIR:?set DATA_DIR}"
OUT="${OUT:?set OUT}"
AVG_NEIGHBORS="${AVG_NEIGHBORS:?set AVG_NEIGHBORS}"
E0_KEYS="${E0_KEYS:?set E0_KEYS}"
E0_VALUES="${E0_VALUES:?set E0_VALUES}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:?set BASELINE_CHECKPOINT}"
CHORUS_CHECKPOINT="${CHORUS_CHECKPOINT:?set CHORUS_CHECKPOINT}"
EVAL_PREFIX="${EVAL_PREFIX:-test}"
PHASE_DENSITY_RANK="${PHASE_DENSITY_RANK:-8}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${OUT}"

common=(
  --data-dir "${DATA_DIR}" --train-prefix train --val-prefix "${EVAL_PREFIX}"
  --channels 128 --lmax 2 --max-ell 2
  --num-interaction 2 --correlation 3
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg
  --first-layer-self-connection --mace-compatible-random-init
  --readout-hidden-channels 64
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6
  --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}"
  --atomic-energy-keys "${E0_KEYS}" --atomic-energy-values="${E0_VALUES}"
  --scaling std_scaling --batch-size 16
  --dtype float32 --device cuda --num-workers 0
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0
  --epochs 0 --max-steps 0 --seed 20260616
  --phase-hidden-channels 32 --phase-scale-init 0.05
  --phase-density-rank "${PHASE_DENSITY_RANK}"
  --eval-only
)

evaluate() {
  local name="$1"
  local checkpoint="$2"
  shift 2
  local phase_args=("$@")
  "${PYTHON_BIN}" -m chorus.cli.train \
    "${common[@]}" "${phase_args[@]}" \
    --resume-checkpoint "${checkpoint}" \
    --checkpoint "${OUT}/${name}.unused.pth" \
    >"${OUT}/${name}_${EVAL_PREFIX}.log" 2>&1
}

baseline_phase=(--phase-mode none)
chorus_phase=(
  --phase-mode final-full-l-residual
  --phase-amplitude softplus --phase-coefficient polar
  --phase-context content --phase-density-pairs full-nonlinear
  --phase-normalization avg-neighbors
  --phase-placement pre-product-full-l --phase-scope final
)

evaluate baseline "${BASELINE_CHECKPOINT}" "${baseline_phase[@]}"
evaluate chorus "${CHORUS_CHECKPOINT}" "${chorus_phase[@]}"
touch "${OUT}/DONE"
