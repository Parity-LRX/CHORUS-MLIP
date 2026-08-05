#!/usr/bin/env bash
set -euo pipefail

# Evaluate best-validation checkpoints on an untouched official xxMD-DFT test
# split.  Architecture flags are reconstructed from the mode encoded in each
# checkpoint name, avoiding accidental evaluation with a different operator.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to one prepared xxMD molecule directory}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
AVG_NEIGHBORS="${AVG_NEIGHBORS:?set AVG_NEIGHBORS from preparation metadata}"
E0_KEYS="${E0_KEYS:?set E0_KEYS from the training metadata}"
E0_VALUES="${E0_VALUES:?set E0_VALUES from the training metadata}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100}"
PHASE_DENSITY_RANK="${PHASE_DENSITY_RANK:-8}"
EVAL_PREFIX="${EVAL_PREFIX:-test}"

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
  phase_args=(--phase-mode none)
  case "${name}" in
    *ictc_bridge_u_makefx*)
      ;;
    *ictc_phase_charge2_full_l_makefx*)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs charge2
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    *ictc_phase_full_l_adaptive_makefx*)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full-adaptive
        --phase-coherence-init 0.1
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    *ictc_phase_full_l_adaptive_env_makefx*)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full-adaptive-env
        --phase-coherence-init "${PHASE_COHERENCE_INIT:-0.5}"
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    *ictc_phase_full_l_gated_makefx*)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full-gated
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    *ictc_phase_full_l_softplus_makefx*)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    *ictc_phase_diagonal_full_l_makefx*)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs diagonal
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    *ictc_attention_legacy_makefx*)
      phase_args=(
        --phase-mode none
        --attn-heads 4
        --attn-mode legacy-softmax
        --attn-scope all
      )
      ;;
    *ictc_attention_makefx*)
      phase_args=(
        --phase-mode none
        --attn-heads 4
        --attn-mode density-preserving
        --attn-scope final
      )
      ;;
    *)
      echo "cannot infer xxMD mode from checkpoint ${name}" >&2
      exit 3
      ;;
  esac

  log="${OUT_DIR}/${name}_official_test.log"
  echo "START ${name} $(date)" | tee -a "${OUT_DIR}/status.log"
  "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir "${DATA_DIR}" --train-prefix train --val-prefix "${EVAL_PREFIX}" \
    --channels 64 --lmax 1 --max-ell 2 --num-interaction 2 --correlation 2 \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels 64 \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}" \
    --atomic-energy-keys "${E0_KEYS}" --atomic-energy-values="${E0_VALUES}" \
    --scaling std_scaling --batch-size "${BATCH_SIZE}" \
    --dtype float32 --device cuda --num-workers 2 \
    --loss mse --energy-weight 1 --force-weight "${FORCE_WEIGHT}" --stress-weight 0 \
    --epochs 1 --seed 20260616 \
    --phase-hidden-channels 32 --phase-scale-init 0.05 \
    --phase-density-rank "${PHASE_DENSITY_RANK}" \
    --resume-checkpoint "${checkpoint}" --eval-only \
    --checkpoint "${OUT_DIR}/${name}_unused.pth" \
    "${phase_args[@]}" >"${log}" 2>&1
  grep -A3 "\[EVAL-ONLY\]" "${log}" | tee -a "${OUT_DIR}/status.log"
  echo "END ${name} rc=0 $(date)" | tee -a "${OUT_DIR}/status.log"
done

echo "ALL_OK $(date)" | tee -a "${OUT_DIR}/status.log"
