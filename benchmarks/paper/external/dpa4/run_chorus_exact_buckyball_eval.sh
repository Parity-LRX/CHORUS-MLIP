#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/buckyball_fair_r5_noema_20260724}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
OUT_DIR="${OUT_DIR:-${ROOT}/chorus_exact_eval}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}/DONE" "${OUT_DIR}/FAILED"
trap 'touch "${OUT_DIR}/FAILED"' ERR

run_eval() {
  local label="$1"
  local channels="$2"
  local rank="$3"
  local split="$4"
  local checkpoint="$5"

  "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir "${DATA_DIR}" --train-prefix train --val-prefix "${split}" \
    --channels "${channels}" --lmax 2 --max-ell 2 \
    --num-interaction 2 --correlation 3 \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels 64 \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius 5.0 --avg-num-neighbors 30.3929 \
    --atomic-energy-keys 1,6 \
    --atomic-energy-values=-230.09867339,-986.13717166 \
    --scaling std_scaling --batch-size 4 \
    --dtype float32 --device cuda --num-workers 0 \
    --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
    --epochs 1 --seed 20260616 \
    --phase-hidden-channels 32 --phase-scale-init 0.05 \
    --phase-density-rank "${rank}" \
    --phase-mode final-full-l-residual \
    --phase-amplitude softplus --phase-coefficient polar \
    --phase-context content --phase-density-pairs full-nonlinear \
    --phase-normalization avg-neighbors \
    --phase-placement pre-product-full-l --phase-scope final \
    --resume-checkpoint "${checkpoint}" --eval-only \
    --checkpoint "${OUT_DIR}/${label}_unused.pth" \
    >"${OUT_DIR}/${label}.log" 2>&1
}

small_checkpoint="${ROOT}/chorus_c64_l2_corr3_rank8/checkpoints/md22_buckyball_catcher_c64_l2_corr3_rank8_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs300.pth"
big_checkpoint="${ROOT}/chorus_c128_l2_corr3_rank16/checkpoints/md22_buckyball_catcher_c128_l2_corr3_rank16_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs300.pth"

run_eval small_val 64 8 val "${small_checkpoint}"
run_eval small_test 64 8 test "${small_checkpoint}"
run_eval big_test 128 16 test "${big_checkpoint}"

touch "${OUT_DIR}/DONE"
