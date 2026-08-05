#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/ylzhang/chorus_runs/full_nonlinear_single_seed_screen_20260724/transition1x_plain_full_c48_l2}"
OUT="${OUT:-/home/ylzhang/chorus_runs/transition1x_plain_full_c48_l2_calibrated_20260724}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SOURCE_ROOT}/checkpoints/transition1x_ictc_phase_full_c48_l2_rank8_seed20260616_steps100000.e32s100000.pth}"
OUTPUT_CHECKPOINT="${OUT}/transition1x_plain_full_c48_l2_rank8_final_calibrated.pth"
LOG="${OUT}/calibration.log"

mkdir -p "${OUT}"
export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"

{
  echo "START $(date -Is)"
  echo "SOURCE_CHECKPOINT ${SOURCE_CHECKPOINT}"
  echo "CALIBRATION_FIT_SPLIT train"
  echo "EVALUATION_SPLIT validation"
} >"${OUT}/protocol.log"

/usr/bin/time -f "WALL_SECONDS %e" \
  "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir /home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616 \
    --train-prefix train --val-prefix val \
    --channels 48 --lmax 2 --max-ell 2 \
    --num-interaction 2 --correlation 2 \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels 64 \
    --element-energy-correction --final-fit-element-energy-correction \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius 5.0 --avg-num-neighbors 10.71685543435131 \
    --atomic-energy-keys 1,6,7,8 \
    --atomic-energy-values=-13.62222753701504,-1029.4130839658328,-1484.8710358098756,-2041.8396277138045 \
    --scaling rms_forces_scaling --atomic-inter-scale 0.7642790079116821 \
    --no-atomic-inter-shift \
    --epochs 0 --max-steps 100000 --batch-size 16 \
    --dtype float32 --device cuda --num-workers 2 \
    --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
    --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
    --optimizer adamw --optimizer-param-groups mace \
    --weight-decay 5e-7 --amsgrad --max-grad-norm 10.0 \
    --phase-hidden-channels 32 --phase-heads 1 --phase-scale-init 0.05 \
    --phase-density-rank 8 --ema-decay 0.0 --ema-start-step 0 \
    --seed 20260616 --log-interval 200 \
    --phase-mode final-full-l-residual \
    --phase-amplitude softplus --phase-coefficient polar \
    --phase-context content --phase-density-pairs full \
    --phase-normalization avg-neighbors \
    --phase-placement pre-product-full-l --phase-scope final \
    --resume-checkpoint "${SOURCE_CHECKPOINT}" \
    --checkpoint "${OUTPUT_CHECKPOINT}" \
    >"${LOG}" 2>&1

echo "DONE $(date -Is)" | tee -a "${OUT}/protocol.log"
