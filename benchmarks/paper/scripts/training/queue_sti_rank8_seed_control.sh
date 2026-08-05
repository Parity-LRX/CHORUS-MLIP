#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/large_scale_main_20260724}"
OUT="${OUT:-${ROOT}/xxmd/sti/chorus_c128_l2_corr3_rank8_seed20260617}"
STATUS="${STATUS:-${OUT}/status.log}"
WAIT_SCREEN="${WAIT_SCREEN:-chorus_rank8_main_queue}"
SEED="${SEED:-20260617}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

mkdir -p "${OUT}/checkpoints" "${OUT}/logs"
printf 'WAIT_%s %s\n' "${WAIT_SCREEN}" "$(date -Is)" | tee -a "${STATUS}"
while screen_exists "${WAIT_SCREEN}"; do
  sleep 30
done

if [[ -f "${OUT}/DONE" ]]; then
  printf 'SKIP_DONE %s\n' "$(date -Is)" | tee -a "${STATUS}"
  exit 0
fi

TAG="xxmd_sti_c128_l2_corr3_rank8_seed${SEED}"
LOG="${OUT}/logs/${TAG}.log"
CHECKPOINT="${OUT}/checkpoints/${TAG}_ictc_phase_full_l_nonlinear_makefx_seed${SEED}_epochs57.pth"
printf 'START_%s %s\n' "${TAG}" "$(date -Is)" | tee -a "${STATUS}"

"${PYTHON_BIN}" -m chorus.cli.train \
  --data-dir /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/sti \
  --train-prefix train --val-prefix val \
  --channels 128 --lmax 2 --max-ell 2 \
  --num-interaction 2 --correlation 3 \
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
  --first-layer-self-connection --mace-compatible-random-init \
  --readout-hidden-channels 64 \
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
  --max-radius 5.0 --avg-num-neighbors 16.62877403846154 \
  --atomic-energy-keys 1,6 \
  --atomic-energy-values=-518.6243286132812,-605.061767578125 \
  --scaling std_scaling \
  --epochs 57 --batch-size 16 \
  --dtype float32 --device cuda --num-workers 0 \
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
  --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
  --optimizer adamw --optimizer-param-groups mace \
  --weight-decay 5e-7 --amsgrad --max-grad-norm 10 \
  --phase-hidden-channels 32 --phase-scale-init 0.05 \
  --phase-density-rank 8 \
  --train-makefx-compile --require-train-makefx-compile \
  --makefx-buckets 4 --makefx-max-slots 8 \
  --pad-nodes-to-max --pad-edges-to-max \
  --seed "${SEED}" --log-interval 20 \
  --max-steps 45000 --keep-checkpoints 100 \
  --phase-mode final-full-l-residual \
  --phase-amplitude softplus --phase-coefficient polar \
  --phase-context content --phase-density-pairs full-nonlinear \
  --phase-normalization avg-neighbors \
  --phase-placement pre-product-full-l --phase-scope final \
  --checkpoint "${CHECKPOINT}" >"${LOG}" 2>&1

touch "${OUT}/DONE"
printf 'DONE_%s %s\n' "${TAG}" "$(date -Is)" | tee -a "${STATUS}"
