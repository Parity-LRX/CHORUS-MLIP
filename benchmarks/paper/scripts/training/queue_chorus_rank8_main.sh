#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/large_scale_main_20260724}"
STATUS="${STATUS:-${ROOT}/chorus_rank8_queue_status.log}"
WAIT_SCREEN="${WAIT_SCREEN:-xxmd_dia_rank8_control}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

run_rank8() {
  local tag="$1"
  local data_dir="$2"
  local out_dir="$3"
  local avg_neighbors="$4"
  local e0_keys="$5"
  local e0_values="$6"
  local epochs="$7"
  local max_steps="$8"
  local batch_size="$9"
  local keep_checkpoints="${10}"

  local checkpoint="${out_dir}/checkpoints/${tag}_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs${epochs}.pth"
  local done="${out_dir}/DONE"
  local log="${out_dir}/logs/${tag}_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs${epochs}.log"
  if [[ -f "${done}" ]]; then
    mark "SKIP_${tag}_DONE"
    return
  fi

  mkdir -p "${out_dir}/checkpoints" "${out_dir}/logs"
  mark "START_${tag}"
  "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir "${data_dir}" --train-prefix train --val-prefix val \
    --channels 128 --lmax 2 --max-ell 2 \
    --num-interaction 2 --correlation 3 \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels 64 \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius 5.0 --avg-num-neighbors "${avg_neighbors}" \
    --atomic-energy-keys "${e0_keys}" --atomic-energy-values="${e0_values}" \
    --scaling std_scaling \
    --epochs "${epochs}" --batch-size "${batch_size}" \
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
    --seed 20260616 --log-interval 20 \
    --max-steps "${max_steps}" --keep-checkpoints "${keep_checkpoints}" \
    --phase-mode final-full-l-residual \
    --phase-amplitude softplus --phase-coefficient polar \
    --phase-context content --phase-density-pairs full-nonlinear \
    --phase-normalization avg-neighbors \
    --phase-placement pre-product-full-l --phase-scope final \
    --checkpoint "${checkpoint}" >"${log}" 2>&1
  touch "${done}"
  mark "DONE_${tag}"
}

mark "WAIT_${WAIT_SCREEN}"
while screen_exists "${WAIT_SCREEN}"; do
  sleep 30
done

# Formal CHORUS runs use rank 8. Historical rank-16 directories are retained
# only for diagnostics and are never resumed by this queue.
run_rank8 \
  t1x_c128_l2_corr3_rank8_mae \
  /home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616 \
  "${ROOT}/t1x/chorus_c128_l2_corr3_rank8_mae_ckpts" \
  10.71685543435131 \
  1,6,7,8 \
  -13.622227668762207,-1029.4130859375,-1484.87109375,-2041.839599609375 \
  32 100000 16 40

run_rank8 \
  xxmd_mal_c128_l2_corr3_rank8 \
  /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/mal \
  "${ROOT}/xxmd/mal/chorus_c128_l2_corr3_rank8" \
  7.99384126984127 \
  1,6,8 \
  -1001.3306884765625,-750.9979858398438,-500.66534423828125 \
  52 45000 16 100

run_rank8 \
  xxmd_sti_c128_l2_corr3_rank8 \
  /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/sti \
  "${ROOT}/xxmd/sti/chorus_c128_l2_corr3_rank8" \
  16.62877403846154 \
  1,6 \
  -518.6243286132812,-605.061767578125 \
  57 45000 16 100

run_rank8 \
  md22_buckyball_catcher_c128_l2_corr3_rank8 \
  /home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
  /home/ylzhang/chorus_runs/buckyball_fair_r5_noema_20260724/chorus_c128_l2_corr3_rank8 \
  30.3929 \
  1,6 \
  -230.09867339,-986.13717166 \
  300 45000 4 100

mark "ALL_RANK8_MAIN_DONE"
