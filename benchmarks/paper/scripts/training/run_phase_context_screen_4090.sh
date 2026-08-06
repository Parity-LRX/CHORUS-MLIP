#!/usr/bin/env bash
set -euo pipefail

# Exploratory phase-context screen.  The three models share the same backbone,
# seed, optimiser-step budget, and validation split.  Only the invariant node
# summary supplied to the phase MLP changes.

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/phase_context_screen_c128_r16_20260806}"
MAX_STEPS="${MAX_STEPS:-15000}"
SEED="${SEED:-20260616}"
CONTEXTS=(content irrep-norm content-irrep-norm)

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${ROOT}/status.log"
}

run_one() {
  local dataset="$1" data_dir="$2" avg_neighbors="$3"
  local e0_keys="$4" e0_values="$5" batch_size="$6" epochs="$7"
  local context="$8"
  local tag="${dataset}_${context}"
  local out="${ROOT}/${tag}"
  local checkpoint="${out}/checkpoints/${tag}.pth"
  if [[ -f "${out}/DONE" ]]; then
    mark "SKIP_${tag}_DONE"
    return
  fi
  mkdir -p "${out}/checkpoints" "${out}/logs"
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
    --epochs "${epochs}" --max-steps "${MAX_STEPS}" --batch-size "${batch_size}" \
    --dtype float32 --device cuda --num-workers 0 \
    --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
    --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
    --optimizer adamw --optimizer-param-groups mace \
    --weight-decay 5e-7 --amsgrad --max-grad-norm 10 \
    --phase-hidden-channels 32 --phase-scale-init 0.05 \
    --phase-density-rank 16 \
    --phase-mode final-full-l-residual \
    --phase-amplitude softplus --phase-coefficient polar \
    --phase-context "${context}" --phase-density-pairs full-nonlinear \
    --phase-normalization avg-neighbors \
    --phase-placement pre-product-full-l --phase-scope final \
    --train-makefx-compile --require-train-makefx-compile \
    --makefx-buckets 4 --makefx-max-slots 8 \
    --pad-nodes-to-max --pad-edges-to-max \
    --seed "${SEED}" --log-interval 20 --keep-checkpoints 0 \
    --checkpoint "${checkpoint}" >"${out}/logs/train.log" 2>&1
  touch "${out}/DONE"
  mark "DONE_${tag}"
}

run_dataset_parallel() {
  local dataset="$1" data_dir="$2" avg_neighbors="$3"
  local e0_keys="$4" e0_values="$5" batch_size="$6" epochs="$7"
  local pids=()
  for context in "${CONTEXTS[@]}"; do
    run_one "${dataset}" "${data_dir}" "${avg_neighbors}" \
      "${e0_keys}" "${e0_values}" "${batch_size}" "${epochs}" \
      "${context}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    mark "FAILED_${dataset}"
    return 1
  fi
  mark "DONE_ALL_${dataset}"
}

cat >"${ROOT}/protocol.json" <<EOF
{
  "purpose": "MACE-ICTC phase-context single-variable screen",
  "contexts": ["content", "irrep-norm", "content-irrep-norm"],
  "channels": 128,
  "lmax": 2,
  "correlation": 3,
  "interactions": 2,
  "rank": 16,
  "phase_scope": "final",
  "max_steps": ${MAX_STEPS},
  "seed": ${SEED},
  "precision": "strict float32; TF32 disabled",
  "note": "Concurrent wall time is exploratory and must not enter throughput tables. Wider norm contexts have additional phase-trunk parameters, which must be reported."
}
EOF

# Run one dataset family at a time.  Within each family, contexts run in
# parallel because only accuracy versus optimiser steps is used here.
run_dataset_parallel \
  buckyball /home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
  30.3929 1,6 -230.09867339,-986.13717166 4 100

run_dataset_parallel \
  mal /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/mal \
  7.99384126984127 1,6,8 -1001.330686760234,-750.9980150701755,-500.665343380117 \
  16 18

run_dataset_parallel \
  3bpa /home/ylzhang/lrx/3bpa/standard_450_50_seed20260616_r5 \
  16.712427983539094 1,6,7,8 \
  -723.2941476475917,-723.2941476475917,-120.549024607932,-60.27451230396598 \
  16 520

mark COMPLETE
