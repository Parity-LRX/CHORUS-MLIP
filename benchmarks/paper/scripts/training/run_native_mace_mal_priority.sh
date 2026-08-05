#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/large_scale_main_20260724/baselines}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/mal}"
OUT="${OUT:-${ROOT}/native_mace_xxmd_mal}"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

if [[ -f "${OUT}/DONE" ]]; then
  printf 'SKIP_NATIVE_MACE_xxmd_mal_DONE %s\n' "$(date -Is)"
  exit 0
fi

mkdir -p "${OUT}/logs" "${OUT}/models" "${OUT}/checkpoints" "${OUT}/results"
printf 'START_NATIVE_MACE_xxmd_mal %s\n' "$(date -Is)" | tee -a "${OUT}/status.log"

set +e
env PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m mace.cli.run_train \
    --name native_mace_xxmd_mal --seed 20260616 \
    --device cuda --default_dtype float32 \
    --log_dir "${OUT}/logs" --model_dir "${OUT}/models" \
    --checkpoints_dir "${OUT}/checkpoints" --results_dir "${OUT}/results" \
    --model ScaleShiftMACE --r_max 5.0 \
    --radial_type bessel --num_radial_basis 8 --num_cutoff_basis 6 \
    --max_ell 2 --num_interactions 2 --correlation 3 --use_reduced_cg True \
    --num_channels 128 --max_L 2 \
    --hidden_irreps "128x0e + 128x1o + 128x2e" \
    --MLP_irreps "64x0e" --radial_MLP "[64, 64, 64]" \
    --interaction RealAgnosticResidualInteractionBlock \
    --interaction_first RealAgnosticResidualInteractionBlock \
    --enable_cueq True --only_cueq True \
    --train_file "${DATA_DIR}/train.extxyz" \
    --valid_file "${DATA_DIR}/val.extxyz" \
    --energy_key energy --forces_key forces \
    --atomic_numbers "[1, 6, 8]" \
    --E0s "{1: -1001.3306884765625, 6: -750.9979858398438, 8: -500.66534423828125}" \
    --avg_num_neighbors 7.99384126984127 --scaling std_scaling \
    --loss weighted --energy_weight 1 --forces_weight 100 \
    --batch_size 16 --valid_batch_size 16 \
    --max_num_epochs 51 \
    --lr 0.001 --weight_decay 5e-7 --optimizer adamw \
    --scheduler ExponentialLR --lr_scheduler_gamma 0.8733261623828432 \
    --amsgrad --num_workers 0 \
    --compute_forces True --compute_stress False \
    --eval_interval 1 --keep_checkpoints --save_all_checkpoints \
    >"${OUT}/train.log" 2>&1
train_rc=$?
set -e

if (( train_rc != 0 )); then
  checkpoint_count="$(
    find "${OUT}/checkpoints" -type f -name '*epoch-*.pt' | wc -l | tr -d ' '
  )"
  eval_count="$(
    grep -c '"mode": "eval"' \
      "${OUT}/results/native_mace_xxmd_mal_run-20260616_train.txt" \
      2>/dev/null || true
  )"
  if grep -q 'Training complete' "${OUT}/train.log" \
    && grep -q 'ScriptFunction cannot be pickled' "${OUT}/train.log" \
    && (( checkpoint_count >= 51 )) \
    && (( eval_count >= 51 )); then
    {
      printf 'MACE 0.3.16 CuEq whole-model export failed after training.\n'
      printf 'Training complete; epoch checkpoints=%s; eval records=%s.\n' \
        "${checkpoint_count}" "${eval_count}"
      printf 'The checkpoint state dictionaries and validation records are retained.\n'
    } >"${OUT}/CUEQ_EXPORT_FAILED_AFTER_TRAINING"
    printf 'RECOVERED_CUEQ_EXPORT_FAILURE %s checkpoints=%s evals=%s\n' \
      "$(date -Is)" "${checkpoint_count}" "${eval_count}" \
      | tee -a "${OUT}/status.log"
  else
    printf 'FAILED_NATIVE_MACE_xxmd_mal %s rc=%s checkpoints=%s evals=%s\n' \
      "$(date -Is)" "${train_rc}" "${checkpoint_count}" "${eval_count}" \
      | tee -a "${OUT}/status.log"
    exit "${train_rc}"
  fi
fi

touch "${OUT}/DONE"
printf 'DONE_NATIVE_MACE_xxmd_mal %s\n' "$(date -Is)" | tee -a "${OUT}/status.log"
