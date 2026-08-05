#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/large_scale_main_20260724/baselines}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"
WAIT_SCREENS="${WAIT_SCREENS:-chorus_rank8_main_queue external_large_xxmd_queue}"
SEED="${SEED:-20260616}"
SYSTEMS="${SYSTEMS:-t1x xxmd_mal xxmd_sti xxmd_dia buckyball}"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

for wait_screen in ${WAIT_SCREENS}; do
  mark "WAIT_${wait_screen}"
  while screen_exists "${wait_screen}"; do
    sleep 30
  done
done

run_ictc_phaseoff_buckyball() {
  local out="${ROOT}/mace_ictc_phaseoff_buckyball_c128_l2_corr3"
  local checkpoint="${out}/checkpoints/mace_ictc_phaseoff_buckyball_c128_l2_corr3.pth"
  if [[ -f "${out}/DONE" ]]; then
    mark "SKIP_ICTC_PHASEOFF_BUCKYBALL_DONE"
    return
  fi
  mkdir -p "${out}/checkpoints" "${out}/logs"
  mark "START_ICTC_PHASEOFF_BUCKYBALL"
  env PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m chorus.cli.train \
      --data-dir /home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
      --train-prefix train --val-prefix val \
      --channels 128 --lmax 2 --max-ell 2 \
      --num-interaction 2 --correlation 3 \
      --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
      --first-layer-self-connection --mace-compatible-random-init \
      --readout-hidden-channels 64 \
      --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
      --max-radius 5.0 --avg-num-neighbors 30.3929 \
      --atomic-energy-keys 1,6 \
      --atomic-energy-values=-230.09867339,-986.13717166 \
      --scaling std_scaling \
      --epochs 300 --max-steps 45000 --batch-size 4 \
      --dtype float32 --device cuda --num-workers 0 \
      --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
      --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
      --optimizer adamw --optimizer-param-groups mace \
      --weight-decay 5e-7 --amsgrad --max-grad-norm 10 \
      --train-makefx-compile --require-train-makefx-compile \
      --makefx-buckets 4 --makefx-max-slots 8 \
      --pad-nodes-to-max --pad-edges-to-max \
      --seed "${SEED}" --log-interval 20 --keep-checkpoints 300 \
      --phase-mode none \
      --checkpoint "${checkpoint}" \
      >"${out}/logs/train.log" 2>&1
  touch "${out}/DONE"
  mark "DONE_ICTC_PHASEOFF_BUCKYBALL"
}

prepare_buckyball_extxyz() {
  local source_root=/home/ylzhang/lrx/md22/chorus_lowdata600_20260720
  local out="${source_root}/native_mace_split"
  if [[ -f "${out}/DONE" ]]; then
    return
  fi
  mkdir -p "${out}"
  "${PYTHON_BIN}" - "${source_root}" "${out}" <<'PY'
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

source = Path(sys.argv[1])
out = Path(sys.argv[2])
frames = read(source / "candidate_1200.extxyz", index=":")
for split in ("train", "val"):
    indices = np.load(source / "processed" / f"{split}_indices.npy")
    selected = []
    for index in indices:
        atoms = frames[int(index)].copy()
        atoms.info["energy"] = float(atoms.info.pop("Energy"))
        atoms.arrays["forces"] = atoms.arrays.pop("force")
        selected.append(atoms)
    write(out / f"{split}.extxyz", selected, format="extxyz")
(out / "DONE").touch()
PY
}

run_native_mace() {
  local tag="$1"
  local data_dir="$2"
  local atomic_numbers="$3"
  local e0s="$4"
  local avg_neighbors="$5"
  local epochs="$6"
  local batch_size="$7"
  local gamma="$8"

  local out="${ROOT}/native_mace_${tag}"
  local train_log="${out}/train.log"
  local recovery_marker="${out}/RECOVERED_CUEQ_EXPORT_FAILURE"

  recover_completed_cueq_run() {
    local checkpoint_count
    checkpoint_count="$(find "${out}/checkpoints" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -f "${train_log}" ]] \
      && grep -q 'Training complete' "${train_log}" \
      && grep -q 'ScriptFunction cannot be pickled' "${train_log}" \
      && (( checkpoint_count >= epochs )); then
      printf 'RECOVERED_CUEQ_EXPORT_FAILURE %s checkpoints=%s expected=%s\n' \
        "$(date -Is)" "${checkpoint_count}" "${epochs}" >"${recovery_marker}"
      touch "${out}/DONE"
      mark "RECOVERED_NATIVE_MACE_${tag}_CUEQ_EXPORT_FAILURE_checkpoints${checkpoint_count}"
      return 0
    fi
    return 1
  }

  if [[ -f "${out}/DONE" ]]; then
    mark "SKIP_NATIVE_MACE_${tag}_DONE"
    return
  fi
  # CuEq-backed MACE 0.3.16 can fail only while deep-copying the fully
  # trained model for final export.  Recover such runs before considering a
  # restart, but only after auditing the completion line and all epoch
  # checkpoints.
  if recover_completed_cueq_run; then
    return
  fi
  mkdir -p "${out}/logs" "${out}/models" "${out}/checkpoints" "${out}/results"
  mark "START_NATIVE_MACE_${tag}"
  local exit_code=0
  if env PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m mace.cli.run_train \
      --name "native_mace_${tag}" --seed "${SEED}" \
      --device cuda --default_dtype float32 \
      --log_dir "${out}/logs" --model_dir "${out}/models" \
      --checkpoints_dir "${out}/checkpoints" --results_dir "${out}/results" \
      --model ScaleShiftMACE --r_max 5.0 \
      --radial_type bessel --num_radial_basis 8 --num_cutoff_basis 6 \
      --max_ell 2 --num_interactions 2 --correlation 3 --use_reduced_cg True \
      --num_channels 128 --max_L 2 \
      --hidden_irreps "128x0e + 128x1o + 128x2e" \
      --MLP_irreps "64x0e" --radial_MLP "[64, 64, 64]" \
      --interaction RealAgnosticResidualInteractionBlock \
      --interaction_first RealAgnosticResidualInteractionBlock \
      --enable_cueq True --only_cueq True \
      --train_file "${data_dir}/train.extxyz" \
      --valid_file "${data_dir}/val.extxyz" \
      --energy_key energy --forces_key forces \
      --atomic_numbers "${atomic_numbers}" --E0s "${e0s}" \
      --avg_num_neighbors "${avg_neighbors}" --scaling std_scaling \
      --loss weighted --energy_weight 1 --forces_weight 100 \
      --batch_size "${batch_size}" --valid_batch_size "${batch_size}" \
      --max_num_epochs "${epochs}" \
      --lr 0.001 --weight_decay 5e-7 --optimizer adamw \
      --scheduler ExponentialLR --lr_scheduler_gamma "${gamma}" --amsgrad \
      --num_workers 0 --compute_forces True --compute_stress False \
      --eval_interval 1 --keep_checkpoints --save_all_checkpoints \
      >"${train_log}" 2>&1; then
    touch "${out}/DONE"
    mark "DONE_NATIVE_MACE_${tag}"
    return
  else
    exit_code=$?
  fi

  if recover_completed_cueq_run; then
    return
  fi
  mark "FAILED_NATIVE_MACE_${tag}_exit${exit_code}"
  return "${exit_code}"
}

prepare_buckyball_extxyz

# ExponentialLR is stepped once per epoch in mace-torch 0.3.16.  Each gamma
# takes 1e-3 to approximately 1e-6 over the listed near-matched step budget.
if [[ " ${SYSTEMS} " == *" t1x "* ]]; then run_native_mace \
  t1x \
  /home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616 \
  "[1, 6, 7, 8]" \
  "{1: -13.622227668762207, 6: -1029.4130859375, 7: -1484.87109375, 8: -2041.839599609375}" \
  10.71685543435131 32 16 0.8058421877614819; fi

if [[ " ${SYSTEMS} " == *" xxmd_mal "* ]]; then run_native_mace \
  xxmd_mal \
  /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/mal \
  "[1, 6, 8]" \
  "{1: -1001.3306884765625, 6: -750.9979858398438, 8: -500.66534423828125}" \
  7.99384126984127 51 16 0.8733261623828432; fi

if [[ " ${SYSTEMS} " == *" xxmd_sti "* ]]; then run_native_mace \
  xxmd_sti \
  /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/sti \
  "[1, 6]" \
  "{1: -518.6243286132812, 6: -605.061767578125}" \
  16.62877403846154 56 16 0.8832510172826996; fi

if [[ " ${SYSTEMS} " == *" xxmd_dia "* ]]; then run_native_mace \
  xxmd_dia \
  /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/dia \
  "[1, 6, 16]" \
  "{1: -1531.021284830928, 6: -1913.77660603866, 16: -382.755321207732}" \
  14.036274193548387 58 16 0.8877197088985865; fi

if [[ " ${SYSTEMS} " == *" buckyball "* ]]; then run_native_mace \
  buckyball \
  /home/ylzhang/lrx/md22/chorus_lowdata600_20260720/native_mace_split \
  "[1, 6]" \
  "{1: -230.09867339, 6: -986.13717166}" \
  30.3929 300 4 0.9772372209558107; fi

mark "ALL_BASELINES_DONE"
