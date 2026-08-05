#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/compact_table_missing_20260728}"
BASE="${BASE:-/home/ylzhang/chorus_runs/large_scale_main_20260724/baselines}"
WAIT_SCREEN="${WAIT_SCREEN:-dpa4_c48_scaling}"
STATUS="${ROOT}/queue_status.log"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

mark "WAIT_${WAIT_SCREEN}"
while screen_exists "${WAIT_SCREEN}"; do
  sleep 30
done

dpa_c48_root=/home/ylzhang/chorus_runs/dpa4_c48_scaling_20260727/t1x_c48_mix3
dpa_c48_eval="${dpa_c48_root}/full_val_force_mae_eval/metrics.json"
dpa_c48_calibration="${dpa_c48_root}/train_only_energy_calibration"
if [[ -f "${dpa_c48_eval}" && ! -f "${dpa_c48_calibration}/DONE" ]]; then
  mark "START_DPA_C48_T1X_TRAIN_ONLY_CALIBRATION"
  selected_checkpoint="$("/home/ylzhang/venvs/dpa4-master/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' \
    "${dpa_c48_eval}")"
  "/home/ylzhang/venvs/dpa4-master/bin/python" \
    "${REPO}/benchmarks/paper/scripts/training/calibrate_dpa4_t1x_energy.py" \
    --checkpoint "${selected_checkpoint}" \
    --data /home/ylzhang/lrx/transition1x/deepmd_reaction_id_50k_seed20260616 \
    --out "${dpa_c48_calibration}" \
    >"${dpa_c48_calibration}.log" 2>&1
  mark "DONE_DPA_C48_T1X_TRAIN_ONLY_CALIBRATION"
fi

run_native() {
  local tag="$1"
  local test_file="$2"
  local atomic_numbers="$3"
  local avg_neighbors="$4"
  local train_file="${5:-}"
  local out="${ROOT}/native_mace_${tag}"
  if [[ -f "${out}/DONE" ]]; then
    mark "SKIP_NATIVE_${tag}_DONE"
    return
  fi
  mark "START_NATIVE_${tag}"
  local args=(
    --run-dir "${BASE}/native_mace_${tag}"
    --test-file "${test_file}"
    --name "native_mace_${tag}"
    --atomic-numbers "${atomic_numbers}"
    --avg-num-neighbors "${avg_neighbors}"
    --out "${out}"
  )
  if [[ -n "${train_file}" ]]; then
    args+=(--calibration-train-file "${train_file}")
  fi
  env PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" \
      "${REPO}/benchmarks/paper/scripts/training/evaluate_native_mace_selected.py" \
      "${args[@]}" >"${out}.log" 2>&1
  mark "DONE_NATIVE_${tag}"
}

run_native \
  t1x \
  /home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616/test.extxyz \
  1,6,7,8 \
  10.71685543435131 \
  /home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616/train.extxyz

run_native \
  xxmd_mal \
  /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/mal/test.extxyz \
  1,6,8 \
  7.99384126984127

run_native \
  buckyball \
  /home/ylzhang/lrx/md22/chorus_lowdata600_20260720/heldout_test.extxyz \
  1,6 \
  30.3929

ictc_out="${ROOT}/mace_ictc_phaseoff_buckyball"
if [[ ! -f "${ictc_out}/DONE" ]]; then
  mkdir -p "${ictc_out}"
  mark "START_ICTC_PHASEOFF_BUCKYBALL"
  "${PYTHON_BIN}" - \
    "${BASE}/mace_ictc_phaseoff_buckyball_c128_l2_corr3" \
    "${ictc_out}/selection.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = list(csv.DictReader((run_dir / "checkpoints" / "loss.csv").open()))
rows = [row for row in rows if row["kind"] == "epoch"]
best = min(rows, key=lambda row: (float(row["val_force_mae"]), int(row["step"])))
epoch, step = int(best["epoch"]), int(best["step"])
candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
if len(candidates) != 1:
    raise RuntimeError(f"expected one checkpoint, found {candidates}")
result = {
    "selection_rule": "minimum full-validation Force MAE; test never selects",
    "selected_epoch": epoch,
    "selected_step": step,
    "selected_checkpoint": str(candidates[0]),
    "validation_checkpoint_count": len(rows),
    "validation": {
        key: float(best[key])
        for key in (
            "val_energy_mae",
            "val_energy_rmse",
            "val_force_mae",
            "val_force_rmse",
        )
    },
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
  checkpoint="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' \
    "${ictc_out}/selection.json")"
  env PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir /home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
    --train-prefix train --val-prefix test \
    --channels 128 --lmax 2 --max-ell 2 \
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
    --epochs 0 --max-steps 0 --seed 20260616 \
    --phase-mode none --eval-only \
    --resume-checkpoint "${checkpoint}" \
    --checkpoint "${ictc_out}/unused.pth" \
    >"${ictc_out}/test.log" 2>&1
  touch "${ictc_out}/DONE"
  mark "DONE_ICTC_PHASEOFF_BUCKYBALL"
fi

touch "${ROOT}/DONE"
mark "ALL_DONE"
