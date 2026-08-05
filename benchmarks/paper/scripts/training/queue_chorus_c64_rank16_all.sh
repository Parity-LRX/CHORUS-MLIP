#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/chorus_c64_l2_corr3_rank16_all_20260728}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"
SEED="${SEED:-20260616}"
CHANNELS="${CHANNELS:-64}"
RANK="${RANK:-16}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
MODE="${MODE:-chorus}"
SYSTEMS="${SYSTEMS:-t1x,mal,sti,bucky,3bpa}"
T1X_DATA_DIR="${T1X_DATA_DIR:-/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616}"
MAL_DATA_DIR="${MAL_DATA_DIR:-/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/mal}"
STI_DATA_DIR="${STI_DATA_DIR:-/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/sti}"
BUCKY_DATA_DIR="${BUCKY_DATA_DIR:-/home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed}"
THREE_BPA_DATA_DIR="${THREE_BPA_DATA_DIR:-/home/ylzhang/lrx/3bpa/standard_450_50_seed20260616_r5}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

if [[ -f "${ROOT}/STOP_REQUESTED" ]]; then
  mark "SKIP_ALL_C64_RANK16_STOP_REQUESTED"
  exit 0
fi

case "${MODE}" in
  chorus)
    MODE_TAG="chorus"
    PHASE_SCOPE="final"
    PHASE_ARGS=(
      --phase-mode final-full-l-residual
      --phase-amplitude softplus --phase-coefficient polar
      --phase-context content --phase-density-pairs full-nonlinear
      --phase-normalization avg-neighbors
      --phase-placement pre-product-full-l --phase-scope final
    )
    ;;
  persistent)
    MODE_TAG="persistent"
    PHASE_SCOPE="persistent"
    PHASE_ARGS=(
      --phase-mode final-full-l-residual
      --phase-amplitude softplus --phase-coefficient polar
      --phase-context content --phase-density-pairs full-nonlinear
      --phase-normalization avg-neighbors
      --phase-placement pre-product-full-l --phase-scope persistent
    )
    ;;
  phaseoff)
    MODE_TAG="phaseoff"
    PHASE_SCOPE="final"
    PHASE_ARGS=(--phase-mode none)
    ;;
  *)
    echo "unknown MODE=${MODE}; expected chorus, persistent, or phaseoff" >&2
    exit 2
    ;;
esac

should_run() {
  [[ ",${SYSTEMS}," == *",$1,"* ]]
}

select_checkpoint() {
  local run_dir="$1"
  local selection_json="$2"
  "${PYTHON_BIN}" - "${run_dir}" "${selection_json}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
destination = Path(sys.argv[2])
with (run_dir / "checkpoints" / "loss.csv").open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["kind"] == "epoch"]
if not rows:
    raise RuntimeError(f"no validation rows in {run_dir}")
best = min(rows, key=lambda row: (float(row["val_force_mae"]), int(row["step"])))
epoch, step = int(best["epoch"]), int(best["step"])
candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
if len(candidates) != 1:
    raise RuntimeError(
        f"expected one checkpoint for epoch={epoch}, step={step}, got {candidates}"
    )
result = {
    "selection_rule": "minimum validation Force MAE; earliest step breaks ties",
    "test_used_for_selection": False,
    "selected_checkpoint": str(candidates[0]),
    "epoch": epoch,
    "step": step,
    "validation": {
        key: float(best[key])
        for key in (
            "val_energy_mae",
            "val_energy_rmse",
            "val_force_mae",
            "val_force_rmse",
        )
    },
    "validation_checkpoint_count": len(rows),
}
destination.write_text(json.dumps(result, indent=2) + "\n")
print(candidates[0])
PY
}

parse_test_metrics() {
  local selection_json="$1"
  local test_log="$2"
  local destination="$3"
  "${PYTHON_BIN}" - "${selection_json}" "${test_log}" "${destination}" <<'PY'
import json
import re
import sys
from pathlib import Path

selection_path, log_path, destination = map(Path, sys.argv[1:])
text = log_path.read_text(errors="replace")
patterns = {
    "energy_mae_ev_per_atom": r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom",
    "energy_rmse_ev_per_atom": r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom",
    "force_mae_ev_per_angstrom": r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A",
    "force_rmse_ev_per_angstrom": r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A",
}
test = {}
for key, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"missing {key} in {log_path}")
    test[key] = float(matches[-1])
result = json.loads(selection_path.read_text())
result["test"] = test
destination.write_text(json.dumps(result, indent=2) + "\n")
PY
}

run_system() {
  local system="$1"
  local data_dir="$2"
  local avg_neighbors="$3"
  local e0_keys="$4"
  local e0_values="$5"
  local epochs="$6"
  local max_steps="$7"
  local batch_size="$8"
  local keep_checkpoints="$9"

  local tag="${system}_${MODE_TAG}_c${CHANNELS}_l2_corr3_rank${RANK}"
  local out="${ROOT}/${system}"
  local checkpoint="${out}/checkpoints/${tag}_final.pth"
  local train_log="${out}/logs/train.log"
  local eval_dir="${out}/validation_force_selected_eval"
  mkdir -p "${out}/checkpoints" "${out}/logs" "${eval_dir}"

  if [[ ! -f "${out}/DONE" ]]; then
    local resume_args=()
    if [[ -f "${checkpoint}" ]]; then
      resume_args=(
        --resume-checkpoint "${checkpoint}"
        --resume-training-state
      )
      mark "RESUME_${system}_FROM_${checkpoint}"
    fi
    mark "START_${system}_TRAIN"
    "${PYTHON_BIN}" -m chorus.cli.train \
      --data-dir "${data_dir}" --train-prefix train --val-prefix val \
      --channels "${CHANNELS}" --lmax 2 --max-ell 2 \
      --num-interaction "${NUM_INTERACTIONS}" --correlation 3 \
      --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
      --first-layer-self-connection --mace-compatible-random-init \
      --readout-hidden-channels 64 \
      --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
      --max-radius 5.0 --avg-num-neighbors "${avg_neighbors}" \
      --atomic-energy-keys "${e0_keys}" --atomic-energy-values="${e0_values}" \
      --scaling std_scaling \
      --epochs "${epochs}" --max-steps "${max_steps}" \
      --batch-size "${batch_size}" \
      --dtype float32 --device cuda --num-workers 0 \
      --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
      --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
      --optimizer adamw --optimizer-param-groups mace \
      --weight-decay 5e-7 --amsgrad --max-grad-norm 10 \
      --phase-hidden-channels 32 --phase-scale-init 0.05 \
      --phase-density-rank "${RANK}" \
      --train-makefx-compile --require-train-makefx-compile \
      --makefx-buckets 4 --makefx-max-slots 8 \
      --pad-nodes-to-max --pad-edges-to-max \
      --seed "${SEED}" --log-interval 20 \
      --keep-checkpoints "${keep_checkpoints}" \
      "${PHASE_ARGS[@]}" \
      "${resume_args[@]}" \
      --checkpoint "${checkpoint}" >"${train_log}" 2>&1
    touch "${out}/DONE"
    mark "DONE_${system}_TRAIN"
  else
    mark "SKIP_${system}_TRAIN_DONE"
  fi

  if [[ ! -f "${eval_dir}/DONE" ]]; then
    local selected
    selected="$(select_checkpoint "${out}" "${eval_dir}/selection.json")"
    mark "START_${system}_TEST"
    "${PYTHON_BIN}" -m chorus.cli.train \
      --data-dir "${data_dir}" --train-prefix train --val-prefix test \
      --channels "${CHANNELS}" --lmax 2 --max-ell 2 \
      --num-interaction "${NUM_INTERACTIONS}" --correlation 3 \
      --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
      --first-layer-self-connection --mace-compatible-random-init \
      --readout-hidden-channels 64 \
      --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
      --max-radius 5.0 --avg-num-neighbors "${avg_neighbors}" \
      --atomic-energy-keys "${e0_keys}" --atomic-energy-values="${e0_values}" \
      --scaling std_scaling --epochs 1 --batch-size "${batch_size}" \
      --dtype float32 --device cuda --num-workers 0 \
      --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
      --phase-hidden-channels 32 --phase-scale-init 0.05 \
      --phase-density-rank "${RANK}" --seed "${SEED}" \
      "${PHASE_ARGS[@]}" \
      --resume-checkpoint "${selected}" --eval-only \
      --checkpoint "${eval_dir}/unused.pth" >"${eval_dir}/test.log" 2>&1
    parse_test_metrics \
      "${eval_dir}/selection.json" "${eval_dir}/test.log" \
      "${eval_dir}/metrics.json"
    touch "${eval_dir}/DONE"
    mark "DONE_${system}_TEST"
  else
    mark "SKIP_${system}_TEST_DONE"
  fi

  if [[ "${system}" == "t1x" && ! -f "${out}/train_only_energy_calibration/ALL_DONE" ]]; then
    local selected
    selected="$("${PYTHON_BIN}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' \
      "${eval_dir}/selection.json")"
    mark "START_T1X_TRAIN_ONLY_ENERGY_CALIBRATION"
    if [[ "${MODE}" == "phaseoff" ]]; then
      REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
        DATA_DIR="${data_dir}" ROOT="${out}" \
        OUT="${out}/train_only_energy_calibration" \
        CHANNELS="${CHANNELS}" PHASE_DENSITY_RANK="${RANK}" \
        NUM_INTERACTIONS="${NUM_INTERACTIONS}" \
        RUN_BASELINE=1 RUN_CHORUS=0 BASELINE_NAME="phaseoff_c${CHANNELS}" \
        BASELINE_SOURCE="${selected}" \
        bash "${REPO}/benchmarks/paper/scripts/training/calibrate_t1x_large_mae_checkpoints.sh"
    else
      REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
        DATA_DIR="${data_dir}" ROOT="${out}" \
        OUT="${out}/train_only_energy_calibration" \
        CHANNELS="${CHANNELS}" PHASE_DENSITY_RANK="${RANK}" \
        NUM_INTERACTIONS="${NUM_INTERACTIONS}" \
        CHORUS_SCOPE="${PHASE_SCOPE}" \
        RUN_BASELINE=0 RUN_CHORUS=1 CHORUS_NAME="chorus_c${CHANNELS}_rank${RANK}" \
        CHORUS_SOURCE="${selected}" \
        bash "${REPO}/benchmarks/paper/scripts/training/calibrate_t1x_large_mae_checkpoints.sh"
    fi
    mark "DONE_T1X_TRAIN_ONLY_ENERGY_CALIBRATION"
  fi
}

if should_run t1x; then
  run_system \
    t1x \
    "${T1X_DATA_DIR}" \
    10.71685543435131 \
    1,6,7,8 \
    -13.622227668762207,-1029.4130859375,-1484.87109375,-2041.839599609375 \
    32 100000 16 40
fi

if should_run mal; then
  run_system \
    mal \
    "${MAL_DATA_DIR}" \
    7.99384126984127 \
    1,6,8 \
    -1001.3306884765625,-750.9979858398438,-500.66534423828125 \
    52 45000 16 100
fi

if should_run sti; then
  run_system \
    sti \
    "${STI_DATA_DIR}" \
    16.62877403846154 \
    1,6 \
    -518.6243286132812,-605.061767578125 \
    57 45000 16 100
fi

if should_run bucky; then
  run_system \
    bucky \
    "${BUCKY_DATA_DIR}" \
    30.3929 \
    1,6 \
    -230.09867339,-986.13717166 \
    300 45000 4 300
fi

if should_run 3bpa; then
  mark "START_3BPA"
  if [[ "${MODE}" == "phaseoff" ]]; then
    THREE_BPA_MODES="phaseoff"
  elif [[ "${MODE}" == "persistent" ]]; then
    THREE_BPA_MODES="persistent"
  else
    THREE_BPA_MODES="chorus"
  fi
  REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
    ROOT="${ROOT}/3bpa" CHANNELS="${CHANNELS}" RANK="${RANK}" \
    DATA_DIR="${THREE_BPA_DATA_DIR}" \
    SEED="${SEED}" \
    NUM_INTERACTIONS="${NUM_INTERACTIONS}" \
    MODES="${THREE_BPA_MODES}" MAX_STEPS=45000 BATCH_SIZE=16 \
    bash "${REPO}/benchmarks/paper/scripts/training/run_3bpa_chorus_phaseoff.sh"
  mark "DONE_3BPA"
fi

touch "${ROOT}/ALL_DONE"
mark "ALL_${MODE_TAG}_C${CHANNELS}_RANK${RANK}_TASKS_DONE"
