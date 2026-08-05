#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/ylzhang/datasets/BOTNet-datasets}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/3bpa/standard_450_50_seed20260616_r5}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/3bpa_standard_20260727}"
SEED="${SEED:-20260616}"
MAX_STEPS="${MAX_STEPS:-45000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RANK="${RANK:-32}"
CHANNELS="${CHANNELS:-128}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
MODES="${MODES:-chorus_rank32 phaseoff}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EXPECTED_SOURCE_COMMIT="29e6d467317e4b5967b7ea5cbee54de953fa0d45"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${ROOT}/status.log"
}

if [[ ! -f "${DATA_DIR}/metadata.json" ]]; then
  if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${SOURCE_ROOT}")"
    git clone https://github.com/davkovacs/BOTNet-datasets.git "${SOURCE_ROOT}"
  fi
  source_commit="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
  if [[ "${source_commit}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
    echo "unexpected BOTNet-datasets commit: ${source_commit}" >&2
    exit 2
  fi
  mkdir -p "${DATA_DIR}"
  "${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/training/prepare_3bpa_standard.py" \
    --source-root "${SOURCE_ROOT}" \
    --out-dir "${DATA_DIR}" \
    --seed "${SEED}" \
    --train-size 450 --valid-size 50 --max-radius 5.0 \
    --source-commit "${source_commit}" >"${DATA_DIR}/prepare.log" 2>&1
fi
# shellcheck disable=SC1091
source "${DATA_DIR}/training.env"

train_one() {
  local mode="$1"
  local tag
  local phase_args
  local attention_args=()
  case "${mode}" in
    chorus|chorus_rank32)
      tag="3bpa_chorus_c${CHANNELS}_l2_corr3_rank${RANK}"
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full-nonlinear
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    persistent)
      tag="3bpa_persistent_c${CHANNELS}_l2_corr3_rank${RANK}"
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full-nonlinear
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope persistent
      )
      ;;
    chorus_rank16_attention)
      tag="3bpa_chorus_attention_c${CHANNELS}_l2_corr3_rank${RANK}"
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full-nonlinear
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      attention_args=(
        --attn-heads 4
        --attn-mode density-preserving
        --attn-scope all
      )
      ;;
    phaseoff)
      tag="3bpa_phaseoff_c${CHANNELS}_l2_corr3"
      phase_args=(--phase-mode none)
      ;;
    *)
      echo "unknown mode: ${mode}" >&2
      return 2
      ;;
  esac

  local out="${ROOT}/${tag}"
  local final_checkpoint="${out}/checkpoints/${tag}_final.pth"
  mkdir -p "${out}/checkpoints" "${out}/logs"
  if [[ ! -f "${out}/DONE" ]]; then
    mark "START_${tag}"
    "${PYTHON_BIN}" -m chorus.cli.train \
      --data-dir "${DATA_DIR}" --train-prefix train --val-prefix val \
      --channels "${CHANNELS}" --lmax 2 --max-ell 2 \
      --num-interaction "${NUM_INTERACTIONS}" --correlation 3 \
      --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
      --first-layer-self-connection --mace-compatible-random-init \
      --readout-hidden-channels 64 \
      --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
      --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}" \
      --atomic-energy-keys "${E0_KEYS}" --atomic-energy-values="${E0_VALUES}" \
      --scaling std_scaling \
      --epochs 1600 --max-steps "${MAX_STEPS}" --batch-size "${BATCH_SIZE}" \
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
      --seed "${SEED}" --log-interval 20 --keep-checkpoints 1600 \
      "${attention_args[@]}" \
      "${phase_args[@]}" \
      --checkpoint "${final_checkpoint}" >"${out}/logs/train.log" 2>&1
    touch "${out}/DONE"
    mark "DONE_${tag}"
  fi

  if [[ "${SKIP_EVAL}" == "1" ]]; then
    mark "SKIP_${tag}_EVAL_REQUESTED"
    return
  fi

  local eval_root="${out}/validation_force_selected_eval"
  mkdir -p "${eval_root}"
  if [[ -f "${eval_root}/DONE" ]]; then
    mark "SKIP_${tag}_EVAL_DONE"
    return
  fi
  local selected
  selected="$("${PYTHON_BIN}" - "${out}" "${eval_root}/selection.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
destination = Path(sys.argv[2])
with (run_dir / "checkpoints" / "loss.csv").open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["kind"] == "epoch"]
if not rows:
    raise RuntimeError("no validation rows")
best = min(rows, key=lambda row: (float(row["val_force_mae"]), int(row["step"])))
epoch, step = int(best["epoch"]), int(best["step"])
candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
if len(candidates) != 1:
    raise RuntimeError(f"checkpoint mismatch: {candidates}")
result = {
    "selection_rule": "minimum 300K validation Force MAE; earliest step breaks ties",
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
)"

  for temperature in 300K 600K 1200K; do
    local eval_dir="${eval_root}/${temperature}"
    local eval_data="${eval_dir}/data"
    mkdir -p "${eval_data}"
    ln -sfn "${DATA_DIR}/processed_train.h5" "${eval_data}/processed_train.h5"
    ln -sfn "${DATA_DIR}/processed_train.h5.counts.npz" \
      "${eval_data}/processed_train.h5.counts.npz"
    ln -sfn "${DATA_DIR}/processed_test_${temperature}.h5" \
      "${eval_data}/processed_val.h5"
    ln -sfn "${DATA_DIR}/processed_test_${temperature}.h5.counts.npz" \
      "${eval_data}/processed_val.h5.counts.npz"
    mark "START_${tag}_TEST_${temperature}"
    "${PYTHON_BIN}" -m chorus.cli.train \
      --data-dir "${eval_data}" --train-prefix train --val-prefix val \
      --channels "${CHANNELS}" --lmax 2 --max-ell 2 \
      --num-interaction "${NUM_INTERACTIONS}" --correlation 3 \
      --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
      --first-layer-self-connection --mace-compatible-random-init \
      --readout-hidden-channels 64 \
      --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
      --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}" \
      --atomic-energy-keys "${E0_KEYS}" --atomic-energy-values="${E0_VALUES}" \
      --scaling std_scaling --epochs 1 --batch-size "${BATCH_SIZE}" \
      --dtype float32 --device cuda --num-workers 0 \
      --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
      --phase-hidden-channels 32 --phase-scale-init 0.05 \
      --phase-density-rank "${RANK}" \
      --seed "${SEED}" "${attention_args[@]}" "${phase_args[@]}" \
      --resume-checkpoint "${selected}" --eval-only \
      --checkpoint "${eval_dir}/unused.pth" >"${eval_dir}/test.log" 2>&1
  done

  "${PYTHON_BIN}" - "${eval_root}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = json.loads((root / "selection.json").read_text())
result["tests"] = {}
patterns = {
    "energy_mae_ev_per_atom": r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom",
    "energy_rmse_ev_per_atom": r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom",
    "force_mae_ev_per_angstrom": r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A",
    "force_rmse_ev_per_angstrom": r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A",
}
for temperature in ("300K", "600K", "1200K"):
    text = (root / temperature / "test.log").read_text(errors="replace")
    metrics = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"missing {key} for {temperature}")
        metrics[key] = float(matches[-1])
    result["tests"][temperature] = metrics
(root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
PY
  touch "${eval_root}/DONE"
  mark "DONE_${tag}_ALL_TESTS"
}

for mode in ${MODES}; do
  train_one "${mode}"
done
mark "ALL_3BPA_DONE"
