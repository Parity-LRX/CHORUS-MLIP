#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
SOURCE_RUN="${SOURCE_RUN:-/home/ylzhang/chorus_runs/large_scale_main_20260724/rank32_pilot/xxmd_dia_c128_l2_corr3_rank32}"
DATA_DIR="/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/dia"
OUT_DIR="${OUT_DIR:-${SOURCE_RUN}/three_window_validation_selection}"
WINDOW_ROOT="${OUT_DIR}/window_data"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

mkdir -p "${OUT_DIR}" "${WINDOW_ROOT}"

"${PYTHON_BIN}" - "${DATA_DIR}" "${WINDOW_ROOT}" <<'PY'
import h5py
import os
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
root = Path(sys.argv[2])
source_val = (data_dir / "processed_val.h5").resolve()
source_train = (data_dir / "processed_train.h5").resolve()

with h5py.File(source_val, "r") as source:
    count = len(source.keys())
    attrs = dict(source.attrs)

bounds = [0, count // 3, 2 * count // 3, count]
for index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
    window_dir = root / f"window_{index}"
    window_dir.mkdir(parents=True, exist_ok=True)
    train_link = window_dir / "processed_train.h5"
    if not train_link.exists():
        train_link.symlink_to(source_train)
    destination = window_dir / "processed_val.h5"
    if destination.exists():
        destination.unlink()
    with h5py.File(destination, "w") as target:
        for key, value in attrs.items():
            target.attrs[key] = value
        target.attrs["temporal_window_start"] = start
        target.attrs["temporal_window_stop"] = stop
        target.attrs["temporal_window_source"] = str(source_val)
        for local_index, source_index in enumerate(range(start, stop)):
            target[f"sample_{local_index}"] = h5py.ExternalLink(
                str(source_val), f"/sample_{source_index}"
            )
PY

eval_checkpoint() {
  local checkpoint="$1"
  local step="$2"
  local window="$3"
  local log="${OUT_DIR}/step_${step}.window_${window}.log"
  if grep -q "\\[EVAL-ONLY\\]" "${log}" 2>/dev/null; then
    return
  fi
  "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir "${WINDOW_ROOT}/window_${window}" \
    --train-prefix train --val-prefix val \
    --channels 128 --lmax 2 --max-ell 2 \
    --num-interaction 2 --correlation 3 \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels 64 \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius 5.0 --avg-num-neighbors 14.036274193548387 \
    --atomic-energy-keys 1,6,16 \
    --atomic-energy-values=-1531.021284830928,-1913.77660603866,-382.755321207732 \
    --scaling std_scaling \
    --epochs 1 --batch-size 8 \
    --dtype float32 --device cuda --num-workers 0 \
    --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
    --phase-hidden-channels 32 --phase-scale-init 0.05 \
    --phase-density-rank 32 \
    --seed 20260616 \
    --phase-mode final-full-l-residual \
    --phase-amplitude softplus --phase-coefficient polar \
    --phase-context content --phase-density-pairs full-nonlinear \
    --phase-normalization avg-neighbors \
    --phase-placement pre-product-full-l --phase-scope final \
    --resume-checkpoint "${checkpoint}" --eval-only \
    --checkpoint "${OUT_DIR}/unused_${step}_${window}.pth" >"${log}" 2>&1
}

mapfile -t checkpoints < <(
  find "${SOURCE_RUN}/checkpoints" -maxdepth 1 -type f -name "*.e*s*.pth" |
    sort -t s -k 2,2n
)

for checkpoint in "${checkpoints[@]}"; do
  name="$(basename "${checkpoint}")"
  step="$(sed -E 's/.*\\.e[0-9]+s([0-9]+)\\.pth/\\1/' <<<"${name}")"
  for window in 0 1 2; do
    eval_checkpoint "${checkpoint}" "${step}" "${window}"
  done
done

"${PYTHON_BIN}" - "${SOURCE_RUN}" "${OUT_DIR}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

source_run = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with (source_run / "checkpoints" / "loss.csv").open(newline="") as handle:
    validation = {
        int(row["step"]): row for row in csv.DictReader(handle) if row["kind"] == "epoch"
    }

force_rmse_pattern = r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A"
records = []
for step, row in validation.items():
    window_rmse = []
    for window in range(3):
        text = (out_dir / f"step_{step}.window_{window}.log").read_text(
            errors="replace"
        )
        matches = re.findall(force_rmse_pattern, text)
        if not matches:
            raise RuntimeError(f"missing Force RMSE for step={step}, window={window}")
        window_rmse.append(1000.0 * float(matches[-1]))
    records.append(
        {
            "step": step,
            "epoch": int(row["epoch"]),
            "window_force_rmse_mev_per_angstrom": window_rmse,
            "worst_window_force_rmse_mev_per_angstrom": max(window_rmse),
            "full_validation": {
                "energy_mae_mev_per_atom": 1000.0 * float(row["val_energy_mae"]),
                "energy_rmse_mev_per_atom": 1000.0 * float(row["val_energy_rmse"]),
                "force_mae_mev_per_angstrom": 1000.0 * float(row["val_force_mae"]),
                "force_rmse_mev_per_angstrom": 1000.0 * float(row["val_force_rmse"]),
            },
        }
    )

selected = min(
    records,
    key=lambda item: (
        item["worst_window_force_rmse_mev_per_angstrom"],
        item["step"],
    ),
)
checkpoint_matches = list(
    (source_run / "checkpoints").glob(
        f"*.e{selected['epoch']}s{selected['step']}.pth"
    )
)
if len(checkpoint_matches) != 1:
    raise RuntimeError(f"expected one selected checkpoint, got {checkpoint_matches}")
payload = {
    "selection_rule": "minimum worst contiguous validation-window Force RMSE",
    "test_used_for_selection": False,
    "window_count": 3,
    "selected_checkpoint": str(checkpoint_matches[0]),
    "selected": selected,
    "records": sorted(records, key=lambda item: item["step"]),
}
(out_dir / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
print(checkpoint_matches[0])
PY

selected="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "${OUT_DIR}/selection.json")"
"${PYTHON_BIN}" -m chorus.cli.train \
  --data-dir "${DATA_DIR}" --train-prefix train --val-prefix test \
  --channels 128 --lmax 2 --max-ell 2 \
  --num-interaction 2 --correlation 3 \
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
  --first-layer-self-connection --mace-compatible-random-init \
  --readout-hidden-channels 64 \
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
  --max-radius 5.0 --avg-num-neighbors 14.036274193548387 \
  --atomic-energy-keys 1,6,16 \
  --atomic-energy-values=-1531.021284830928,-1913.77660603866,-382.755321207732 \
  --scaling std_scaling \
  --epochs 1 --batch-size 8 \
  --dtype float32 --device cuda --num-workers 0 \
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
  --phase-hidden-channels 32 --phase-scale-init 0.05 \
  --phase-density-rank 32 \
  --seed 20260616 \
  --phase-mode final-full-l-residual \
  --phase-amplitude softplus --phase-coefficient polar \
  --phase-context content --phase-density-pairs full-nonlinear \
  --phase-normalization avg-neighbors \
  --phase-placement pre-product-full-l --phase-scope final \
  --resume-checkpoint "${selected}" --eval-only \
  --checkpoint "${OUT_DIR}/unused_test.pth" >"${OUT_DIR}/test.log" 2>&1

"${PYTHON_BIN}" - "${OUT_DIR}/selection.json" "${OUT_DIR}/test.log" \
  "${OUT_DIR}/metrics.json" <<'PY'
import json
import re
import sys
from pathlib import Path

selection_path, log_path, destination = map(Path, sys.argv[1:])
text = log_path.read_text(errors="replace")
patterns = {
    "energy_mae_mev_per_atom": r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom",
    "energy_rmse_mev_per_atom": r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom",
    "force_mae_mev_per_angstrom": r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A",
    "force_rmse_mev_per_angstrom": r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A",
}
result = json.loads(selection_path.read_text())
result["test"] = {}
for key, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"missing {key}")
    result["test"][key] = 1000.0 * float(matches[-1])
destination.write_text(json.dumps(result, indent=2) + "\n")
PY

touch "${OUT_DIR}/DONE"
