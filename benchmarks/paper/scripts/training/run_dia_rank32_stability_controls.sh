#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/dia_rank32_stability_controls_20260727}"
DATA_DIR="/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/dia"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

mkdir -p "${ROOT}"

run_control() {
  local tag="$1"
  local scale_init="$2"
  local orthogonal_weight="$3"
  local out="${ROOT}/${tag}"
  local checkpoint="${out}/checkpoints/${tag}.pth"

  if [[ ! -f "${out}/DONE" ]]; then
    mkdir -p "${out}/checkpoints" "${out}/logs"
    "${PYTHON_BIN}" -m chorus.cli.train \
      --data-dir "${DATA_DIR}" --train-prefix train --val-prefix val \
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
      --epochs 59 --max-steps 45000 --batch-size 16 \
      --dtype float32 --device cuda --num-workers 0 \
      --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
      --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
      --optimizer adamw --optimizer-param-groups mace \
      --weight-decay 5e-7 --amsgrad --max-grad-norm 10 \
      --phase-hidden-channels 32 --phase-scale-init "${scale_init}" \
      --phase-density-rank 32 \
      --phase-rank-orthogonal-weight "${orthogonal_weight}" \
      --train-makefx-compile --require-train-makefx-compile \
      --makefx-buckets 4 --makefx-max-slots 8 \
      --pad-nodes-to-max --pad-edges-to-max \
      --seed 20260616 --log-interval 20 --keep-checkpoints 100 \
      --phase-mode final-full-l-residual \
      --phase-amplitude softplus --phase-coefficient polar \
      --phase-context content --phase-density-pairs full-nonlinear \
      --phase-normalization avg-neighbors \
      --phase-placement pre-product-full-l --phase-scope final \
      --checkpoint "${checkpoint}" >"${out}/logs/train.log" 2>&1
    touch "${out}/DONE"
  fi

  local eval_dir="${out}/validation_force_mae_selected_eval"
  if [[ -f "${eval_dir}/DONE" ]]; then
    return
  fi
  mkdir -p "${eval_dir}"
  local selected
  selected="$("${PYTHON_BIN}" - "${out}" "${eval_dir}/selection.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
destination = Path(sys.argv[2])
with (run_dir / "checkpoints" / "loss.csv").open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["kind"] == "epoch"]
best = min(rows, key=lambda row: (float(row["val_force_mae"]), int(row["step"])))
epoch, step = int(best["epoch"]), int(best["step"])
candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
if len(candidates) != 1:
    raise RuntimeError(f"expected one checkpoint, got {candidates}")
result = {
    "selection_rule": "minimum validation Force MAE; earliest step breaks ties",
    "test_used_for_selection": False,
    "selected_checkpoint": str(candidates[0]),
    "step": step,
    "epoch": epoch,
    "validation": {
        key: 1000.0 * float(best[key])
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
    --phase-hidden-channels 32 --phase-scale-init "${scale_init}" \
    --phase-density-rank 32 \
    --seed 20260616 \
    --phase-mode final-full-l-residual \
    --phase-amplitude softplus --phase-coefficient polar \
    --phase-context content --phase-density-pairs full-nonlinear \
    --phase-normalization avg-neighbors \
    --phase-placement pre-product-full-l --phase-scope final \
    --resume-checkpoint "${selected}" --eval-only \
    --checkpoint "${eval_dir}/unused.pth" >"${eval_dir}/test.log" 2>&1

  "${PYTHON_BIN}" - "${eval_dir}/selection.json" "${eval_dir}/test.log" \
    "${eval_dir}/metrics.json" <<'PY'
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
  touch "${eval_dir}/DONE"
}

case "${CONTROL_FILTER:-all}" in
  all)
    run_control rank32_scale0025 0.025 0.0
    run_control rank32_orthogonal1e3 0.05 0.001
    touch "${ROOT}/ALL_DONE"
    ;;
  scale)
    run_control rank32_scale0025 0.025 0.0
    ;;
  orthogonal)
    run_control rank32_orthogonal1e3 0.05 0.001
    ;;
  *)
    echo "unknown CONTROL_FILTER=${CONTROL_FILTER}" >&2
    exit 2
    ;;
esac
