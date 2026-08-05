#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
RUN_DIR="${RUN_DIR:-/home/ylzhang/chorus_runs/large_scale_main_20260724/rank32_pilot/xxmd_dia_c128_l2_corr3_rank32}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/predeclared_early_stop_test_diagnostics}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

mkdir -p "${OUT_DIR}"

declare -A RULES=(
  [30225]="minimum_validation_force_rmse"
  [26350]="earliest_within_5pct_of_best_validation_force_mae"
  [21700]="earliest_within_10pct_of_best_validation_force_mae"
)

for step in 30225 26350 21700; do
  checkpoint="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name "*s${step}.pth" -print -quit)"
  if [[ -z "${checkpoint}" ]]; then
    echo "missing checkpoint for step ${step}" >&2
    exit 2
  fi

  log="${OUT_DIR}/step_${step}.test.log"
  "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/dia \
    --train-prefix train --val-prefix test \
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
    --checkpoint "${OUT_DIR}/unused_${step}.pth" >"${log}" 2>&1
done

"${PYTHON_BIN}" - "${RUN_DIR}/checkpoints/loss.csv" "${OUT_DIR}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

loss_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
rules = {
    30225: "minimum_validation_force_rmse",
    26350: "earliest_within_5pct_of_best_validation_force_mae",
    21700: "earliest_within_10pct_of_best_validation_force_mae",
}

with loss_path.open(newline="") as handle:
    rows = {int(row["step"]): row for row in csv.DictReader(handle)}

patterns = {
    "energy_mae_mev_per_atom": (
        r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom"
    ),
    "energy_rmse_mev_per_atom": (
        r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom"
    ),
    "force_mae_mev_per_angstrom": r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A",
    "force_rmse_mev_per_angstrom": r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A",
}

results = []
for step, rule in rules.items():
    row = rows[step]
    text = (out_dir / f"step_{step}.test.log").read_text(errors="replace")
    test = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"missing {key} for step {step}")
        test[key] = 1000.0 * float(matches[-1])
    results.append(
        {
            "step": step,
            "selection_rule": rule,
            "test_used_for_rule": False,
            "validation": {
                "energy_mae_mev_per_atom": 1000.0 * float(row["val_energy_mae"]),
                "energy_rmse_mev_per_atom": 1000.0 * float(row["val_energy_rmse"]),
                "force_mae_mev_per_angstrom": 1000.0 * float(row["val_force_mae"]),
                "force_rmse_mev_per_angstrom": 1000.0 * float(row["val_force_rmse"]),
            },
            "test": test,
        }
    )

payload = {
    "purpose": "diagnose DIA rank-32 temporal extrapolation with predeclared validation-only checkpoint rules",
    "formal_primary_rule": "minimum_validation_force_rmse",
    "warning": "Do not select among these candidates using test metrics.",
    "results": results,
}
(out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
(out_dir / "DONE").touch()
PY
