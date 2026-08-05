#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616}"
OUT_ROOT="${OUT_ROOT:-/home/ylzhang/chorus_runs/t1x_diagonal_c128_l2_corr3_rank16_20260731}"
CALIBRATOR="${CALIBRATOR:-${REPO}/benchmarks/paper/scripts/training/calibrate_t1x_large_mae_checkpoints.sh}"

SEED="${SEED:-20260616}"
EPOCHS=32
MAX_STEPS=100000
RANK=16
AVG_NEIGHBORS=10.71685543435131
E0_KEYS="1,6,7,8"
E0_VALUES="-13.622227668762207,-1029.4130859375,-1484.87109375,-2041.839599609375"
NAME="t1x_c128_l2_corr3_rank16_diagonal_makefx_seed${SEED}_epochs${EPOCHS}"
CHECKPOINT="${OUT_ROOT}/checkpoints/${NAME}.pth"
LOG="${OUT_ROOT}/logs/train.log"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

mkdir -p "${OUT_ROOT}/checkpoints" "${OUT_ROOT}/logs"

cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "dataset": "Transition1x reaction-disjoint 50k/10k/10k",
  "data_dir": "${DATA_DIR}",
  "operator": "diagonal Hermitian density (j=k only)",
  "seed": ${SEED},
  "epochs": ${EPOCHS},
  "max_steps": ${MAX_STEPS},
  "batch_size": 16,
  "channels": 128,
  "hidden_lmax": 2,
  "max_ell": 2,
  "num_interactions": 2,
  "correlation": 3,
  "phase_density_rank": ${RANK},
  "phase_density_pairs": "diagonal",
  "phase_scope": "final",
  "phase_placement": "pre-product-full-l",
  "radius_angstrom": 5.0,
  "loss": "MSE energy:force=1:100",
  "scheduler": "optimizer-step cosine, 1e-3 to 1e-6",
  "dtype": "float32",
  "tf32": false,
  "ema": false,
  "makefx": true,
  "keep_validation_checkpoints": 40,
  "checkpoint_selection": "minimum validation Force MAE; earliest step breaks ties",
  "energy_calibration": "per-element constant residual fitted on train only after checkpoint selection"
}
EOF

if [[ ! -f "${OUT_ROOT}/TRAIN_DONE" ]]; then
  printf 'START_TRAIN %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/status.log"
  /usr/bin/time -f 'WALL_SECONDS %e' "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir "${DATA_DIR}" --train-prefix train --val-prefix val \
    --channels 128 --lmax 2 --max-ell 2 \
    --num-interaction 2 --correlation 3 \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels 64 \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}" \
    --atomic-energy-keys "${E0_KEYS}" --atomic-energy-values="${E0_VALUES}" \
    --scaling std_scaling \
    --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}" \
    --batch-size 16 --dtype float32 --device cuda --num-workers 0 \
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
    --keep-checkpoints 40 \
    --phase-mode final-full-l-residual \
    --phase-amplitude softplus --phase-coefficient polar \
    --phase-context content --phase-density-pairs diagonal \
    --phase-normalization avg-neighbors \
    --phase-placement pre-product-full-l --phase-scope final \
    --checkpoint "${CHECKPOINT}" >"${LOG}" 2>&1
  touch "${OUT_ROOT}/TRAIN_DONE"
  printf 'DONE_TRAIN %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/status.log"
fi

SELECTED="$("${PYTHON_BIN}" - "${OUT_ROOT}" "${OUT_ROOT}/selection.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
destination = Path(sys.argv[2])
with (run_dir / "checkpoints" / "loss.csv").open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["kind"] == "epoch"]
if not rows:
    raise RuntimeError("no validation checkpoints found")
best_force = min(
    rows, key=lambda row: (float(row["val_force_mae"]), int(row["step"]))
)
best_energy = min(
    rows, key=lambda row: (float(row["val_energy_mae"]), int(row["step"]))
)
epoch = int(best_force["epoch"])
step = int(best_force["step"])
candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
if len(candidates) != 1:
    raise RuntimeError(
        f"expected one checkpoint at epoch={epoch}, step={step}; got {candidates}"
    )

def metrics(row):
    return {
        key: float(row[key])
        for key in (
            "val_energy_mae",
            "val_energy_rmse",
            "val_force_mae",
            "val_force_rmse",
        )
    }

result = {
    "selection_rule": "minimum validation Force MAE; earliest step breaks ties",
    "test_used_for_selection": False,
    "selected_checkpoint": str(candidates[0]),
    "validation_checkpoint_count": len(rows),
    "force_mae_selected": {
        "epoch": epoch,
        "step": step,
        "metrics": metrics(best_force),
    },
    "independent_energy_mae_envelope": {
        "epoch": int(best_energy["epoch"]),
        "step": int(best_energy["step"]),
        "metrics": metrics(best_energy),
    },
}
destination.write_text(json.dumps(result, indent=2) + "\n")
print(candidates[0])
PY
)"

if [[ ! -f "${OUT_ROOT}/train_only_calibration/ALL_DONE" ]]; then
  printf 'START_TRAIN_ONLY_CALIBRATION %s\n' "$(date -Is)" \
    | tee -a "${OUT_ROOT}/status.log"
  RUN_BASELINE=0 \
  RUN_CHORUS=1 \
  CHORUS_NAME=diagonal \
  CHORUS_SOURCE="${SELECTED}" \
  PHASE_DENSITY_RANK="${RANK}" \
  PHASE_DENSITY_PAIRS=diagonal \
  OUT="${OUT_ROOT}/train_only_calibration" \
  DATA_DIR="${DATA_DIR}" \
  REPO="${REPO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
  bash "${CALIBRATOR}"
  printf 'DONE_TRAIN_ONLY_CALIBRATION %s\n' "$(date -Is)" \
    | tee -a "${OUT_ROOT}/status.log"
fi

touch "${OUT_ROOT}/DONE"
printf 'ALL_DONE %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/status.log"
