#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
DATA_DIR="${DATA_DIR:?set DATA_DIR}"
BASELINE_DIR="${BASELINE_DIR:?set BASELINE_DIR}"
CHORUS_DIR="${CHORUS_DIR:?set CHORUS_DIR}"
OUT="${OUT:?set OUT}"
AVG_NEIGHBORS="${AVG_NEIGHBORS:?set AVG_NEIGHBORS}"
E0_KEYS="${E0_KEYS:?set E0_KEYS}"
E0_VALUES="${E0_VALUES:?set E0_VALUES}"

mkdir -p "${OUT}"
if [[ -f "${OUT}/DONE" ]]; then
  echo "REUSE_DONE ${OUT}"
  exit 0
fi

"${PYTHON_BIN}" - \
  "${BASELINE_DIR}" "${CHORUS_DIR}" "${OUT}/selection.json" <<'PY'
import csv
import json
import sys
from pathlib import Path


def select(run_dir: Path):
    with (run_dir / "checkpoints" / "loss.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    epoch_rows = [row for row in rows if row["kind"] == "epoch"]
    if not epoch_rows:
        raise RuntimeError(f"no validation rows in {run_dir}")
    best = min(
        epoch_rows,
        key=lambda row: (float(row["val_force_mae"]), int(row["step"])),
    )
    epoch = int(best["epoch"])
    step = int(best["step"])
    candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one checkpoint for epoch={epoch}, step={step}, got {candidates}"
        )
    return {
        "checkpoint": str(candidates[0]),
        "epoch": epoch,
        "step": step,
        "selection_rule": "minimum validation Force MAE; earliest step breaks ties",
        "validation": {
            key: float(best[key])
            for key in (
                "val_force_mae",
                "val_force_rmse",
                "val_energy_mae",
                "val_energy_rmse",
            )
        },
        "validation_checkpoint_count": len(epoch_rows),
    }


baseline_dir, chorus_dir, destination = map(Path, sys.argv[1:])
result = {
    "baseline": select(baseline_dir),
    "chorus": select(chorus_dir),
    "test_used_for_selection": False,
}
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

BASELINE_CHECKPOINT="$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["baseline"]["checkpoint"])' \
  "${OUT}/selection.json")"
CHORUS_CHECKPOINT="$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["chorus"]["checkpoint"])' \
  "${OUT}/selection.json")"

DATA_DIR="${DATA_DIR}" \
OUT="${OUT}" \
AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
E0_KEYS="${E0_KEYS}" \
E0_VALUES="${E0_VALUES}" \
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT}" \
CHORUS_CHECKPOINT="${CHORUS_CHECKPOINT}" \
EVAL_PREFIX=test \
  bash "${REPO}/benchmarks/paper/scripts/training/evaluate_xxmd_large_mae_selected.sh"

