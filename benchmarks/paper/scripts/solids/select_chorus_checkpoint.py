#!/usr/bin/env python3
"""Audit CHORUS validation history and select one Force-MAE checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "val_energy_mae",
    "val_energy_rmse",
    "val_force_mae",
    "val_force_rmse",
)


def metric_dict(row: dict[str, str]) -> dict[str, float]:
    return {key: float(row[key]) for key in METRICS}


def checkpoint_for(run_dir: Path, row: dict[str, str]) -> Path:
    epoch, step = int(row["epoch"]), int(row["step"])
    candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one checkpoint for epoch={epoch}, step={step}; got {candidates}"
        )
    return candidates[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loss_csv = args.run_dir / "checkpoints" / "loss.csv"
    with loss_csv.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("kind") == "epoch"]
    if not rows:
        raise RuntimeError(f"no validation rows in {loss_csv}")
    force_row = min(rows, key=lambda row: (float(row["val_force_mae"]), int(row["step"])))
    energy_row = min(rows, key=lambda row: (float(row["val_energy_mae"]), int(row["step"])))
    selected = checkpoint_for(args.run_dir, force_row)

    result = {
        "selection_rule": "minimum full-validation Force MAE; earliest step breaks ties",
        "test_used_for_selection": False,
        "selected_checkpoint": str(selected),
        "selected_epoch": int(force_row["epoch"]),
        "selected_step": int(force_row["step"]),
        "selected_validation": metric_dict(force_row),
        "independent_validation_envelopes": {
            "force_mae": {
                "epoch": int(force_row["epoch"]),
                "step": int(force_row["step"]),
                "simultaneous_metrics": metric_dict(force_row),
            },
            "energy_mae": {
                "epoch": int(energy_row["epoch"]),
                "step": int(energy_row["step"]),
                "simultaneous_metrics": metric_dict(energy_row),
            },
        },
        "validation_checkpoint_count": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(selected)


if __name__ == "__main__":
    main()
