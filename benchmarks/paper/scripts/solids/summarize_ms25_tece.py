#!/usr/bin/env python3
"""Summarize one validation-selected TECE MS25 run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = {
    "energy_mae_mev_per_atom": r"energy_per_atom_mae:\s*([0-9.eE+-]+)",
    "energy_rmse_mev_per_atom": r"energy_per_atom_rmse:\s*([0-9.eE+-]+)",
    "force_mae_mev_per_angstrom": r"forces_mae:\s*([0-9.eE+-]+)",
    "force_rmse_mev_per_angstrom": r"forces_rmse:\s*([0-9.eE+-]+)",
}


def parse_eval(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    result = {}
    for key, pattern in METRICS.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"missing {key} in {path}")
        result[key] = float(matches[-1])
    result.update(
        {key.replace("_mev_", "_ev_"): value / 1000.0 for key, value in list(result.items())}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--size-transfer-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection_text = (args.run_dir / "selection.txt").read_text()
    checkpoint = selection_text.splitlines()[0]
    force_match = re.search(r"val_forces_mae=([0-9.eE+-]+)", selection_text)
    step_match = re.search(r"checkpoint_step=(\d+)", selection_text)
    if force_match is None or step_match is None:
        raise RuntimeError("incomplete TECE selection.txt")
    train_text = (args.run_dir / "train.log").read_text(errors="replace")
    parameter_matches = re.findall(r"Total number of parameters in the model:\s*(\d+)", train_text)
    if not parameter_matches:
        raise RuntimeError("missing TECE parameter count")
    versions = sorted(
        (args.run_dir / "lightning_logs").glob("version_*"),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not versions:
        raise RuntimeError("missing TECE TensorBoard validation history")
    events = EventAccumulator(str(versions[-1]))
    events.Reload()
    tags = {
        "energy_mae_mev_per_atom": "val/energy_per_atom_mae",
        "energy_rmse_mev_per_atom": "val/energy_per_atom_rmse",
        "force_mae_mev_per_angstrom": "val/forces_mae",
        "force_rmse_mev_per_angstrom": "val/forces_rmse",
    }
    by_step: dict[int, dict[str, float | int]] = {}
    for key, tag in tags.items():
        for point in events.Scalars(tag):
            row = by_step.setdefault(point.step + 1, {"step": point.step + 1})
            row[key] = float(point.value)
    validation_scan = [by_step[step] for step in sorted(by_step)]
    complete = [row for row in validation_scan if all(key in row for key in tags)]
    if not complete:
        raise RuntimeError("no complete TECE validation rows")
    for row in complete:
        row.update(
            {
                key.replace("_mev_", "_ev_"): float(row[key]) / 1000.0
                for key in tags
            }
        )
    force_row = min(
        complete,
        key=lambda row: (float(row["force_mae_mev_per_angstrom"]), int(row["step"])),
    )
    energy_row = min(
        complete,
        key=lambda row: (float(row["energy_mae_mev_per_atom"]), int(row["step"])),
    )
    if int(force_row["step"]) != int(step_match.group(1)):
        raise RuntimeError(
            "selection.txt does not match the TensorBoard Force-MAE envelope: "
            f"{step_match.group(1)} != {force_row['step']}"
        )
    result = {
        "selection_rule": "minimum full-validation Force MAE; test never selects",
        "test_used_for_selection": False,
        "selected_checkpoint": checkpoint,
        "selected_step": int(step_match.group(1)),
        "selected_validation_force_mae_mev_per_angstrom": float(force_match.group(1)),
        "trainable_parameters": int(parameter_matches[-1]),
        "validation_checkpoint_count": len(complete),
        "independent_validation_envelopes": {
            "force_mae": force_row,
            "energy_mae": energy_row,
        },
        "validation_scan": complete,
        "validation": parse_eval(args.run_dir / "val.log"),
        "test": parse_eval(args.run_dir / "test.log"),
    }
    if args.size_transfer_log is not None:
        result["mgo_4x4_size_transfer_test"] = parse_eval(args.size_transfer_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
