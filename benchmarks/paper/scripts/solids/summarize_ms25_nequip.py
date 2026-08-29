#!/usr/bin/env python3
"""Summarize one validation-selected NequIP MS25 run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TEST_KEYS = {
    "force_mae_ev_per_angstrom": "f_mae",
    "force_rmse_ev_per_angstrom": "f_rmse",
    "energy_mae_ev_per_atom": "e/N_mae",
    "energy_rmse_ev_per_atom": "e/N_rmse",
}


def parse_test(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    result: dict[str, float] = {}
    for output_key, log_key in TEST_KEYS.items():
        matches = re.findall(
            rf"^\s*{re.escape(log_key)}\s*=\s*([0-9.eE+-]+)\s*$",
            text,
            flags=re.MULTILINE,
        )
        if not matches:
            raise RuntimeError(f"missing {log_key} in {path}")
        result[output_key] = float(matches[-1])
    return result


def mev(metrics: dict[str, float]) -> dict[str, float]:
    return {key.replace("_ev_", "_mev_"): 1000.0 * value for key, value in metrics.items()}


def validation_metrics(row: dict[str, str]) -> dict[str, float]:
    return {
        "energy_mae_ev_per_atom": float(row["validation_e/N_mae"]),
        "energy_rmse_ev_per_atom": float(row["validation_e/N_rmse"]),
        "force_mae_ev_per_angstrom": float(row["validation_f_mae"]),
        "force_rmse_ev_per_angstrom": float(row["validation_f_rmse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--size-transfer-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    csv_path = args.work_dir / "run" / "metrics_epoch.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no validation rows in {csv_path}")

    scan = []
    for row in rows:
        epoch = int(row["epoch"])
        metrics = validation_metrics(row)
        scan.append(
            {
                "epoch": epoch,
                "step": epoch * args.steps_per_epoch,
                **metrics,
                **mev(metrics),
            }
        )
    force_row = min(scan, key=lambda row: (row["force_mae_ev_per_angstrom"], row["step"]))
    energy_row = min(scan, key=lambda row: (row["energy_mae_ev_per_atom"], row["step"]))

    train_text = args.train_log.read_text(errors="replace")
    parameter_matches = re.findall(r"Number of trainable weights:\s*(\d+)", train_text)
    if not parameter_matches:
        raise RuntimeError(f"missing trainable parameter count in {args.train_log}")
    best_model = args.work_dir / "run" / "best_model.pth"
    if not best_model.is_file():
        raise FileNotFoundError(best_model)

    test = parse_test(args.test_log)
    result = {
        "selection_rule": "minimum full-validation Force MAE; test never selects",
        "test_used_for_selection": False,
        "selected_epoch": force_row["epoch"],
        "selected_step": force_row["step"],
        "selected_checkpoint": str(best_model.resolve()),
        "validation_checkpoint_count": len(scan),
        "trainable_parameters": int(parameter_matches[-1]),
        "independent_validation_envelopes": {
            "force_mae": force_row,
            "energy_mae": energy_row,
        },
        "validation_scan": scan,
        "test": {**test, **mev(test)},
    }
    if args.size_transfer_log is not None:
        transfer = parse_test(args.size_transfer_log)
        result["mgo_4x4_size_transfer_test"] = {**transfer, **mev(transfer)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
