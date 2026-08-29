#!/usr/bin/env python3
"""Summarize a train-only Zr--O CHORUS energy-residual calibration."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


METRIC_PATTERNS = {
    "energy_mae_ev_per_atom": (
        r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom"
    ),
    "energy_rmse_ev_per_atom": (
        r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom"
    ),
    "force_mae_ev_per_angstrom": r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A",
    "force_rmse_ev_per_angstrom": r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A",
}


def parse_eval(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    metrics: dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"missing {key} in {path}")
        metrics[key] = float(matches[-1])
    metrics.update(
        {
            key.replace("_ev_", "_mev_"): 1000.0 * value
            for key, value in list(metrics.items())
        }
    )
    return metrics


def calibration_report(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    matches = re.findall(
        r"closed-form train-only element calibration:\s*(\{.*\})",
        text,
    )
    if not matches:
        raise RuntimeError(f"missing calibration report in {path}")
    report = ast.literal_eval(matches[-1])
    if not isinstance(report, dict):
        raise TypeError("calibration report is not a mapping")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-metrics", type=Path, required=True)
    parser.add_argument("--calibration-log", type=Path, required=True)
    parser.add_argument("--calibrated-val-log", type=Path, required=True)
    parser.add_argument("--calibrated-test-log", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--calibrated-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.raw_metrics.read_text())
    raw_test = raw["test"]
    calibrated_val = parse_eval(args.calibrated_val_log)
    calibrated_test = parse_eval(args.calibrated_test_log)
    report = calibration_report(args.calibration_log)

    raw_force = {
        "mae": float(raw_test["force_mae_ev_per_angstrom"]),
        "rmse": float(raw_test["force_rmse_ev_per_angstrom"]),
    }
    calibrated_force = {
        "mae": calibrated_test["force_mae_ev_per_angstrom"],
        "rmse": calibrated_test["force_rmse_ev_per_angstrom"],
    }
    max_force_difference = max(
        abs(raw_force[key] - calibrated_force[key]) for key in raw_force
    )
    if max_force_difference > 2.0e-6:
        raise RuntimeError(
            "force metrics changed beyond evaluation tolerance: "
            f"{max_force_difference:.3e} eV/angstrom"
        )

    result = {
        "selection_rule": raw["selection_rule"],
        "selected_checkpoint": raw["selected_checkpoint"],
        "selected_step": raw["selected_step"],
        "test_used_for_selection": False,
        "calibration_fit_split": "train only",
        "calibration_target": "model-specific residual energy",
        "elements": ["O", "Zr"],
        "forces_changed": False,
        "force_metric_max_abs_recompute_difference_ev_per_angstrom": (
            max_force_difference
        ),
        "checkpoint_audit": json.loads(args.audit.read_text()),
        "calibrated_checkpoint": str(args.calibrated_checkpoint),
        "fit_report": report,
        "validation_calibrated": calibrated_val,
        "test_raw": raw_test,
        "test_calibrated": calibrated_test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
