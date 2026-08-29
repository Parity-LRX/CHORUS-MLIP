#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


METRIC_PATTERNS = {
    "energy_mae_ev_per_atom": re.compile(
        r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom"
    ),
    "energy_rmse_ev_per_atom": re.compile(
        r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom"
    ),
    "force_mae_ev_per_angstrom": re.compile(
        r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A"
    ),
    "force_rmse_ev_per_angstrom": re.compile(
        r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A"
    ),
}


def parse_eval(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    result: dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        values = pattern.findall(text)
        if not values:
            raise RuntimeError(f"missing {key} in {path}")
        result[key] = float(values[-1])
    result.update(
        {
            key.replace("_ev_", "_mev_"): value * 1000.0
            for key, value in tuple(result.items())
        }
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    loss_path = args.run_dir / "checkpoints" / "loss.csv"
    with loss_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validation = [row for row in rows if row["val_force_mae"]]
    if not validation:
        raise RuntimeError(f"no validation rows in {loss_path}")
    best_force = min(
        validation,
        key=lambda row: (float(row["val_force_mae"]), int(row["step"])),
    )
    best_energy = min(
        validation,
        key=lambda row: (float(row["val_energy_mae"]), int(row["step"])),
    )

    train_text = (args.run_dir / "logs" / "train.log").read_text(errors="replace")
    param_matches = re.findall(r"\bparams=(\d+)", train_text)
    if not param_matches:
        raise RuntimeError("parameter count missing from training log")

    tests = {
        path.parent.name: parse_eval(path)
        for path in sorted((args.run_dir / "eval").glob("*/test.log"))
    }
    if not tests:
        raise RuntimeError("no test logs found")

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    result = {
        **manifest,
        "parameter_count": int(param_matches[-1]),
        "checkpoint_selection": {
            "rule": "minimum validation Force MAE; earliest validation breaks ties",
            "test_used_for_selection": False,
            "validation_checkpoint_count": len(validation),
            "force_mae_envelope": {
                "epoch": int(best_force["epoch"]),
                "step": int(best_force["step"]),
                "energy_mae_mev_per_atom": float(best_force["val_energy_mae"]) * 1000.0,
                "energy_rmse_mev_per_atom": float(best_force["val_energy_rmse"]) * 1000.0,
                "force_mae_mev_per_angstrom": float(best_force["val_force_mae"]) * 1000.0,
                "force_rmse_mev_per_angstrom": float(best_force["val_force_rmse"]) * 1000.0,
            },
            "energy_mae_envelope_diagnostic_only": {
                "epoch": int(best_energy["epoch"]),
                "step": int(best_energy["step"]),
                "energy_mae_mev_per_atom": float(best_energy["val_energy_mae"]) * 1000.0,
                "force_mae_mev_per_angstrom": float(best_energy["val_force_mae"]) * 1000.0,
            },
        },
        "tests": tests,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
