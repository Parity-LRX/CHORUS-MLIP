#!/usr/bin/env python3
"""Attach CHORUS test metrics to an audited validation selection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "energy_mae_ev_per_atom": r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom",
    "energy_rmse_ev_per_atom": r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom",
    "force_mae_ev_per_angstrom": r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A",
    "force_rmse_ev_per_angstrom": r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument(
        "--train-log",
        type=Path,
        help="optional training log used to audit the trainable parameter count",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.test_log.read_text(errors="replace")
    test = {}
    for key, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"missing {key} in {args.test_log}")
        test[key] = float(matches[-1])
    result = json.loads(args.selection.read_text())
    if args.train_log is not None:
        train_text = args.train_log.read_text(errors="replace")
        parameter_matches = re.findall(r"\bparams=(\d+)\b", train_text)
        if not parameter_matches:
            raise RuntimeError(f"missing parameter count in {args.train_log}")
        result["trainable_parameters"] = int(parameter_matches[-1])
    result["test"] = test
    result["test_mev"] = {
        key.replace("_ev_", "_mev_"): value * 1000.0 for key, value in test.items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
