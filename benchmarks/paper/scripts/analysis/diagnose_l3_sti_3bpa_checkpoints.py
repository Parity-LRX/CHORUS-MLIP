#!/usr/bin/env python3
"""Diagnostic-only test scans for the L=3 STI and 3BPA controls.

The script samples checkpoints chosen without reading test metrics.  Its output
must not replace the formally reported validation-selected checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path


METRIC_PATTERNS = {
    "energy_mae_ev_per_atom": r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom",
    "energy_rmse_ev_per_atom": r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom",
    "force_mae_ev_per_angstrom": r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A",
    "force_rmse_ev_per_angstrom": r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("phaseoff", "chorus", "persistent"), required=True)
    parser.add_argument("--mode-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--mace-torch-path", type=Path, required=True)
    parser.add_argument("--sti-data-dir", type=Path, required=True)
    parser.add_argument("--three-bpa-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--sti-target-steps",
        default="800,2400,4800,7200,9600,12000,16000,24000",
    )
    parser.add_argument(
        "--three-bpa-target-steps",
        default="2800,5600,11200,19600,28000,35000,40000",
    )
    return parser.parse_args()


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "checkpoints" / "loss.csv").open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["kind"] == "epoch"]


def choose_checkpoints(run_dir: Path, target_steps: list[int]) -> list[dict[str, object]]:
    rows = read_rows(run_dir)
    formal = min(rows, key=lambda row: (float(row["val_force_mae"]), int(row["step"])))
    chosen = {int(formal["step"]): formal}
    for target in target_steps:
        row = min(rows, key=lambda item: (abs(int(item["step"]) - target), int(item["step"])))
        chosen[int(row["step"])] = row

    result = []
    for step, row in sorted(chosen.items()):
        epoch = int(row["epoch"])
        candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
        if len(candidates) != 1:
            raise RuntimeError(f"checkpoint mismatch for epoch={epoch}, step={step}: {candidates}")
        result.append(
            {
                "epoch": epoch,
                "step": step,
                "checkpoint": str(candidates[0]),
                "formal_validation_choice": step == int(formal["step"]),
                "validation": {
                    key: float(row[key])
                    for key in (
                        "val_energy_mae",
                        "val_energy_rmse",
                        "val_force_mae",
                        "val_force_rmse",
                    )
                },
            }
        )
    return result


def phase_args(mode: str) -> list[str]:
    if mode == "phaseoff":
        return ["--phase-mode", "none"]
    scope = "persistent" if mode == "persistent" else "final"
    return [
        "--phase-mode", "final-full-l-residual",
        "--phase-amplitude", "softplus",
        "--phase-coefficient", "polar",
        "--phase-context", "content",
        "--phase-density-pairs", "full-nonlinear",
        "--phase-normalization", "avg-neighbors",
        "--phase-placement", "pre-product-full-l",
        "--phase-scope", scope,
    ]


def evaluate(
    args: argparse.Namespace,
    checkpoint: str,
    data_dir: Path,
    val_prefix: str,
    batch_size: int,
    avg_neighbors: str,
    e0_keys: str,
    e0_values: str,
    destination: Path,
) -> dict[str, float]:
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    command = [
        args.python_bin, "-m", "chorus.cli.train",
        "--data-dir", str(data_dir), "--train-prefix", "train", "--val-prefix", val_prefix,
        "--channels", "128", "--lmax", "3", "--max-ell", "3",
        "--num-interaction", "2", "--correlation", "3",
        "--product-backend", "ictd-bridge-u", "--angular-basis", "ictd",
        "--use-reduced-cg", "--first-layer-self-connection",
        "--mace-compatible-random-init", "--readout-hidden-channels", "64",
        "--function-type", "bessel", "--num-basis", "8",
        "--polynomial-cutoff-p", "6", "--max-radius", "5.0",
        "--avg-num-neighbors", avg_neighbors,
        "--atomic-energy-keys", e0_keys, f"--atomic-energy-values={e0_values}",
        "--scaling", "std_scaling", "--epochs", "1", "--batch-size", str(batch_size),
        "--dtype", "float32", "--device", "cuda", "--num-workers", "0",
        "--loss", "mse", "--energy-weight", "1", "--force-weight", "100",
        "--stress-weight", "0", "--phase-hidden-channels", "32",
        "--phase-scale-init", "0.05", "--phase-density-rank", "16",
        "--seed", "20260616", *phase_args(args.mode),
        "--resume-checkpoint", checkpoint, "--eval-only",
        "--checkpoint", str(destination / "unused.pth"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        (str(args.repo), str(args.mace_torch_path), env.get("PYTHONPATH", ""))
    )
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    log_path = destination / "test.log"
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)

    text = log_path.read_text(errors="replace")
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"missing {key} in {log_path}")
        metrics[key] = float(matches[-1])
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def find_3bpa_run(mode_root: Path) -> Path:
    candidates = list((mode_root / "3bpa").glob("*/checkpoints/loss.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one 3BPA run under {mode_root}, got {candidates}")
    return candidates[0].parents[1]


def main() -> None:
    args = parse_args()
    output = args.output_dir or args.mode_root / "diagnostic_test_scan_l3_overfit_20260815"
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "purpose": "diagnostic-only checkpoint scan; test metrics are not used for formal selection",
        "mode": args.mode,
        "configuration": "C128, lmax=max_ell=3, interactions=2, correlation=3, rank=16",
        "systems": {},
    }

    sti_run = args.mode_root / "sti"
    sti_points = choose_checkpoints(
        sti_run, [int(value) for value in args.sti_target_steps.split(",")]
    )
    for point in sti_points:
        point["test"] = evaluate(
            args, str(point["checkpoint"]), args.sti_data_dir, "test", 16,
            "16.62877403846154", "1,6", "-518.6243286132812,-605.061767578125",
            output / "sti" / f"step{point['step']}",
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["systems"]["sti"] = sti_points

    bpa_run = find_3bpa_run(args.mode_root)
    bpa_points = choose_checkpoints(
        bpa_run, [int(value) for value in args.three_bpa_target_steps.split(",")]
    )
    for point in bpa_points:
        point["tests"] = {}
        for temperature in ("300K", "600K", "1200K"):
            point["tests"][temperature] = evaluate(
                args, str(point["checkpoint"]), args.three_bpa_data_dir,
                f"test_{temperature}", 16, "16.712427983539094", "1,6,7,8",
                "-723.2941476475917,-723.2941476475917,-120.549024607932,-60.27451230396598",
                output / "3bpa" / f"step{point['step']}" / temperature,
            )
        summary["systems"]["3bpa"] = bpa_points
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    (output / "DONE").touch()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
