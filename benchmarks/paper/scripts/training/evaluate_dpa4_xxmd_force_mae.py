#!/usr/bin/env python3
"""Scan DPA-4 checkpoints on full validation and test the Force-MAE winner."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


METRICS = {
    "energy_mae_ev_per_atom": re.compile(r"Energy MAE/Natoms\s*:\s*([0-9.eE+-]+)"),
    "energy_rmse_ev_per_atom": re.compile(r"Energy RMSE/Natoms\s*:\s*([0-9.eE+-]+)"),
    "force_mae_ev_per_angstrom": re.compile(r"Force\s+MAE\s*:\s*([0-9.eE+-]+)"),
    "force_rmse_ev_per_angstrom": re.compile(r"Force\s+RMSE\s*:\s*([0-9.eE+-]+)"),
}


def parse_metrics(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, float] = {}
    for name, pattern in METRICS.items():
        matches = pattern.findall(text)
        if not matches:
            raise RuntimeError(f"missing {name} in {log_path}")
        # ``dp test`` prints each recursively discovered system followed by a
        # global weighted summary.  The final occurrence is therefore the
        # correct full-split metric for both one- and multi-system datasets.
        result[name] = float(matches[-1])
    result["energy_mae_mev_per_atom"] = 1000.0 * result["energy_mae_ev_per_atom"]
    result["energy_rmse_mev_per_atom"] = 1000.0 * result["energy_rmse_ev_per_atom"]
    result["force_mae_mev_per_angstrom"] = (
        1000.0 * result["force_mae_ev_per_angstrom"]
    )
    result["force_rmse_mev_per_angstrom"] = (
        1000.0 * result["force_rmse_ev_per_angstrom"]
    )
    return result


def checkpoint_step(path: Path) -> int:
    match = re.search(r"-(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint step from {path}")
    return int(match.group(1))


def evaluate(
    *,
    dp: Path,
    checkpoint: Path,
    system: Path,
    log_path: Path,
    detail_prefix: Path | None = None,
) -> None:
    command = [
        str(dp),
        "--pt",
        "test",
        "-m",
        str(checkpoint),
        "-s",
        str(system),
        "-n",
        "0",
    ]
    if detail_prefix is not None:
        command.extend(["-d", str(detail_prefix)])
    environment = os.environ.copy()
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    environment["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    environment.setdefault("DP_INFER_BATCH_SIZE", "8192")
    temporary_log = log_path.with_suffix(log_path.suffix + ".tmp")
    with temporary_log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    temporary_log.replace(log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="training step budget; inferred when the run has one ckpt_steps* dir",
    )
    parser.add_argument(
        "--dp",
        type=Path,
        default=Path("/home/ylzhang/venvs/dpa4-master/bin/dp"),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "evaluate every Nth saved checkpoint; the final checkpoint is "
            "always included"
        ),
    )
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")

    if args.steps is None:
        checkpoint_dirs = sorted(args.run.glob("ckpt_steps*"))
        if len(checkpoint_dirs) != 1:
            raise RuntimeError(
                f"expected one ckpt_steps* directory below {args.run}, "
                f"got {checkpoint_dirs}; pass --steps"
            )
        checkpoint_dir = checkpoint_dirs[0]
    else:
        checkpoint_dir = args.run / f"ckpt_steps{args.steps}"
    checkpoints = sorted(checkpoint_dir.glob("model.ckpt-*.pt"), key=checkpoint_step)
    if not checkpoints:
        raise RuntimeError(f"no checkpoints found under {args.run}")
    if args.checkpoint_every > 1:
        checkpoints = checkpoints[:: args.checkpoint_every]
        final_checkpoint = max(
            checkpoint_dir.glob("model.ckpt-*.pt"), key=checkpoint_step
        )
        if checkpoints[-1] != final_checkpoint:
            checkpoints.append(final_checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)

    validation_rows = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        log_path = args.out / f"val_{step}.log"
        try:
            metrics = parse_metrics(log_path)
        except (FileNotFoundError, RuntimeError):
            log_path.unlink(missing_ok=True)
            evaluate(
                dp=args.dp,
                checkpoint=checkpoint,
                system=args.data / "val",
                log_path=log_path,
            )
            metrics = parse_metrics(log_path)
        validation_rows.append(
            {"step": step, "checkpoint": str(checkpoint), **metrics}
        )
        print(
            f"VAL step={step} "
            f"Fmae={metrics['force_mae_mev_per_angstrom']:.6f} "
            f"Frmse={metrics['force_rmse_mev_per_angstrom']:.6f}",
            flush=True,
        )

    selected = min(
        validation_rows,
        key=lambda row: (row["force_mae_mev_per_angstrom"], row["step"]),
    )
    selected_checkpoint = Path(selected["checkpoint"])
    test_log = args.out / f"test_selected_{selected['step']}.log"
    try:
        test_metrics = parse_metrics(test_log)
    except (FileNotFoundError, RuntimeError):
        test_log.unlink(missing_ok=True)
        evaluate(
            dp=args.dp,
            checkpoint=selected_checkpoint,
            system=args.data / "test",
            log_path=test_log,
            detail_prefix=args.out / f"test_selected_{selected['step']}",
        )
        test_metrics = parse_metrics(test_log)
    result = {
        "selection_rule": "minimum full-validation Force MAE; test never selects",
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_count": len(checkpoints),
        "validation_scan": validation_rows,
        "selected": selected,
        "test": {"step": selected["step"], **test_metrics},
    }
    (args.out / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "DONE").touch()
    print(json.dumps({"selected": selected, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
