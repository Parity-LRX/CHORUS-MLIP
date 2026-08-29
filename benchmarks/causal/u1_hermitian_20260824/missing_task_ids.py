#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("mal", "bucky", "3bpa", "zro")
MODES = ("full", "charge2", "diagonal", "off")
SEEDS = (20260616, 20260617, 20260618, 20260619, 20260620)


def task_record(task_id: int, run_root: Path) -> dict[str, object]:
    mode = MODES[task_id % 4]
    dataset = DATASETS[(task_id // 4) % 4]
    seed = SEEDS[task_id // 16]
    run_dir = run_root / dataset / f"seed{seed}" / mode
    return {
        "task_id": task_id,
        "dataset": dataset,
        "seed": seed,
        "mode": mode,
        "run_dir": str(run_dir),
        "complete": (run_dir / "metrics.json").is_file() and (run_dir / "DONE").is_file(),
        "failed_marker": (run_dir / "FAILED").is_file(),
        "has_recovery": (run_dir / "checkpoints" / "model.recovery.pth").is_file(),
        "training_done": (run_dir / "TRAIN_DONE").is_file(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--format", choices=("ids", "json"), default="ids")
    args = ap.parse_args()
    missing = [
        record
        for task_id in range(80)
        if not (record := task_record(task_id, args.run_root))["complete"]
    ]
    if args.format == "json":
        print(json.dumps(missing, indent=2))
    else:
        print(",".join(str(record["task_id"]) for record in missing))


if __name__ == "__main__":
    main()
