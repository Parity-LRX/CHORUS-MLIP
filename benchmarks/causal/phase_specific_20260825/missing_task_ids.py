#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = (20260616, 20260617, 20260618, 20260619, 20260620)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("probe", "aligned"), required=True)
    args = parser.parse_args()
    missing = []
    for task_id in range(20):
        dataset = DATASETS[task_id // 5]
        seed = SEEDS[task_id % 5]
        path = args.root / dataset / f"seed{seed}" / "DONE"
        if args.kind == "aligned":
            path = args.root / dataset / f"seed{seed}" / "aligned" / "DONE"
        if not path.is_file():
            missing.append(str(task_id))
    print(",".join(missing))


if __name__ == "__main__":
    main()
