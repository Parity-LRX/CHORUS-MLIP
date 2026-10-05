#!/usr/bin/env python3
"""Summarize nearest element-pair distances in the LiPS reference splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from ase.io import iread
from ase.neighborlist import neighbor_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {}
    for split in ("train", "val", "test"):
        path = args.data_root / f"{split}.extxyz"
        minima: dict[str, list[float]] = defaultdict(list)
        frame_count = 0
        for atoms in iread(path, index=":", format="extxyz"):
            frame_count += 1
            i, j, distances = neighbor_list("ijd", atoms, cutoff=5.0)
            unique = i < j
            i = i[unique]
            j = j[unique]
            distances = distances[unique]
            symbols = np.asarray(atoms.get_chemical_symbols())
            frame_minima: dict[str, float] = {}
            for left, right, distance in zip(symbols[i], symbols[j], distances):
                key = "-".join(sorted((str(left), str(right))))
                value = float(distance)
                if key not in frame_minima or value < frame_minima[key]:
                    frame_minima[key] = value
            for key, value in frame_minima.items():
                minima[key].append(value)

        split_report: dict[str, object] = {"frames": frame_count, "pairs": {}}
        for key, values in sorted(minima.items()):
            array = np.asarray(values, dtype=np.float64)
            split_report["pairs"][key] = {
                "minimum_angstrom": float(array.min()),
                "q001_angstrom": float(np.quantile(array, 0.001)),
                "q01_angstrom": float(np.quantile(array, 0.01)),
                "q05_angstrom": float(np.quantile(array, 0.05)),
                "median_angstrom": float(np.median(array)),
            }
        report[split] = split_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
