#!/usr/bin/env python3
"""Preprocess a fixed MD22 held-out file as ``processed_test.h5``.

Unlike the generic train/validation preprocessor, this script does not split or
fit anything on the test structures.  The correction-energy side data use the
E0 values fitted on the training set, while the processed H5 retains the raw
total energies consumed by the trainer/evaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chorus.data.preprocessing import (
    compute_correction,
    extract_data_blocks,
    save_set,
    save_to_h5_parallel,
)


def csv_ints(value: str) -> np.ndarray:
    return np.asarray([int(x) for x in value.split(",") if x.strip()], dtype=np.int64)


def csv_floats(value: str) -> np.ndarray:
    return np.asarray([float(x) for x in value.split(",") if x.strip()], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atomic-energy-keys", default="1,6")
    parser.add_argument("--atomic-energy-values", required=True)
    parser.add_argument("--max-radius", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    keys = csv_ints(args.atomic_energy_keys)
    values = csv_floats(args.atomic_energy_values)
    if len(keys) != len(values):
        raise ValueError("atomic-energy keys and values must have equal length")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocks, _, raw_energy, cells, pbcs, stresses = extract_data_blocks(str(args.input))
    indices = np.arange(len(blocks), dtype=np.int64)
    correction = compute_correction(blocks, raw_energy, keys, values)
    save_set(
        "test",
        indices,
        blocks,
        raw_energy,
        correction,
        cells,
        pbc_list=pbcs,
        stress_list=stresses,
        output_dir=str(args.output_dir),
    )
    save_to_h5_parallel(
        "test",
        args.max_radius,
        args.num_workers,
        data_dir=str(args.output_dir),
    )
    print(f"prepared {len(blocks)} held-out frames as processed_test.h5")


if __name__ == "__main__":
    main()
