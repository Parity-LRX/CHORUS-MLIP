#!/usr/bin/env python3
"""Create a deterministic energy-stratified MD22 low-data source file.

The resulting extxyz contains only the candidate train/validation pool.  The
remaining source frames are recorded as a disjoint test pool.  The ordinary
ICTC preprocessor subsequently splits the candidate pool 50:50, yielding 600
train and 600 validation frames with the default arguments below.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ENERGY_RE = re.compile(r"(?:^|\s)(?:energy|Energy)=([^\s]+)")


def scan_extxyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    offsets: list[int] = []
    energies: list[float] = []
    with path.open("r") as handle:
        while True:
            offset = handle.tell()
            header = handle.readline()
            if not header:
                break
            n_atoms = int(header.split()[0])
            comment = handle.readline()
            match = ENERGY_RE.search(comment)
            if match is None:
                raise ValueError(f"missing energy at frame {len(offsets)}")
            offsets.append(offset)
            energies.append(float(match.group(1).strip('"')))
            for _ in range(n_atoms):
                if not handle.readline():
                    raise ValueError(f"truncated frame {len(offsets) - 1}")
    return np.asarray(offsets, dtype=np.int64), np.asarray(energies, dtype=np.float64)


def stratified_indices(
    energies: np.ndarray,
    count: int,
    bins: int,
    seed: int,
) -> np.ndarray:
    if count <= 0 or count > len(energies):
        raise ValueError(f"invalid subset size {count} for {len(energies)} frames")
    order = np.argsort(energies, kind="stable")
    strata = np.array_split(order, min(int(bins), len(order)))
    exact = np.asarray([count * len(group) / len(order) for group in strata])
    allocation = np.floor(exact).astype(int)
    remainder = count - int(allocation.sum())
    if remainder:
        fractional_order = np.argsort(-(exact - allocation), kind="stable")
        allocation[fractional_order[:remainder]] += 1

    rng = np.random.default_rng(seed)
    selected = [
        rng.choice(group, size=n_take, replace=False)
        for group, n_take in zip(strata, allocation)
        if n_take
    ]
    result = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
    if len(result) != count or len(np.unique(result)) != count:
        raise RuntimeError("stratified selection did not produce the requested unique count")
    return result


def write_selected(
    source: Path,
    destination: Path,
    offsets: np.ndarray,
    selected: np.ndarray,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r") as src, destination.open("w") as dst:
        for frame_index in selected:
            src.seek(int(offsets[int(frame_index)]))
            header = src.readline()
            n_atoms = int(header.split()[0])
            dst.write(header)
            dst.write(src.readline())
            for _ in range(n_atoms):
                dst.write(src.readline())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=1200)
    parser.add_argument("--energy-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument(
        "--write-heldout",
        action="store_true",
        help="Also materialize the disjoint held-out frames as heldout_test.extxyz.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets, energies = scan_extxyz(args.input)
    selected = stratified_indices(
        energies,
        count=args.pool_size,
        bins=args.energy_bins,
        seed=args.seed,
    )
    heldout = np.setdiff1d(np.arange(len(energies), dtype=np.int64), selected)

    subset_path = args.output_dir / "candidate_1200.extxyz"
    write_selected(args.input, subset_path, offsets, selected)
    heldout_path = args.output_dir / "heldout_test.extxyz"
    if args.write_heldout:
        write_selected(args.input, heldout_path, offsets, heldout)
    np.save(args.output_dir / "candidate_source_indices.npy", selected)
    np.save(args.output_dir / "test_source_indices.npy", heldout)
    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(
            {
                "source": str(args.input),
                "source_frames": int(len(energies)),
                "candidate_frames": int(len(selected)),
                "heldout_test_frames": int(len(heldout)),
                "selection": "energy-stratified candidate pool",
                "energy_bins": int(args.energy_bins),
                "seed": int(args.seed),
                "candidate_energy_min": float(energies[selected].min()),
                "candidate_energy_max": float(energies[selected].max()),
                "source_energy_min": float(energies.min()),
                "source_energy_max": float(energies.max()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {len(selected)} candidate frames to {subset_path}")
    print(f"reserved {len(heldout)} disjoint source frames for final testing")
    if args.write_heldout:
        print(f"wrote held-out test frames to {heldout_path}")


if __name__ == "__main__":
    main()
