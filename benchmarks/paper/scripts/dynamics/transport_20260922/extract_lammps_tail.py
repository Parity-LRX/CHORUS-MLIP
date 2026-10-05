#!/usr/bin/env python3
"""Extract and summarize the tail of a LiPS LAMMPS dump.

This diagnostic is intentionally independent of the production runner.  It
streams the dump, keeps only the requested number of final frames, restores
the Li/P/S element mapping, and records the closest periodic pair and maximum
atomic speed in each retained frame.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from ase.io import iread, write
from ase.neighborlist import neighbor_list


TYPE_TO_Z = {1: 3, 2: 15, 3: 16}
Z_TO_SYMBOL = {3: "Li", 15: "P", 16: "S"}


def _restore_elements(atoms) -> None:
    type_ids = np.asarray(atoms.numbers, dtype=np.int64)
    unknown = sorted(set(type_ids.tolist()) - TYPE_TO_Z.keys())
    if unknown:
        raise RuntimeError(f"unknown LAMMPS type identifiers: {unknown}")
    atoms.numbers = np.asarray([TYPE_TO_Z[int(value)] for value in type_ids])
    atoms.pbc = True
    atoms.wrap()


def _frame_summary(atoms) -> dict[str, object]:
    i, j, distances = neighbor_list("ijd", atoms, cutoff=5.0)
    unique = i < j
    i = i[unique]
    j = j[unique]
    distances = distances[unique]
    if distances.size == 0:
        raise RuntimeError("no periodic pairs found within 5 Angstrom")
    argmin = int(np.argmin(distances))
    left = int(atoms.numbers[int(i[argmin])])
    right = int(atoms.numbers[int(j[argmin])])
    velocities = atoms.get_velocities()
    max_speed = None
    if velocities is not None:
        max_speed = float(np.linalg.norm(velocities, axis=1).max())
    return {
        "step": int(atoms.info.get("timestep", -1)),
        "minimum_distance_angstrom": float(distances[argmin]),
        "minimum_pair": sorted([Z_TO_SYMBOL[left], Z_TO_SYMBOL[right]]),
        "minimum_pair_atom_indices_zero_based": [int(i[argmin]), int(j[argmin])],
        "maximum_speed_ase_units": max_speed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-frames", type=int, default=50)
    args = parser.parse_args()

    if args.tail_frames <= 0:
        raise ValueError("--tail-frames must be positive")
    frames = deque(maxlen=args.tail_frames)
    for atoms in iread(args.trajectory, index=":", format="lammps-dump-text"):
        _restore_elements(atoms)
        frames.append(atoms)
    if not frames:
        raise RuntimeError(f"no frames found in {args.trajectory}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeline = [_frame_summary(atoms) for atoms in frames]
    write(args.output_dir / "last_safe_frame.extxyz", frames[-1], format="extxyz")
    report = {
        "trajectory": str(args.trajectory.resolve()),
        "retained_frames": len(frames),
        "timeline": timeline,
        "last_frame": timeline[-1],
    }
    (args.output_dir / "tail_geometry.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
