#!/usr/bin/env python3
"""Stream an ICTC HDF5 split to extxyz without loading it all into memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz


def _sample_key(key: str) -> int:
    return int(key.rsplit("_", maxsplit=1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-key", default="energy")
    parser.add_argument("--forces-key", default="forces")
    args = parser.parse_args()

    done = args.output.with_suffix(args.output.suffix + ".DONE")
    if args.output.is_file() and done.is_file():
        print(f"REUSE_DONE {args.output}")
        return
    if args.output.exists():
        raise RuntimeError(f"{args.output} exists without {done.name}")

    temporary = args.output.with_suffix(args.output.suffix + ".building")
    temporary.unlink(missing_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as handle:
        keys = sorted(handle.keys(), key=_sample_key)

        def frames():
            for key in keys:
                group = handle[key]
                atoms = Atoms(
                    numbers=np.asarray(group["A"], dtype=np.int64),
                    positions=np.asarray(group["pos"], dtype=np.float64),
                    cell=np.asarray(group["cell"], dtype=np.float64),
                    pbc=False,
                )
                atoms.info[args.energy_key] = float(group["y"][()])
                atoms.arrays[args.forces_key] = np.asarray(
                    group["force"], dtype=np.float64
                )
                yield atoms

        with temporary.open("w", encoding="utf-8") as output:
            write_extxyz(output, frames())

    temporary.replace(args.output)
    done.touch()
    print(f"WROTE {len(keys)} frames to {args.output}")


if __name__ == "__main__":
    main()
