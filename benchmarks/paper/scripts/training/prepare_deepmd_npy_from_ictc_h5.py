#!/usr/bin/env python3
"""Convert fixed-topology ICTC HDF5 splits to DeePMD NumPy systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


def _sample_key(key: str) -> int:
    return int(key.rsplit("_", maxsplit=1)[-1])


def convert_split(source: Path, destination: Path, type_map: list[int]) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    set_dir = destination / "set.000"
    set_dir.mkdir(exist_ok=True)

    with h5py.File(source, "r") as handle:
        keys = sorted(handle.keys(), key=_sample_key)
        if not keys:
            raise ValueError(f"no samples in {source}")

        atomic_numbers = np.asarray(handle[keys[0]]["A"], dtype=np.int64)
        for key in keys[1:]:
            candidate = np.asarray(handle[key]["A"], dtype=np.int64)
            if not np.array_equal(candidate, atomic_numbers):
                raise ValueError(
                    f"{source} is not fixed-topology: {key} has a different atom sequence"
                )

        coords = np.stack(
            [np.asarray(handle[key]["pos"], dtype=np.float64) for key in keys]
        )
        forces = np.stack(
            [np.asarray(handle[key]["force"], dtype=np.float64) for key in keys]
        )
        energies = np.asarray(
            [float(handle[key]["y"][()]) for key in keys], dtype=np.float64
        ).reshape(-1, 1)
        boxes = np.stack(
            [np.asarray(handle[key]["cell"], dtype=np.float64) for key in keys]
        )

    type_lookup = {atomic_number: index for index, atomic_number in enumerate(type_map)}
    try:
        atom_types = np.asarray(
            [type_lookup[int(value)] for value in atomic_numbers], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"atomic number {error.args[0]} is absent from --type-map") from error

    np.save(set_dir / "coord.npy", coords.reshape(coords.shape[0], -1))
    np.save(set_dir / "force.npy", forces.reshape(forces.shape[0], -1))
    np.save(set_dir / "energy.npy", energies)
    np.save(set_dir / "box.npy", boxes.reshape(boxes.shape[0], -1))
    (destination / "type.raw").write_text(
        " ".join(map(str, atom_types.tolist())) + "\n", encoding="utf-8"
    )
    (destination / "type_map.raw").write_text(
        "\n".join(SYMBOLS[value] for value in type_map) + "\n", encoding="utf-8"
    )
    (destination / "nopbc").touch()

    return {
        "frames": int(coords.shape[0]),
        "atoms": int(coords.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type-map", default="1,6")
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()

    type_map = [int(value) for value in args.type_map.split(",")]
    manifest: dict[str, object] = {
        "source_dir": str(args.source_dir.resolve()),
        "type_map_atomic_numbers": type_map,
        "type_map_symbols": [SYMBOLS[value] for value in type_map],
        "splits": {},
    }
    for split in args.splits.split(","):
        source = args.source_dir / f"processed_{split}.h5"
        destination = args.output_dir / split
        manifest["splits"][split] = convert_split(source, destination, type_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
