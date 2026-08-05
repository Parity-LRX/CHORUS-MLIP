#!/usr/bin/env python3
"""Convert variable-topology ICTC HDF5 splits to grouped DeepMD systems.

Each distinct atomic-number sequence becomes one DeepMD system.  Samples are
not shuffled or reassigned between train, validation, and test: grouping only
adapts the existing split to DeepMD's fixed-topology-per-system format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from ase.data import chemical_symbols


def _sample_key(key: str) -> int:
    return int(key.rsplit("_", maxsplit=1)[-1])


def _topology_name(numbers: tuple[int, ...]) -> str:
    payload = np.asarray(numbers, dtype=np.int16).tobytes()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"n{len(numbers):03d}_{digest}"


def _write_system(
    *,
    handle: h5py.File,
    keys: list[str],
    destination: Path,
    numbers: tuple[int, ...],
    type_map: list[int],
) -> dict[str, object]:
    set_dir = destination / "set.000"
    set_dir.mkdir(parents=True)
    n_frames = len(keys)
    n_atoms = len(numbers)

    coordinates = np.empty((n_frames, 3 * n_atoms), dtype=np.float64)
    forces = np.empty_like(coordinates)
    energies = np.empty((n_frames, 1), dtype=np.float64)
    boxes = np.empty((n_frames, 9), dtype=np.float64)
    for index, key in enumerate(keys):
        group = handle[key]
        candidate = tuple(np.asarray(group["A"], dtype=np.int64).tolist())
        if candidate != numbers:
            raise RuntimeError(f"{key}: topology changed while converting")
        coordinates[index] = np.asarray(group["pos"], dtype=np.float64).reshape(-1)
        forces[index] = np.asarray(group["force"], dtype=np.float64).reshape(-1)
        energies[index, 0] = float(group["y"][()])
        boxes[index] = np.asarray(group["cell"], dtype=np.float64).reshape(-1)

    type_lookup = {
        atomic_number: type_index
        for type_index, atomic_number in enumerate(type_map)
    }
    try:
        atom_types = [type_lookup[number] for number in numbers]
    except KeyError as error:
        raise ValueError(
            f"atomic number {error.args[0]} is absent from --type-map"
        ) from error

    np.save(set_dir / "coord.npy", coordinates)
    np.save(set_dir / "force.npy", forces)
    np.save(set_dir / "energy.npy", energies)
    np.save(set_dir / "box.npy", boxes)
    (destination / "type.raw").write_text(
        " ".join(map(str, atom_types)) + "\n", encoding="utf-8"
    )
    (destination / "type_map.raw").write_text(
        "\n".join(chemical_symbols[number] for number in type_map) + "\n",
        encoding="utf-8",
    )
    (destination / "nopbc").touch()
    return {
        "frames": n_frames,
        "atoms": n_atoms,
        "atomic_numbers": list(numbers),
        "first_source_key": keys[0],
        "last_source_key": keys[-1],
    }


def _convert_split(
    source: Path, destination: Path, type_map: list[int]
) -> dict[str, object]:
    with h5py.File(source, "r") as handle:
        keys = sorted(handle.keys(), key=_sample_key)
        if not keys:
            raise ValueError(f"no samples in {source}")

        grouped: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for key in keys:
            numbers = tuple(np.asarray(handle[key]["A"], dtype=np.int64).tolist())
            grouped[numbers].append(key)

        systems = {}
        for numbers in sorted(grouped, key=lambda value: (len(value), value)):
            name = _topology_name(numbers)
            systems[name] = _write_system(
                handle=handle,
                keys=grouped[numbers],
                destination=destination / name,
                numbers=numbers,
                type_map=type_map,
            )

    return {
        "source": str(source.resolve()),
        "frames": len(keys),
        "topologies": len(systems),
        "systems": systems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type-map", default="1,6,7,8")
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()

    if (args.output_dir / "DONE").is_file():
        print((args.output_dir / "conversion_manifest.json").read_text())
        return
    if args.output_dir.exists():
        raise RuntimeError(
            f"{args.output_dir} exists without DONE; inspect it before retrying"
        )

    type_map = [int(value) for value in args.type_map.split(",")]
    work_dir = args.output_dir.with_name(args.output_dir.name + ".building")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    manifest: dict[str, object] = {
        "format": "deepmd/npy grouped fixed-topology systems",
        "grouping_only": True,
        "samples_reassigned_between_splits": False,
        "type_map_atomic_numbers": type_map,
        "type_map_symbols": [chemical_symbols[number] for number in type_map],
        "splits": {},
    }
    for split in args.splits.split(","):
        manifest["splits"][split] = _convert_split(
            args.source_dir / f"processed_{split}.h5",
            work_dir / split,
            type_map,
        )

    (work_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (work_dir / "DONE").touch()
    work_dir.replace(args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
