#!/usr/bin/env python3
"""Convert fixed-topology extxyz splits to DeepMD ``npy`` systems.

This converter is intentionally strict: every frame within one split must use
the same atom ordering and the train/validation/test splits must share that
ordering.  That is true for the official temporal xxMD molecule splits and
prevents an unnoticed topology mismatch from entering the DPA-4 comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols
from ase.io import iread


def _read_split(path: Path, expected_numbers: np.ndarray | None):
    coordinates: list[np.ndarray] = []
    energies: list[float] = []
    forces: list[np.ndarray] = []
    reference_numbers = expected_numbers

    for frame_index, atoms in enumerate(iread(path, index=":")):
        numbers = np.asarray(atoms.numbers, dtype=np.int64)
        if reference_numbers is None:
            reference_numbers = numbers.copy()
        if not np.array_equal(numbers, reference_numbers):
            raise ValueError(
                f"{path}: atom ordering differs at frame {frame_index}; "
                "DPA fixed-topology conversion is unsafe"
            )
        coordinates.append(np.asarray(atoms.positions, dtype=np.float64).reshape(-1))
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1))

    if not coordinates or reference_numbers is None:
        raise ValueError(f"{path}: no structures found")
    return (
        np.stack(coordinates),
        np.asarray(energies, dtype=np.float64).reshape(-1, 1),
        np.stack(forces),
        reference_numbers,
    )


def _write_split(
    destination: Path,
    coordinates: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    type_ids: np.ndarray,
    type_map: list[str],
) -> None:
    set_dir = destination / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    n_frames = coordinates.shape[0]
    boxes = np.tile(np.diag([100.0, 100.0, 100.0]).reshape(1, 9), (n_frames, 1))
    np.save(set_dir / "coord.npy", coordinates)
    np.save(set_dir / "energy.npy", energies)
    np.save(set_dir / "force.npy", forces)
    np.save(set_dir / "box.npy", boxes)
    (destination / "type.raw").write_text(
        " ".join(str(value) for value in type_ids.tolist()) + "\n",
        encoding="utf-8",
    )
    (destination / "type_map.raw").write_text(
        "\n".join(type_map) + "\n", encoding="utf-8"
    )
    (destination / "nopbc").write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_paths = {"train": args.train, "val": args.val, "test": args.test}
    split_data = {}
    reference_numbers = None
    for split, path in split_paths.items():
        coords, energies, forces, numbers = _read_split(path, reference_numbers)
        if reference_numbers is None:
            reference_numbers = numbers
        split_data[split] = (coords, energies, forces)

    assert reference_numbers is not None
    unique_numbers = sorted(int(number) for number in np.unique(reference_numbers))
    number_to_type = {number: index for index, number in enumerate(unique_numbers)}
    type_ids = np.asarray(
        [number_to_type[int(number)] for number in reference_numbers], dtype=np.int64
    )
    type_map = [chemical_symbols[number] for number in unique_numbers]

    args.output.mkdir(parents=True, exist_ok=True)
    for split, (coords, energies, forces) in split_data.items():
        _write_split(
            args.output / split, coords, energies, forces, type_ids, type_map
        )

    manifest = {
        "format": "deepmd/npy",
        "fixed_topology_asserted": True,
        "source": {name: str(path.resolve()) for name, path in split_paths.items()},
        "frames": {
            name: int(values[0].shape[0]) for name, values in split_data.items()
        },
        "n_atoms": int(reference_numbers.size),
        "atomic_numbers": reference_numbers.tolist(),
        "type_map": type_map,
        "energy_unit": "eV",
        "force_unit": "eV/angstrom",
        "periodic": False,
    }
    (args.output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
