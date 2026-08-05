#!/usr/bin/env python3
"""Prepare the standard 3BPA temperature-extrapolation benchmark.

The public 300 K training pool contains 500 configurations.  Following the
MACE benchmark protocol, this script makes a deterministic 450/50 train/valid
split and leaves the public 300/600/1200 K test sets untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from ase.io import read, write
from matscipy.neighbours import neighbour_list


EXPECTED_SOURCE_COMMIT = "29e6d467317e4b5967b7ea5cbee54de953fa0d45"
TEST_FILES = {
    "300K": "test_300K.xyz",
    "600K": "test_600K.xyz",
    "1200K": "test_1200K.xyz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frames(frames, *, label: str) -> dict:
    if not frames:
        raise ValueError(f"{label} is empty")
    reference_numbers = frames[0].get_atomic_numbers()
    composition = Counter(int(z) for z in reference_numbers)
    energies = []
    force_norms = []
    for index, atoms in enumerate(frames):
        if not np.array_equal(atoms.get_atomic_numbers(), reference_numbers):
            raise ValueError(f"{label} frame {index} changes topology or atom order")
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        if forces.shape != (len(atoms), 3):
            raise ValueError(f"{label} frame {index} has force shape {forces.shape}")
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise ValueError(f"{label} frame {index} contains non-finite labels")
        if np.any(atoms.get_pbc()):
            raise ValueError(f"{label} frame {index} is unexpectedly periodic")
        energies.append(energy)
        force_norms.extend(np.linalg.norm(forces, axis=1).tolist())
    return {
        "frames": len(frames),
        "atoms_per_frame": int(len(frames[0])),
        "composition": {str(z): int(n) for z, n in sorted(composition.items())},
        "energy_min_ev": float(np.min(energies)),
        "energy_max_ev": float(np.max(energies)),
        "force_norm_max_ev_per_angstrom": float(np.max(force_norms)),
    }


def write_h5(path: Path, frames, *, max_radius: float) -> dict:
    node_counts = np.empty(len(frames), dtype=np.int64)
    edge_counts = np.empty(len(frames), dtype=np.int64)
    max_atoms = 0
    max_edges = 0
    edge_total = 0
    with h5py.File(path, "w") as h5:
        for index, atoms in enumerate(frames):
            positions = np.asarray(atoms.get_positions(), dtype=np.float64)
            numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
            pbc = tuple(bool(value) for value in atoms.get_pbc())
            src, dst, shifts = neighbour_list(
                "ijS",
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=float(max_radius),
            )
            src = np.asarray(src, dtype=np.int64)
            dst = np.asarray(dst, dtype=np.int64)
            shifts = np.asarray(shifts, dtype=np.float64)
            if not any(pbc):
                shifts = np.zeros_like(shifts)
            edge_vectors = positions[dst] - positions[src] + shifts @ cell
            if edge_vectors.size:
                longest = float(np.linalg.norm(edge_vectors, axis=1).max())
                if longest > float(max_radius) + 1e-6:
                    raise ValueError(
                        f"{path.name} frame {index}: edge {longest:.6f} exceeds cutoff"
                    )
            group = h5.create_group(f"sample_{index}")
            group.create_dataset("pos", data=positions)
            group.create_dataset("A", data=numbers)
            group.create_dataset("y", data=np.float64(atoms.get_potential_energy()))
            group.create_dataset("force", data=forces)
            group.create_dataset("edge_src", data=src)
            group.create_dataset("edge_dst", data=dst)
            group.create_dataset("edge_shifts", data=shifts)
            group.create_dataset("cell", data=cell)
            group.create_dataset("stress", data=np.zeros((3, 3), dtype=np.float64))
            node_counts[index] = len(atoms)
            edge_counts[index] = len(src)
            edge_total += len(src)
            max_atoms = max(max_atoms, len(atoms))
            max_edges = max(max_edges, len(src))
        h5.attrs["max_atoms"] = int(max_atoms)
        h5.attrs["max_edges"] = int(max_edges)
        h5.attrs["max_radius"] = float(max_radius)
    np.savez(
        str(path) + ".counts.npz",
        node_counts=node_counts,
        edge_counts=edge_counts,
    )
    return {
        "max_atoms": int(max_atoms),
        "max_edges": int(max_edges),
        "mean_directed_neighbors": float(edge_total / node_counts.sum()),
    }


def minimum_norm_e0(frames) -> tuple[list[int], list[float]]:
    composition = Counter(int(z) for z in frames[0].get_atomic_numbers())
    mean_energy = float(np.mean([atoms.get_potential_energy() for atoms in frames]))
    denominator = float(sum(count * count for count in composition.values()))
    keys = sorted(composition)
    values = [mean_energy * composition[z] / denominator for z in keys]
    return keys, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--train-size", type=int, default=450)
    parser.add_argument("--valid-size", type=int, default=50)
    parser.add_argument("--max-radius", type=float, default=5.0)
    parser.add_argument("--source-commit", default=EXPECTED_SOURCE_COMMIT)
    args = parser.parse_args()

    source = args.source_root / "dataset_3BPA"
    train_source = source / "train_300K.xyz"
    test_sources = {temperature: source / name for temperature, name in TEST_FILES.items()}
    for path in (train_source, *test_sources.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    pool = read(train_source, index=":", format="extxyz")
    if args.train_size + args.valid_size != len(pool):
        raise ValueError(
            f"standard pool has {len(pool)} frames, requested "
            f"{args.train_size}+{args.valid_size}"
        )
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(pool))
    train_indices = np.sort(permutation[: args.train_size])
    valid_indices = np.sort(permutation[args.train_size :])
    train_frames = [pool[int(index)] for index in train_indices]
    valid_frames = [pool[int(index)] for index in valid_indices]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "split_indices.npz",
        train=train_indices,
        val=valid_indices,
    )
    write(args.out_dir / "train.extxyz", train_frames, format="extxyz")
    write(args.out_dir / "val.extxyz", valid_frames, format="extxyz")
    train_h5 = write_h5(
        args.out_dir / "processed_train.h5",
        train_frames,
        max_radius=args.max_radius,
    )
    valid_h5 = write_h5(
        args.out_dir / "processed_val.h5",
        valid_frames,
        max_radius=args.max_radius,
    )

    test_metadata = {}
    for temperature, source_path in test_sources.items():
        frames = read(source_path, index=":", format="extxyz")
        write(args.out_dir / f"test_{temperature}.extxyz", frames, format="extxyz")
        h5_meta = write_h5(
            args.out_dir / f"processed_test_{temperature}.h5",
            frames,
            max_radius=args.max_radius,
        )
        test_metadata[temperature] = {
            **validate_frames(frames, label=f"test_{temperature}"),
            **h5_meta,
            "source_file": str(source_path),
            "source_sha256": sha256(source_path),
        }

    e0_keys, e0_values = minimum_norm_e0(train_frames)
    metadata = {
        "benchmark": "3BPA standard temperature extrapolation",
        "source_repository": "https://github.com/davkovacs/BOTNet-datasets",
        "source_commit": args.source_commit,
        "protocol_reference": "MACE: 500 structures split into 450 train / 50 validation",
        "selection_split": "300K validation only",
        "test_used_for_selection": False,
        "seed": args.seed,
        "max_radius_angstrom": args.max_radius,
        "source_train_sha256": sha256(train_source),
        "train": {
            **validate_frames(train_frames, label="train"),
            **train_h5,
            "source_indices": train_indices.tolist(),
        },
        "validation": {
            **validate_frames(valid_frames, label="validation"),
            **valid_h5,
            "source_indices": valid_indices.tolist(),
        },
        "tests": test_metadata,
        "atomic_energy_fit": {
            "data_source": "training split only",
            "rule": "minimum-norm fixed-composition average-energy solution",
            "keys": e0_keys,
            "values_ev": e0_values,
        },
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "training.env").write_text(
        "AVG_NEIGHBORS=" + repr(train_h5["mean_directed_neighbors"]) + "\n"
        + "E0_KEYS=" + ",".join(str(value) for value in e0_keys) + "\n"
        + "E0_VALUES=" + ",".join(f"{value:.16g}" for value in e0_values) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
