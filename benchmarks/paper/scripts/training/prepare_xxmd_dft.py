#!/usr/bin/env python3
"""Prepare the official xxMD-DFT temporal splits for MACE-ICTC/CHORUS.

The source archive already contains train/validation/test files split by
trajectory time.  This script preserves those files exactly, creates the HDF5
neighbor graphs consumed by the trainer, and records enough metadata to audit
the split and units.  No random re-splitting or unit conversion is performed.

The official xxMD loader passes the extended-XYZ values directly through ASE;
we therefore follow ASE's energy/force convention (eV and eV/Angstrom).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from ase.io import iread
from matscipy.neighbours import neighbour_list


SOURCE_FILES = {
    "azo": {
        "train": "azo_train_uks.xyz",
        "val": "azo_val_uks.xyz",
        "test": "azo_test_uks.xyz",
    },
    "dia": {
        "train": "dia_train_uks.xyz",
        "val": "dia_val_uks.xyz",
        "test": "dia_test_uks.xyz",
    },
    "mal": {
        "train": "mal_train_uks.xyz",
        "val": "mal_val_uks.xyz",
        "test": "mal_test_uks.xyz",
    },
    "sti": {
        "train": "sti_train_uks.xyz",
        "val": "sti_val_uks.xyz",
        "test": "sti_test_uks.xyz",
    },
}


def csv_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(SOURCE_FILES))
    if unknown:
        raise ValueError(f"unknown xxMD molecule codes: {unknown}")
    return names


def prepare_split(
    source: Path,
    destination_xyz: Path,
    destination_h5: Path,
    *,
    max_radius: float,
) -> dict[str, object]:
    shutil.copy2(source, destination_xyz)

    node_counts: list[int] = []
    edge_counts: list[int] = []
    energies: list[float] = []
    force_abs_max = 0.0
    composition: Counter[int] | None = None
    max_atoms = 0
    max_edges = 0
    cell = np.eye(3, dtype=np.float64) * 100.0
    pbc = (False, False, False)

    with h5py.File(destination_h5, "w") as h5:
        for frame_index, atoms in enumerate(iread(str(source), index=":")):
            if bool(np.asarray(atoms.pbc).any()):
                raise ValueError(f"xxMD frame unexpectedly periodic: {source} frame {frame_index}")
            pos = np.asarray(atoms.positions, dtype=np.float64)
            z = np.asarray(atoms.numbers, dtype=np.int64)
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError(f"non-finite label: {source} frame {frame_index}")

            frame_composition = Counter(int(value) for value in z)
            if composition is None:
                composition = frame_composition
            elif frame_composition != composition:
                raise ValueError(f"composition changes within {source}")

            edge_src, edge_dst, shifts = neighbour_list(
                "ijS",
                positions=pos,
                cell=cell,
                pbc=pbc,
                cutoff=float(max_radius),
            )
            edge_src = np.asarray(edge_src, dtype=np.int64)
            edge_dst = np.asarray(edge_dst, dtype=np.int64)
            shifts = np.asarray(shifts, dtype=np.float64)
            if shifts.size and np.any(shifts != 0.0):
                raise ValueError(f"nonzero periodic shift in nonperiodic frame: {source}")
            if edge_src.size:
                edge_length = np.linalg.norm(pos[edge_dst] - pos[edge_src], axis=1)
                if float(edge_length.max()) > float(max_radius) + 1e-6:
                    raise ValueError(f"neighbor edge exceeds cutoff in {source}")

            group = h5.create_group(f"sample_{frame_index}")
            group.create_dataset("pos", data=pos)
            group.create_dataset("A", data=z)
            group.create_dataset("y", data=np.float64(energy))
            group.create_dataset("force", data=forces)
            group.create_dataset("edge_src", data=edge_src)
            group.create_dataset("edge_dst", data=edge_dst)
            group.create_dataset("edge_shifts", data=shifts)
            group.create_dataset("cell", data=cell)
            group.create_dataset("stress", data=np.zeros((3, 3), dtype=np.float64))

            n_atoms = int(len(z))
            n_edges = int(len(edge_src))
            node_counts.append(n_atoms)
            edge_counts.append(n_edges)
            energies.append(energy)
            force_abs_max = max(force_abs_max, float(np.abs(forces).max(initial=0.0)))
            max_atoms = max(max_atoms, n_atoms)
            max_edges = max(max_edges, n_edges)

        h5.attrs["max_atoms"] = int(max_atoms)
        h5.attrs["max_edges"] = int(max_edges)

    if composition is None:
        raise ValueError(f"empty source split: {source}")

    node_array = np.asarray(node_counts, dtype=np.int64)
    edge_array = np.asarray(edge_counts, dtype=np.int64)
    np.savez(
        str(destination_h5) + ".counts.npz",
        node_counts=node_array,
        edge_counts=edge_array,
    )
    energy_array = np.asarray(energies, dtype=np.float64)
    return {
        "source": str(source),
        "extxyz": str(destination_xyz),
        "processed_h5": str(destination_h5),
        "frames": int(len(node_array)),
        "composition": {str(z): int(count) for z, count in sorted(composition.items())},
        "atomic_numbers": sorted(int(z) for z in composition),
        "atoms_per_frame": int(node_array[0]),
        "mean_directed_neighbors": float(edge_array.sum() / node_array.sum()),
        "max_edges": int(max_edges),
        "energy_min_eV": float(energy_array.min()),
        "energy_max_eV": float(energy_array.max()),
        "force_abs_max_eV_A": float(force_abs_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing azo/raw, dia/raw, mal/raw, and sti/raw.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--molecules", default="azo,dia,mal,sti")
    parser.add_argument("--max-radius", type=float, default=5.0)
    args = parser.parse_args()

    molecules = csv_names(args.molecules)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "dataset": "xxMD-DFT",
        "source": "https://doi.org/10.5281/zenodo.10393859",
        "split_policy": "official trajectory-temporal train/validation/test split; preserved exactly",
        "labels": {
            "electronic_structure": "M06 ground-state DFT",
            "energy_unit": "eV (ASE convention; no conversion)",
            "force_unit": "eV/Angstrom (ASE convention; no conversion)",
        },
        "max_radius_angstrom": float(args.max_radius),
        "molecules": {},
    }

    for molecule in molecules:
        output_dir = args.output_root / molecule
        output_dir.mkdir(parents=True, exist_ok=True)
        molecule_metadata: dict[str, object] = {"code": molecule, "splits": {}}
        for split, source_name in SOURCE_FILES[molecule].items():
            source = args.source_root / molecule / "raw" / source_name
            if not source.is_file():
                raise FileNotFoundError(source)
            split_metadata = prepare_split(
                source,
                output_dir / f"{split}.extxyz",
                output_dir / f"processed_{split}.h5",
                max_radius=float(args.max_radius),
            )
            molecule_metadata["splits"][split] = split_metadata
            print(
                f"prepared {molecule}/{split}: {split_metadata['frames']} frames, "
                f"avg_neighbors={split_metadata['mean_directed_neighbors']:.6f}",
                flush=True,
            )
        (output_dir / "metadata.json").write_text(
            json.dumps(molecule_metadata, indent=2, sort_keys=True) + "\n"
        )
        manifest["molecules"][molecule] = molecule_metadata

    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
