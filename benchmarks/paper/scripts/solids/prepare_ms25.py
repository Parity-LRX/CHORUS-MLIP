#!/usr/bin/env python3
"""Prepare official MS25 splits for CHORUS and external-model benchmarks.

No split is generated here.  The official MS25 train/validation/test files are
kept intact, elemental reference energies are fitted on training structures
only, and all provenance/conditioning diagnostics are written to JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

from chorus.data.preprocessing import (
    compute_correction,
    extract_data_blocks,
    fit_baseline_energies,
    save_set,
    save_to_h5_parallel,
)


DATASETS = {
    "hea": {"archive": "HEA", "cutoff": 5.5, "label_source": "EAM"},
    "zr-o": {"archive": "Zr-O", "cutoff": 6.0, "label_source": "DFT-PBE"},
    "mgo-2x2": {"archive": "MgO-2x2", "cutoff": 6.0, "label_source": "DFT-PBE-D3"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_digest(atoms: Atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype="<i2").tobytes())
    digest.update(np.round(atoms.positions, 10).astype("<f8").tobytes())
    digest.update(np.round(atoms.cell.array, 10).astype("<f8").tobytes())
    digest.update(np.asarray(atoms.pbc, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def result_arrays(atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray | None]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    stress = None
    if atoms.calc is not None and "stress" in atoms.calc.results:
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("non-finite energy or force in MS25 frame")
    if stress is not None and not np.isfinite(stress).all():
        raise ValueError("non-finite stress in MS25 frame")
    return energy, forces, stress


def write_extxyz(images: list[Atoms], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    write(temporary, images, format="extxyz", write_results=True)
    temporary.replace(destination)


def count_matrix(images: list[Atoms], elements: np.ndarray) -> np.ndarray:
    columns = {int(z): i for i, z in enumerate(elements)}
    matrix = np.zeros((len(images), len(elements)), dtype=np.float64)
    for row, atoms in enumerate(images):
        values, counts = np.unique(atoms.numbers, return_counts=True)
        for value, count in zip(values, counts):
            matrix[row, columns[int(value)]] = int(count)
    return matrix


def average_neighbours(images: list[Atoms], cutoff: float) -> float:
    atoms_total = 0
    directed_edges = 0
    for atoms in images:
        i = neighbor_list("i", atoms, cutoff, self_interaction=False)
        atoms_total += len(atoms)
        directed_edges += len(i)
    if atoms_total == 0:
        raise ValueError("empty training split")
    return directed_edges / atoms_total


def split_summary(images: list[Atoms], source: Path) -> dict:
    energies = []
    force_absmax = []
    natoms = []
    has_stress = []
    for atoms in images:
        energy, forces, stress = result_arrays(atoms)
        energies.append(energy / len(atoms))
        force_absmax.append(float(np.abs(forces).max()))
        natoms.append(len(atoms))
        has_stress.append(stress is not None)
    return {
        "frames": len(images),
        "natoms_min": int(min(natoms)),
        "natoms_max": int(max(natoms)),
        "energy_ev_per_atom_min": float(min(energies)),
        "energy_ev_per_atom_max": float(max(energies)),
        "force_absmax_ev_per_angstrom": float(max(force_absmax)),
        "stress_available_for_all_frames": bool(all(has_stress)),
        "source_file": str(source.resolve()),
        "source_sha256": sha256_file(source),
    }


def prepare_split(
    split: str,
    extxyz: Path,
    destination: Path,
    keys: np.ndarray,
    e0: np.ndarray,
    cutoff: float,
    workers: int,
) -> None:
    blocks, _, raw_energy, cells, pbcs, stresses = extract_data_blocks(str(extxyz))
    indices = np.arange(len(blocks), dtype=np.int64)
    corrections = compute_correction(blocks, raw_energy, keys, e0)
    save_set(
        split,
        indices,
        blocks,
        raw_energy,
        corrections,
        cells,
        pbc_list=pbcs,
        stress_list=stresses,
        output_dir=str(destination),
    )
    save_to_h5_parallel(split, cutoff, workers, data_dir=str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--set", dest="set_id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force-extxyz", action="store_true")
    parser.add_argument(
        "--transfer-only",
        action="store_true",
        help="prepare only MgO-4x4 transfer data from an existing MgO-2x2 preparation",
    )
    parser.add_argument(
        "--prepare-mgo-size-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepare the MgO-4x4 test split using train-only MgO-2x2 references",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    global_summary = {
        "benchmark": "MS25 solid-state external-model comparison",
        "protocol": {
            "official_split": f"Set{args.set_id}",
            "checkpoint_selection": "minimum full-validation Force MAE",
            "test_used_for_selection": False,
            "element_reference_fit": "training split only, NumPy minimum-norm least squares",
        },
        "datasets": {},
    }
    existing_manifest = args.output_root / f"ms25_set{args.set_id}_prepared_manifest.json"
    if args.transfer_only and existing_manifest.is_file():
        global_summary = json.loads(existing_manifest.read_text())

    if args.transfer_only:
        mgo2_metadata = args.output_root / "mgo-2x2" / f"set{args.set_id}" / "metadata.json"
        if not mgo2_metadata.is_file():
            raise FileNotFoundError(mgo2_metadata)
        global_summary["datasets"]["mgo-2x2"] = json.loads(mgo2_metadata.read_text())

    for slug in (() if args.transfer_only else args.datasets):
        spec = DATASETS[slug]
        source_dir = args.archive_root / "Datasets" / spec["archive"] / f"set{args.set_id}"
        output = args.output_root / slug / f"set{args.set_id}"
        extxyz_dir = output / "extxyz"
        chorus_dir = output / "chorus"
        chorus_dir.mkdir(parents=True, exist_ok=True)

        sources = {
            "train": source_dir / "train-1000.traj",
            "val": source_dir / "valid.traj",
            "test": source_dir / "test.traj",
        }
        for path in sources.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        images = {split: read(path, ":") for split, path in sources.items()}
        expected = {"train": 1000, "val": 80, "test": 8920}
        observed = {split: len(value) for split, value in images.items()}
        if observed != expected:
            raise RuntimeError(f"{slug}: official split counts differ: {observed} != {expected}")

        extxyz = {split: extxyz_dir / f"{split}.extxyz" for split in sources}
        for split in sources:
            if args.force_extxyz or not extxyz[split].exists():
                write_extxyz(images[split], extxyz[split])

        elements = np.unique(
            np.concatenate([atoms.numbers for atoms in images["train"]])
        ).astype(np.int64)
        train_blocks, _, train_raw_energy, _, _, _ = extract_data_blocks(str(extxyz["train"]))
        e0 = fit_baseline_energies(train_blocks, train_raw_energy, elements)
        counts = count_matrix(images["train"], elements)
        rank = int(np.linalg.matrix_rank(counts))
        singular_values = np.linalg.svd(counts, compute_uv=False)
        nonzero = singular_values[singular_values > np.finfo(float).eps * singular_values[0]]
        condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf

        for split in ("train", "val", "test"):
            prepare_split(
                split,
                extxyz[split],
                chorus_dir,
                elements,
                e0,
                float(spec["cutoff"]),
                args.num_workers,
            )

        split_hashes = {
            split: {structure_digest(atoms) for atoms in split_images}
            for split, split_images in images.items()
        }
        overlaps = {
            "train_val": len(split_hashes["train"] & split_hashes["val"]),
            "train_test": len(split_hashes["train"] & split_hashes["test"]),
            "val_test": len(split_hashes["val"] & split_hashes["test"]),
        }
        summary = {
            "dataset": slug,
            "official_name": spec["archive"],
            "label_source": spec["label_source"],
            "cutoff_angstrom": float(spec["cutoff"]),
            "average_train_neighbours": average_neighbours(
                images["train"], float(spec["cutoff"])
            ),
            "atomic_numbers": elements.tolist(),
            "element_reference_energies_ev": e0.tolist(),
            "composition_matrix_rank": rank,
            "composition_columns": len(elements),
            "composition_condition_number_nonzero_subspace": condition,
            "individual_element_references_identifiable": rank == len(elements),
            "exact_structure_overlap": overlaps,
            "splits": {
                split: split_summary(images[split], sources[split])
                for split in sources
            },
            "extxyz_sha256": {
                split: sha256_file(path) for split, path in extxyz.items()
            },
        }
        (output / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
        global_summary["datasets"][slug] = summary
        print(
            f"prepared {slug}: cutoff={spec['cutoff']} A, "
            f"avg_neighbors={summary['average_train_neighbours']:.6f}, "
            f"E0={dict(zip(elements.tolist(), e0.tolist()))}"
        )

    if args.prepare_mgo_size_transfer and "mgo-2x2" in global_summary["datasets"]:
        source = (
            args.archive_root
            / "Datasets"
            / "MgO-4x4"
            / f"set{args.set_id}"
            / "test.traj"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        transfer_images = read(source, ":")
        if len(transfer_images) != 8920:
            raise RuntimeError(
                f"MgO-4x4 official test count differs: {len(transfer_images)} != 8920"
            )
        output = args.output_root / "mgo-4x4-transfer" / f"set{args.set_id}"
        extxyz = output / "extxyz" / "test.extxyz"
        chorus_dir = output / "chorus"
        chorus_dir.mkdir(parents=True, exist_ok=True)
        if args.force_extxyz or not extxyz.exists():
            write_extxyz(transfer_images, extxyz)
        reference = global_summary["datasets"]["mgo-2x2"]
        keys = np.asarray(reference["atomic_numbers"], dtype=np.int64)
        e0 = np.asarray(reference["element_reference_energies_ev"], dtype=np.float64)
        cutoff = float(reference["cutoff_angstrom"])
        prepare_split(
            "test",
            extxyz,
            chorus_dir,
            keys,
            e0,
            cutoff,
            args.num_workers,
        )
        # CHORUS eval-only reconstruction still opens the declared training
        # prefix before loading checkpoint state.  Reuse the MgO-2x2 training
        # artifacts by symlink so no MgO-4x4 label can influence fitting or
        # checkpoint selection.
        mgo2_train_dir = args.output_root / "mgo-2x2" / f"set{args.set_id}" / "chorus"
        for source_train_file in mgo2_train_dir.glob("*train*"):
            target = chorus_dir / source_train_file.name
            if not target.exists():
                target.symlink_to(source_train_file.resolve())
        transfer_summary = {
            "dataset": "mgo-4x4-transfer",
            "official_name": "MgO-4x4",
            "label_source": "DFT-PBE-D3",
            "role": "size-transfer test only",
            "trained_and_selected_on": "MgO-2x2 Set1 train/validation",
            "cutoff_angstrom": cutoff,
            "atomic_numbers": keys.tolist(),
            "element_reference_energies_ev_from_mgo_2x2_train": e0.tolist(),
            "split": split_summary(transfer_images, source),
            "average_test_neighbours": average_neighbours(transfer_images, cutoff),
            "extxyz_sha256": sha256_file(extxyz),
        }
        (output / "metadata.json").write_text(
            json.dumps(transfer_summary, indent=2) + "\n"
        )
        global_summary["datasets"]["mgo-4x4-transfer"] = transfer_summary
        print(
            "prepared mgo-4x4-transfer: test-only, "
            f"avg_neighbors={transfer_summary['average_test_neighbours']:.6f}"
        )

    summary_path = args.output_root / f"ms25_set{args.set_id}_prepared_manifest.json"
    summary_path.write_text(json.dumps(global_summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
