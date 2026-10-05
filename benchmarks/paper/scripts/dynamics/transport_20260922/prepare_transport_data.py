#!/usr/bin/env python3
"""Prepare the fixed public water and LiPS transport benchmark splits.

Water keeps the pre-existing public 1000/300/293 split verbatim.  LiPS is
sampled once from the 25001-frame public trajectory with disjoint
1000/1000/5000 train/validation/test indices.  Element reference energies are
fit only on the training split.  No validation or test label enters fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import iread, read, write
from ase.neighborlist import neighbor_list

from chorus.data.preprocessing import (
    compute_correction,
    extract_data_blocks,
    fit_baseline_energies,
    save_set,
    save_to_h5_parallel,
)


EXPECTED = {
    "water": {
        "frames": {"train": 1000, "val": 300, "test": 293},
        "natoms": 192,
        "composition": {1: 128, 8: 64},
    },
    "lips": {
        "frames": {"train": 1000, "val": 1000, "test": 5000},
        "natoms": 83,
        "composition": {3: 27, 15: 12, 16: 44},
        "source_frames": 25001,
    },
}


def _result_arrays(atoms: Atoms) -> tuple[float, np.ndarray]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("non-finite energy or force")
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"unexpected force shape {forces.shape}")
    return energy, forces


def _composition(atoms: Atoms) -> dict[int, int]:
    values, counts = np.unique(atoms.numbers, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _validate_images(dataset: str, split: str, images: list[Atoms]) -> None:
    spec = EXPECTED[dataset]
    expected_frames = spec["frames"][split]
    if len(images) != expected_frames:
        raise RuntimeError(
            f"{dataset}/{split}: {len(images)} frames, expected {expected_frames}"
        )
    reference_numbers: tuple[int, ...] | None = None
    for index, atoms in enumerate(images):
        if len(atoms) != spec["natoms"]:
            raise RuntimeError(
                f"{dataset}/{split}/{index}: {len(atoms)} atoms, expected {spec['natoms']}"
            )
        if _composition(atoms) != spec["composition"]:
            raise RuntimeError(
                f"{dataset}/{split}/{index}: composition {_composition(atoms)}"
            )
        if not bool(np.all(atoms.pbc)):
            raise RuntimeError(f"{dataset}/{split}/{index}: not fully periodic")
        if abs(float(np.linalg.det(atoms.cell.array))) < 1.0e-8:
            raise RuntimeError(f"{dataset}/{split}/{index}: singular cell")
        _result_arrays(atoms)
        numbers = tuple(map(int, atoms.numbers))
        if reference_numbers is None:
            reference_numbers = numbers
        elif numbers != reference_numbers:
            raise RuntimeError(f"{dataset}/{split}/{index}: atom ordering changed")


def _write_extxyz(images: list[Atoms], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    write(temporary, images, format="extxyz", write_results=True)
    temporary.replace(destination)


def _average_neighbours(images: list[Atoms], cutoff: float) -> float:
    total_atoms = 0
    total_edges = 0
    for atoms in images:
        centres = neighbor_list("i", atoms, cutoff, self_interaction=False)
        total_atoms += len(atoms)
        total_edges += len(centres)
    return total_edges / total_atoms


def _count_matrix(images: list[Atoms], elements: np.ndarray) -> np.ndarray:
    columns = {int(z): column for column, z in enumerate(elements)}
    matrix = np.zeros((len(images), len(elements)), dtype=np.float64)
    for row, atoms in enumerate(images):
        for number, count in _composition(atoms).items():
            matrix[row, columns[number]] = count
    return matrix


def _prepare_chorus_split(
    *,
    split: str,
    extxyz: Path,
    destination: Path,
    elements: np.ndarray,
    e0: np.ndarray,
    cutoff: float,
    workers: int,
) -> None:
    blocks, _, raw_energy, cells, pbcs, stresses = extract_data_blocks(str(extxyz))
    indices = np.arange(len(blocks), dtype=np.int64)
    correction = compute_correction(blocks, raw_energy, elements, e0)
    save_set(
        split,
        indices,
        blocks,
        raw_energy,
        correction,
        cells,
        pbc_list=pbcs,
        stress_list=stresses,
        output_dir=str(destination),
    )
    save_to_h5_parallel(split, cutoff, workers, data_dir=str(destination))


def _water_splits(source: Path, extxyz_dir: Path) -> dict[str, list[Atoms]]:
    result: dict[str, list[Atoms]] = {}
    for split in ("train", "val", "test"):
        original = source / f"{split}.extxyz"
        if not original.is_file():
            raise FileNotFoundError(original)
        destination = extxyz_dir / original.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        images = read(destination, ":")
        _validate_images("water", split, images)
        result[split] = images
    return result


def _lips_splits(source: Path, extxyz_dir: Path, seed: int) -> tuple[dict[str, list[Atoms]], dict[str, list[int]]]:
    spec = EXPECTED["lips"]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(spec["source_frames"], size=7000, replace=False)
    assignments: dict[int, str] = {}
    indices: dict[str, list[int]] = {}
    offset = 0
    for split in ("train", "val", "test"):
        count = spec["frames"][split]
        selected = sorted(map(int, sampled[offset : offset + count]))
        indices[split] = selected
        assignments.update({index: split for index in selected})
        offset += count

    result = {split: [] for split in ("train", "val", "test")}
    observed = 0
    for frame_index, atoms in enumerate(iread(source, index=":")):
        observed += 1
        split = assignments.get(frame_index)
        if split is not None:
            result[split].append(atoms)
    if observed != spec["source_frames"]:
        raise RuntimeError(
            f"LiPS source contains {observed} frames, expected {spec['source_frames']}"
        )
    for split, images in result.items():
        _validate_images("lips", split, images)
        _write_extxyz(images, extxyz_dir / f"{split}.extxyz")
    return result, indices


def _prepare_dataset(
    *,
    dataset: str,
    images: dict[str, list[Atoms]],
    extxyz_dir: Path,
    output: Path,
    cutoff: float,
    workers: int,
    source_description: str,
    selected_indices: dict[str, list[int]] | None,
) -> dict[str, object]:
    chorus_dir = output / "chorus"
    chorus_dir.mkdir(parents=True, exist_ok=True)
    elements = np.unique(
        np.concatenate([atoms.numbers for atoms in images["train"]])
    ).astype(np.int64)
    train_blocks, _, raw_energy, _, _, _ = extract_data_blocks(
        str(extxyz_dir / "train.extxyz")
    )
    e0 = fit_baseline_energies(train_blocks, raw_energy, elements)
    counts = _count_matrix(images["train"], elements)
    rank = int(np.linalg.matrix_rank(counts))
    for split in ("train", "val", "test"):
        _prepare_chorus_split(
            split=split,
            extxyz=extxyz_dir / f"{split}.extxyz",
            destination=chorus_dir,
            elements=elements,
            e0=e0,
            cutoff=cutoff,
            workers=workers,
        )

    metadata: dict[str, object] = {
        "dataset": dataset,
        "source": source_description,
        "cutoff_angstrom": cutoff,
        "average_train_neighbours": _average_neighbours(images["train"], cutoff),
        "atomic_numbers": elements.tolist(),
        "element_reference_energies_ev": e0.tolist(),
        "element_reference_fit": "training split only; minimum-norm least squares",
        "composition_matrix_rank": rank,
        "composition_columns": len(elements),
        "individual_element_references_identifiable": rank == len(elements),
        "fixed_composition_note": (
            "Only the total composition offset is identifiable; individual element "
            "values are the minimum-norm convention."
        ),
        "splits": {
            split: {
                "frames": len(split_images),
                "atoms_per_frame": len(split_images[0]),
                "composition": {
                    str(number): count
                    for number, count in _composition(split_images[0]).items()
                },
                "energy_ev_per_atom_min": min(
                    atoms.get_potential_energy() / len(atoms)
                    for atoms in split_images
                ),
                "energy_ev_per_atom_max": max(
                    atoms.get_potential_energy() / len(atoms)
                    for atoms in split_images
                ),
                "force_absmax_ev_per_angstrom": max(
                    float(np.abs(atoms.get_forces()).max()) for atoms in split_images
                ),
            }
            for split, split_images in images.items()
        },
        "checkpoint_selection": "minimum full-validation Force MAE",
        "test_used_for_selection": False,
    }
    if selected_indices is not None:
        metadata["source_frame_indices"] = selected_indices
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-source", type=Path, required=True)
    parser.add_argument("--lips-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    if (args.output_root / "DATA_READY").is_file():
        print((args.output_root / "manifest.json").read_text())
        return
    args.output_root.mkdir(parents=True, exist_ok=True)

    water_output = args.output_root / "water"
    water_extxyz = water_output / "extxyz"
    water = _water_splits(args.water_source, water_extxyz)
    water_metadata = _prepare_dataset(
        dataset="water",
        images=water,
        extxyz_dir=water_extxyz,
        output=water_output,
        cutoff=args.cutoff,
        workers=args.workers,
        source_description=(
            "Bingqing Cheng public revPBE0-D3 liquid-water data; existing "
            "1000/300/293 split retained verbatim"
        ),
        selected_indices=None,
    )

    lips_output = args.output_root / "lips"
    lips_extxyz = lips_output / "extxyz"
    lips, indices = _lips_splits(args.lips_source, lips_extxyz, args.seed)
    lips_metadata = _prepare_dataset(
        dataset="lips",
        images=lips,
        extxyz_dir=lips_extxyz,
        output=lips_output,
        cutoff=args.cutoff,
        workers=args.workers,
        source_description=(
            "Materials Cloud record 2022.45 Li6.75P3S11; deterministic disjoint "
            "1000/1000/5000 sample"
        ),
        selected_indices=indices,
    )

    manifest = {
        "benchmark": "CHORUS liquid-water and LiPS transport evidence",
        "seed": args.seed,
        "strict_common_cutoff_angstrom": args.cutoff,
        "water_split": "public 1000/300/293 split retained verbatim",
        "lips_split": "deterministic disjoint 1000/1000/5000 sample",
        "water": water_metadata,
        "lips": lips_metadata,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output_root / "DATA_READY").touch()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
