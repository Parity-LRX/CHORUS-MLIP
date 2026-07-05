#!/usr/bin/env python3
"""Prepare the public Cheng liquid-water XYZ set for MACE/MACE-ICTC tests.

Input:
  Bingqing Cheng, ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz

The source file is an extended XYZ trajectory with periodic cells, a frame-level
``TotEnergy`` field, and per-atom ``force`` arrays. The raw coordinates are in
Bohr: the 23.465110 cell gives a 64-water density of approximately 1 g/cm^3
only after Bohr-to-Angstrom conversion. By default this script therefore treats
the source as atomic units and writes Angstrom, eV, and eV/Angstrom outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from ase.io import iread, write
from matscipy.neighbours import neighbour_list


BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_ANG = HARTREE_TO_EV / BOHR_TO_ANGSTROM


def _as_forces(atoms, *, force_scale: float) -> np.ndarray:
    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=np.float64)
    elif "force" in atoms.arrays:
        forces = np.asarray(atoms.arrays["force"], dtype=np.float64)
    else:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    return forces * float(force_scale)


def _energy(atoms, *, energy_scale: float) -> float:
    for key in ("energy", "Energy", "TotEnergy", "total_energy"):
        if key in atoms.info:
            return float(atoms.info[key]) * float(energy_scale)
    return float(atoms.get_potential_energy()) * float(energy_scale)


def _copy_frame(atoms, *, length_scale: float, energy_scale: float, force_scale: float):
    out = atoms.copy()
    out.positions = np.asarray(out.positions, dtype=np.float64) * float(length_scale)
    out.cell = np.asarray(out.cell.array, dtype=np.float64) * float(length_scale)
    out.info.clear()
    out.info["energy"] = _energy(atoms, energy_scale=energy_scale)
    out.arrays["forces"] = _as_forces(atoms, force_scale=force_scale)
    if "force" in out.arrays:
        del out.arrays["force"]
    return out


def _load_frames(path: Path, *, length_scale: float, energy_scale: float, force_scale: float):
    return [
        _copy_frame(atoms, length_scale=length_scale, energy_scale=energy_scale, force_scale=force_scale)
        for atoms in iread(str(path), index=":")
    ]


def _split_indices(n: int, *, train: int, val: int, test: int, seed: int) -> dict[str, np.ndarray]:
    if test < 0:
        test = n - train - val
    need = train + val + test
    if need > n:
        raise ValueError(f"requested {need} frames but source has {n}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    return {
        "train": np.sort(perm[:train]),
        "val": np.sort(perm[train : train + val]),
        "test": np.sort(perm[train + val : train + val + test]),
    }


def _write_processed_h5(path: Path, frames, *, max_radius: float) -> dict[str, int]:
    node_counts = np.zeros(len(frames), dtype=np.int64)
    edge_counts = np.zeros(len(frames), dtype=np.int64)
    max_atoms = 0
    max_edges = 0
    with h5py.File(path, "w") as h5:
        for i, atoms in enumerate(frames):
            pos = np.asarray(atoms.positions, dtype=np.float64)
            z = np.asarray(atoms.numbers, dtype=np.int64)
            forces = _as_forces(atoms, force_scale=1.0)
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
            pbc = tuple(bool(x) for x in atoms.pbc)
            edge_src, edge_dst, shifts = neighbour_list("ijS", atoms, cutoff=float(max_radius))
            edge_src = np.asarray(edge_src, dtype=np.int64)
            edge_dst = np.asarray(edge_dst, dtype=np.int64)
            shifts = np.asarray(shifts, dtype=np.float64)
            if edge_src.size:
                edge_vec = pos[edge_dst] - pos[edge_src] + shifts @ cell
                max_len = float(np.linalg.norm(edge_vec, axis=1).max())
                if max_len > float(max_radius) + 1e-6:
                    raise ValueError(f"edge length {max_len:.6g} exceeds cutoff {max_radius:.6g}")
            g = h5.create_group(f"sample_{i}")
            g.create_dataset("pos", data=pos)
            g.create_dataset("A", data=z)
            g.create_dataset("y", data=np.float64(_energy(atoms, energy_scale=1.0)))
            g.create_dataset("force", data=forces)
            g.create_dataset("edge_src", data=edge_src)
            g.create_dataset("edge_dst", data=edge_dst)
            g.create_dataset("edge_shifts", data=shifts)
            g.create_dataset("cell", data=cell)
            g.create_dataset("stress", data=np.zeros((3, 3), dtype=np.float64))
            node_counts[i] = int(pos.shape[0])
            edge_counts[i] = int(edge_src.shape[0])
            max_atoms = max(max_atoms, int(pos.shape[0]))
            max_edges = max(max_edges, int(edge_src.shape[0]))
        h5.attrs["max_atoms"] = int(max_atoms)
        h5.attrs["max_edges"] = int(max_edges)
    np.savez(str(path) + ".counts.npz", node_counts=node_counts, edge_counts=edge_counts)
    return {"max_atoms": int(max_atoms), "max_edges": int(max_edges)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--test-size", type=int, default=-1, help="negative means use remaining frames")
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--max-radius", type=float, default=4.5)
    p.add_argument(
        "--unit-system",
        choices=["atomic", "ev_angstrom"],
        default="atomic",
        help="atomic: Bohr/Hartree/Hartree-per-Bohr source; ev_angstrom: no conversion.",
    )
    args = p.parse_args()

    if args.unit_system == "atomic":
        length_scale = BOHR_TO_ANGSTROM
        energy_scale = HARTREE_TO_EV
        force_scale = HARTREE_PER_BOHR_TO_EV_PER_ANG
        units_out = {"length": "Angstrom", "energy": "eV", "forces": "eV/Angstrom"}
        units_in = {"length": "Bohr", "energy": "Hartree", "forces": "Hartree/Bohr"}
    else:
        length_scale = 1.0
        energy_scale = 1.0
        force_scale = 1.0
        units_out = {"length": "Angstrom", "energy": "eV", "forces": "eV/Angstrom"}
        units_in = dict(units_out)

    frames = _load_frames(args.input, length_scale=length_scale, energy_scale=energy_scale, force_scale=force_scale)
    splits = _split_indices(
        len(frames),
        train=int(args.train_size),
        val=int(args.val_size),
        test=int(args.test_size),
        seed=int(args.seed),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "split_indices.npz", **splits)

    meta = {
        "source": "BingqingCheng/ab-initio-thermodynamics-of-water training-set/dataset_1593.xyz",
        "source_url": "https://github.com/BingqingCheng/ab-initio-thermodynamics-of-water/tree/master/training-set",
        "unit_system": args.unit_system,
        "units_assumed_input": units_in,
        "units_written": units_out,
        "unit_scales": {
            "length": length_scale,
            "energy": energy_scale,
            "forces": force_scale,
        },
        "n_total_frames": len(frames),
        "seed": int(args.seed),
        "max_radius": float(args.max_radius),
        "splits": {k: int(v.shape[0]) for k, v in splits.items()},
        "atomic_numbers": sorted({int(z) for atoms in frames for z in atoms.numbers}),
        "natoms": int(len(frames[0])) if frames else 0,
        "pbc": [bool(x) for x in frames[0].pbc] if frames else None,
        "cell": np.asarray(frames[0].cell.array).tolist() if frames else None,
    }

    for split, idx in splits.items():
        split_frames = [frames[int(i)] for i in idx]
        write(str(args.out_dir / f"{split}.extxyz"), split_frames, format="extxyz")
        h5_meta = _write_processed_h5(args.out_dir / f"processed_{split}.h5", split_frames, max_radius=args.max_radius)
        meta[f"{split}_path_extxyz"] = str(args.out_dir / f"{split}.extxyz")
        meta[f"{split}_path_h5"] = str(args.out_dir / f"processed_{split}.h5")
        meta[f"{split}_h5"] = h5_meta

    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
