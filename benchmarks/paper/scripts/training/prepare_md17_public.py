#!/usr/bin/env python3
"""Prepare public MD17/rMD17 subsets for MACE and MACE-ICTC training.

Outputs, per molecule:
  - train.extxyz / val.extxyz / test.extxyz for mace-torch
  - processed_train.h5 / processed_val.h5 / processed_test.h5 for MACE-ICTC
  - split_indices.npz and metadata.json for auditability

PyG MD17 labels are kcal/mol and kcal/mol/Angstrom. This script converts them
to eV and eV/Angstrom before writing both formats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from matscipy.neighbours import neighbour_list
from torch_geometric.datasets import MD17


KCAL_MOL_TO_EV = 0.0433641153087705
DEFAULT_ELEMENTS = (1, 6, 7, 8)


def _clean_name(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _write_extxyz(path: Path, frames: list[dict]) -> None:
    with path.open("w") as f:
        for frame in frames:
            z = frame["z"].astype(int)
            pos = frame["pos"]
            forces = frame["forces"]
            energy = float(frame["energy"])
            f.write(f"{len(z)}\n")
            f.write(
                'Properties=species:S:1:pos:R:3:forces:R:3 '
                f'energy={energy:.16e} '
                'pbc="F F F" Lattice="100 0 0 0 100 0 0 0 100"\n'
            )
            for zi, xyz, force in zip(z, pos, forces):
                f.write(
                    f"{_symbol(int(zi))} "
                    f"{xyz[0]:.16e} {xyz[1]:.16e} {xyz[2]:.16e} "
                    f"{force[0]:.16e} {force[1]:.16e} {force[2]:.16e}\n"
                )


def _symbol(z: int) -> str:
    if z == 1:
        return "H"
    if z == 6:
        return "C"
    if z == 7:
        return "N"
    if z == 8:
        return "O"
    raise ValueError(f"unsupported atomic number in MD17 subset: {z}")


def _sanitize_shifts(
    pos: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    shifts: np.ndarray,
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
    max_radius: float,
) -> np.ndarray:
    shifts = np.asarray(shifts, dtype=np.float64)
    if not any(bool(x) for x in pbc):
        if shifts.size and np.any(shifts != 0.0):
            shifts = np.zeros_like(shifts)
        edge_vec = pos[edge_dst] - pos[edge_src]
    else:
        edge_vec = pos[edge_dst] - pos[edge_src] + shifts @ cell
    if edge_vec.size:
        max_len = float(np.linalg.norm(edge_vec, axis=1).max())
        if max_len > float(max_radius) + 1e-6:
            raise ValueError(
                f"neighbor-list edge length {max_len:.6g} exceeds cutoff {float(max_radius):.6g}; "
                "check pbc/cell handling before writing processed H5"
            )
    return shifts


def _write_processed_h5(path: Path, frames: list[dict], *, max_radius: float) -> None:
    node_counts = np.zeros(len(frames), dtype=np.int64)
    edge_counts = np.zeros(len(frames), dtype=np.int64)
    max_atoms = 0
    max_edges = 0
    cell = np.eye(3, dtype=np.float64) * 100.0
    pbc = (False, False, False)
    with h5py.File(path, "w") as h5:
        for i, frame in enumerate(frames):
            pos = frame["pos"].astype(np.float64)
            z = frame["z"].astype(np.int64)
            forces = frame["forces"].astype(np.float64)
            edge_src, edge_dst, shifts = neighbour_list("ijS", positions=pos, cell=cell, pbc=pbc, cutoff=max_radius)
            edge_src = np.asarray(edge_src, dtype=np.int64)
            edge_dst = np.asarray(edge_dst, dtype=np.int64)
            shifts = _sanitize_shifts(pos, edge_src, edge_dst, shifts, cell, pbc, max_radius)
            g = h5.create_group(f"sample_{i}")
            g.create_dataset("pos", data=pos)
            g.create_dataset("A", data=z)
            g.create_dataset("y", data=np.float64(frame["energy"]))
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


def _load_frames(root: Path, name: str, *, revised_url: str | None = None) -> list[dict]:
    if revised_url and str(name).startswith("revised "):
        MD17.revised_url = revised_url
    ds = MD17(str(root), name=name)
    frames: list[dict] = []
    for data in ds:
        z = _to_numpy(data.z).astype(np.int64)
        pos = _to_numpy(data.pos).astype(np.float64)
        energy = float(_to_numpy(data.energy).reshape(-1)[0]) * KCAL_MOL_TO_EV
        forces = _to_numpy(data.force).astype(np.float64) * KCAL_MOL_TO_EV
        frames.append({"z": z, "pos": pos, "energy": energy, "forces": forces})
    return frames


def _split_indices(n: int, *, train: int, val: int, test: int, seed: int) -> dict[str, np.ndarray]:
    need = int(train) + int(val) + int(test)
    if need > n:
        raise ValueError(f"requested {need} frames but dataset has {n}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    return {
        "train": np.sort(perm[:train]),
        "val": np.sort(perm[train:train + val]),
        "test": np.sort(perm[train + val:train + val + test]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="PyG MD17 download/cache root")
    p.add_argument("--out-root", required=True)
    p.add_argument("--molecules", default="revised ethanol,revised benzene,revised aspirin")
    p.add_argument(
        "--md17-revised-url",
        default="https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1",
        help="Override PyG's historical rMD17 URL; older PyG releases point to a 404 Materials Cloud record.",
    )
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--max-radius", type=float, default=4.5)
    args = p.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.molecules.split(",") if x.strip()]
    manifest = {
        "source": "torch_geometric.datasets.MD17",
        "md17_revised_url": args.md17_revised_url,
        "unit_conversion": {"energy": "kcal/mol to eV", "force": "kcal/mol/A to eV/A", "factor": KCAL_MOL_TO_EV},
        "seed": int(args.seed),
        "max_radius": float(args.max_radius),
        "molecules": [],
    }
    for name in names:
        frames = _load_frames(root, name, revised_url=args.md17_revised_url)
        splits = _split_indices(
            len(frames),
            train=args.train_size,
            val=args.val_size,
            test=args.test_size,
            seed=args.seed,
        )
        out_dir = out_root / _clean_name(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "split_indices.npz", **splits)
        mol_meta = {
            "name": name,
            "directory": str(out_dir),
            "num_total_frames": len(frames),
            "splits": {k: int(v.shape[0]) for k, v in splits.items()},
            "atomic_numbers": sorted({int(z) for idx in splits["train"] for z in frames[int(idx)]["z"]}),
        }
        for split, indices in splits.items():
            split_frames = [frames[int(i)] for i in indices]
            _write_extxyz(out_dir / f"{split}.extxyz", split_frames)
            _write_processed_h5(out_dir / f"processed_{split}.h5", split_frames, max_radius=args.max_radius)
            mol_meta[f"{split}_path_extxyz"] = str(out_dir / f"{split}.extxyz")
            mol_meta[f"{split}_path_h5"] = str(out_dir / f"processed_{split}.h5")
        (out_dir / "metadata.json").write_text(json.dumps(mol_meta, indent=2, sort_keys=True))
        manifest["molecules"].append(mol_meta)
        print(json.dumps(mol_meta, sort_keys=True), flush=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
