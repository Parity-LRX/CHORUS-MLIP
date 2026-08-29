#!/usr/bin/env python3
"""Audit checkpoint output parity after switching the ICTC product/conv-TP backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase.neighborlist import neighbor_list


def strict_fp32() -> None:
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def evaluate(checkpoint: str, graph, *, product_backend=None, conv_tp_backend=None):
    from chorus.bench.synthetic_workloads import ictc_force_fn
    from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

    loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint,
        element_types=["H", "C", "N", "O"],
        device="cuda",
        ictd_fix_product_backend=product_backend,
        ictd_tp_backend=conv_tp_backend,
    )
    model = loaded.wrapper.model.eval()
    energy, forces = ictc_force_fn(model, graph, training=False)()
    torch.cuda.synchronize()
    result = (energy.detach().cpu(), forces.detach().cpu())
    del energy, forces, model, loaded
    torch.cuda.empty_cache()
    return result


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = candidate - reference
    scale = reference.abs().clamp_min(1.0e-12)
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
    return {
        "max_abs": float(delta.abs().max()),
        "rms_abs": float(delta.square().mean().sqrt()),
        "rms_relative": float(delta.square().mean().sqrt() / reference_rms),
        "max_rel": float((delta.abs() / scale).max()),
    }


def make_lattice_graph(*, natoms: int, degree: int, cutoff: float, seed: int):
    """Build the same fixed-degree geometry used by the A100 scaling protocol."""
    if degree != 32:
        raise ValueError("The lattice parity audit currently supports degree=32 only")
    dims = [1, 1, 1]
    remaining = int(natoms)
    factor = 2
    while remaining > 1:
        while remaining % factor:
            factor += 1
        dims[dims.index(min(dims))] *= factor
        remaining //= factor
    nx, ny, nz = dims
    spacing = float(cutoff) / 2.05
    rng = np.random.default_rng(int(seed) + int(natoms))
    grid = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    positions = spacing * (grid.astype(np.float64) + 0.001 * rng.normal(size=(natoms, 3)))
    numbers = np.asarray([1, 6, 7, 8], dtype=np.int64)[np.arange(natoms) % 4]
    cell = np.diag(np.asarray([nx, ny, nz], dtype=float) * spacing)

    from ase import Atoms

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    edge_src, edge_dst, unit_shifts = neighbor_list("ijS", atoms, cutoff)
    expected_edges = int(natoms) * int(degree)
    if len(edge_src) != expected_edges:
        raise RuntimeError(f"expected {expected_edges} directed edges, got {len(edge_src)}")
    device = torch.device("cuda")
    return SimpleNamespace(
        pos=torch.as_tensor(positions, dtype=torch.float32, device=device),
        atomic_numbers=torch.as_tensor(numbers, dtype=torch.long, device=device),
        batch=torch.zeros(natoms, dtype=torch.long, device=device),
        edge_src=torch.as_tensor(edge_src, dtype=torch.long, device=device),
        edge_dst=torch.as_tensor(edge_dst, dtype=torch.long, device=device),
        unit_shifts=torch.as_tensor(unit_shifts, dtype=torch.float32, device=device),
        cell=torch.as_tensor(cell, dtype=torch.float32, device=device).reshape(1, 3, 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--natoms", type=int, default=128)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    strict_fp32()
    graph = make_lattice_graph(
        natoms=args.natoms,
        degree=args.degree,
        cutoff=args.cutoff,
        seed=args.seed,
    )
    reference_energy, reference_forces = evaluate(args.checkpoint, graph)
    candidates = {}
    for name, product_backend, conv_tp_backend in (
        ("conv_tp_only", None, "cueq"),
        ("product_only", "cueq", None),
        ("dual_cueq", "cueq", "cueq"),
    ):
        energy, forces = evaluate(
            args.checkpoint,
            graph,
            product_backend=product_backend,
            conv_tp_backend=conv_tp_backend,
        )
        candidates[name] = {
            "backend": {
                "product": product_backend or "checkpoint",
                "conv_tp": conv_tp_backend or "checkpoint",
            },
            "energy": error_stats(reference_energy, energy),
            "forces": error_stats(reference_forces, forces),
        }
    payload = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "natoms": args.natoms,
        "degree": args.degree,
        "cutoff": args.cutoff,
        "dtype": "float32",
        "tf32": False,
        "reference_backend": {"product": "checkpoint", "conv_tp": "checkpoint"},
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
