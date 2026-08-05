#!/usr/bin/env python3
"""Matched atom-count scaling for CHORUS and TECE on one GPU.

The benchmark uses the same deterministic periodic carbon structures for both
models. Neighbour graphs are built once and excluded from the timed region, so
the reported latency isolates energy plus conservative-force model execution.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms


def make_periodic_carbon(side: int, spacing: float, seed: int) -> Atoms:
    grid = np.stack(
        np.meshgrid(
            np.arange(side),
            np.arange(side),
            np.arange(side),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    positions = (grid.astype(np.float64) + 0.5) * spacing
    rng = np.random.default_rng(seed + side)
    positions += rng.normal(scale=0.015 * spacing, size=positions.shape)
    return Atoms(
        numbers=np.full(side**3, 6, dtype=np.int64),
        positions=positions,
        cell=np.eye(3) * side * spacing,
        pbc=True,
    )


def timed_cuda_calls(call, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(1000.0 * (time.perf_counter() - start))
    return samples


def repeat_count(natoms: int) -> int:
    if natoms <= 512:
        return 15
    if natoms <= 1728:
        return 8
    return 5


def summarize_samples(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": float(statistics.median(ordered)),
        "mean_ms": float(statistics.mean(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def enforce_strict_fp32() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_chorus(args):
    from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
    from chorus.training.makefx_compile import trace_and_compile_force
    from chorus.utils.graph_utils import radius_graph_pbc_gpu

    device = torch.device("cuda")
    obj = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.checkpoint,
        element_types=["H", "C", "N", "O"],
        device="cuda",
    )
    model = obj.wrapper.model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dtype = next(model.parameters()).dtype

    def make_inputs(atoms: Atoms):
        pos = torch.as_tensor(atoms.positions, dtype=dtype, device=device)
        atomic_numbers = torch.as_tensor(
            atoms.numbers, dtype=dtype, device=device
        )
        cell = torch.as_tensor(
            atoms.cell.array, dtype=dtype, device=device
        ).unsqueeze(0)
        edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(
            pos, float(obj.rcutfac), cell, pbc=(True, True, True)
        )
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
        return (
            pos,
            atomic_numbers,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
        )

    def prepare(atoms: Atoms):
        inputs = make_inputs(atoms)
        # Some model-internal views retain the traced node count even with
        # dynamic Inductor enabled. This mirrors CHORUS's deployed bucket cache:
        # compile one force graph for each atom-count bucket, then time reuse.
        compiled = trace_and_compile_force(
            model,
            inputs,
            training=False,
            compile_dynamic_shapes=True,
        )

        def call():
            return compiled(*inputs)

        return call, int(inputs[3].numel())

    metadata = {
        "model": "CHORUS",
        "mode": "makefx_dynamic",
        "parameters": sum(p.numel() for p in model.parameters()),
        "cutoff_angstrom": float(obj.rcutfac),
        "compile_scope": "one MakeFX graph per atom-count bucket",
    }
    return prepare, metadata


def build_tece(args):
    from torch_geometric.loader import DataLoader

    from tace.dataset.graph import from_atoms
    from tace.dataset.quantity import KEYS, KeySpecification, update_keyspec_from_kwargs
    from tace.lightning import load_tace

    device = torch.device("cuda")
    model = load_tace(
        args.checkpoint,
        "cuda",
        # A checkpoint trained with the eager first-layer CGTP does not contain
        # cuEquivariance's regenerated, non-trainable polynomial constants.
        # All learned tensors still load by name and shape.
        strict=args.engine != "tece-cue",
        use_ema=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device)
    keyspec = KeySpecification()
    update_keyspec_from_kwargs(keyspec, KEYS)
    element = model.get_torch_element()
    cutoff = float(model.get_cutoff())
    max_neighbors = model.get_max_neighbors()
    target_property = model.get_target_property()
    embedding_property = model.get_embedding_property()
    fidelity_idx = model.get_fidelity_idx()

    def prepare(atoms: Atoms):
        atoms.info["fidelity_idx"] = fidelity_idx
        graph = from_atoms(
            element,
            atoms,
            cutoff,
            max_neighbors=max_neighbors,
            target_property=target_property,
            embedding_property=embedding_property,
            keyspec=keyspec,
            training=False,
            neighborlist_backend="matscipy",
        )
        batch = next(
            iter(DataLoader([graph], batch_size=1, shuffle=False))
        ).to(device)

        def call():
            return model(batch)

        edge_index = getattr(batch, "edge_index")
        return call, int(edge_index.shape[1])

    metadata = {
        "model": "TECE",
        "mode": args.engine.removeprefix("tece-"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "cutoff_angstrom": cutoff,
    }
    return prepare, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=["chorus", "tece-eager", "tece-cue"],
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sides",
        type=lambda value: [int(x) for x in value.split(",")],
        default=[4, 5, 6, 7, 8, 9, 10, 12, 14, 16],
    )
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    # This benchmark requires strict IEEE fp32 for both models. Override any
    # policy restored from a training checkpoint or inherited from the process.
    enforce_strict_fp32()

    if args.engine == "tece-cue":
        os.environ["TACE_USE_CUE"] = "1"
    elif args.engine == "tece-eager":
        os.environ["TACE_USE_CUE"] = "0"
    os.environ.setdefault("TACE_USE_OEQ", "0")
    os.environ.setdefault("TACE_USE_EQT", "0")
    os.environ.setdefault("TACE_USE_COMPILE", "0")

    if args.engine == "chorus":
        prepare, metadata = build_chorus(args)
    else:
        prepare, metadata = build_tece(args)
    # Model loaders may restore the training configuration globally.
    enforce_strict_fp32()

    rows = []
    for side in args.sides:
        atoms = make_periodic_carbon(side, args.spacing, args.seed)
        natoms = len(atoms)
        try:
            torch.cuda.empty_cache()
            prepare_start = time.perf_counter()
            call, nedges = prepare(atoms)
            prepare_seconds = time.perf_counter() - prepare_start
            torch.cuda.reset_peak_memory_stats()
            warmup_start = time.perf_counter()
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - warmup_start
            repeats = repeat_count(natoms)
            samples = timed_cuda_calls(call, repeats)
            timing = summarize_samples(samples)
            median_ms = timing["median_ms"]
            row = {
                "side": side,
                "natoms": natoms,
                "nedges": nedges,
                "neighbors_per_atom": nedges / natoms,
                "repeats": repeats,
                "prepare_seconds": prepare_seconds,
                "warmup_seconds": warmup_seconds,
                **timing,
                "atoms_per_second": 1000.0 * natoms / median_ms,
                "edges_per_second": 1000.0 * nedges / median_ms,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()
                / (1024**3),
                "status": "ok",
            }
        except torch.cuda.OutOfMemoryError as exc:
            row = {
                "side": side,
                "natoms": natoms,
                "status": "oom",
                "error": str(exc),
            }
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "protocol": {
            "hardware": torch.cuda.get_device_name(),
            "dtype": "float32",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "structure": "periodic jittered simple-cubic carbon",
            "spacing_angstrom": args.spacing,
            "graph_build_in_timing": False,
            "quantity": "energy plus conservative forces",
            "warmup_calls_per_size": 3,
        },
        **metadata,
        "checkpoint": args.checkpoint,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
