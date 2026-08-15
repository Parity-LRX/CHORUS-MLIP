#!/usr/bin/env python3
"""Strict-FP32 NequIP energy/force scaling on fixed-degree periodic graphs.

The graph is built before timing.  Inference evaluates energy and conservative
forces.  Training includes the force double backward and one AdamW update.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from nequip.data import AtomicData, AtomicDataDict, dataset_from_config
from nequip.model import model_from_config
from nequip.scripts.train import default_config
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options


SPECIES = (0, 1, 2, 3)


def strict_fp32() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync() -> None:
    torch.cuda.synchronize()


def make_atoms(natoms: int, cutoff: float, seed: int) -> Atoms:
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
    positions = spacing * (
        grid.astype(np.float64) + 0.001 * rng.normal(size=(natoms, 3))
    )
    return Atoms(
        numbers=np.asarray((1, 6, 7, 8), dtype=np.int64)[
            np.arange(natoms) % len(SPECIES)
        ],
        positions=positions,
        cell=np.diag(np.asarray([nx, ny, nz], dtype=float) * spacing),
        pbc=True,
    )


def repeats_for(natoms: int, task: str) -> int:
    if task == "train":
        return 12 if natoms <= 128 else (8 if natoms <= 256 else 5)
    return 30 if natoms <= 128 else (20 if natoms <= 256 else 12)


def summarize(samples_ms: list[float], natoms: int) -> dict[str, float]:
    median_ms = float(statistics.median(samples_ms))
    ordered = sorted(samples_ms)
    return {
        "median_ms": median_ms,
        "mean_ms": float(statistics.mean(samples_ms)),
        "p10_ms": float(ordered[max(0, int(0.1 * (len(ordered) - 1)))]),
        "p90_ms": float(ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]),
        "steps_per_second": 1000.0 / median_ms,
        "atoms_per_second": 1000.0 * float(natoms) / median_ms,
    }


def synthetic_data(*, natoms: int, cutoff: float, seed: int, device: torch.device):
    atoms = make_atoms(natoms, cutoff, seed)
    atom_types = torch.as_tensor(
        np.arange(natoms, dtype=np.int64) % len(SPECIES), dtype=torch.long
    )
    atomic = AtomicData.from_points(
        pos=atoms.positions,
        r_max=cutoff,
        cell=atoms.cell.array,
        pbc=True,
        atom_types=atom_types,
    )
    data = AtomicData.to_AtomicDataDict(atomic.to(device))
    data[AtomicDataDict.BATCH_KEY] = torch.zeros(
        natoms, dtype=torch.long, device=device
    )
    data[AtomicDataDict.BATCH_PTR_KEY] = torch.tensor(
        [0, natoms], dtype=torch.long, device=device
    )
    nedges = int(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1])
    if nedges != 32 * natoms:
        raise RuntimeError(
            f"fixed-degree graph mismatch: expected {32 * natoms} edges, got {nedges}"
        )
    return data, nedges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="128,256,512,1024,2048,4096")
    parser.add_argument("--tasks", default="inference,train")
    parser.add_argument("--max-train-atoms", type=int)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--enable-openequivariance", action="store_true")
    parser.add_argument("--require-openequivariance", action="store_true")
    parser.add_argument("--num-layers", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    strict_fp32()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    config = Config.from_file(str(args.config), defaults=default_config)
    config["allow_tf32"] = False
    config["default_dtype"] = "float32"
    config["model_dtype"] = "float32"
    if args.enable_openequivariance:
        config["openequivariance_enabled"] = True
    if args.num_layers is not None:
        config["num_layers"] = int(args.num_layers)
    _set_global_options(config)
    dataset = dataset_from_config(config, prefix="dataset")
    model = model_from_config(config, initialize=True, dataset=dataset).to(device)
    nparams = sum(parameter.numel() for parameter in model.parameters())
    cutoff = float(config["r_max"])
    oeq_modules = [
        type(module).__module__ + "." + type(module).__name__
        for module in model.modules()
        if "openequivariance" in type(module).__module__.lower()
        or "openequivariance" in type(module).__name__.lower()
    ]
    oeq_metadata: dict[str, object] = {
        "required": bool(args.require_openequivariance),
        "module_count": len(oeq_modules),
        "modules": oeq_modules,
    }
    if args.require_openequivariance:
        if not oeq_modules:
            raise RuntimeError(
                "--require-openequivariance was set but the model contains no "
                "OpenEquivariance tensor-product modules"
            )
        import openequivariance
        from openequivariance._torch import extlib as oeq_extlib

        if not bool(oeq_extlib.USE_PRECOMPILED_EXTENSION):
            raise RuntimeError(
                "formal OpenEquivariance benchmark requires its precompiled "
                "extension; JIT fallback is not accepted"
            )
        if not bool(oeq_extlib.BUILT_EXTENSION):
            raise RuntimeError(
                "OpenEquivariance precompiled extension did not load: "
                f"{oeq_extlib.BUILT_EXTENSION_ERROR}"
            )
        oeq_metadata.update(
            {
                "version": openequivariance.__version__,
                "precompiled_extension": True,
                "extension_path": openequivariance.torch_ext_so_path(),
            }
        )

    rows: list[dict[str, object]] = []
    for natoms in [int(value) for value in args.sizes.split(",")]:
        for task in [value.strip() for value in args.tasks.split(",") if value.strip()]:
            if (
                task == "train"
                and args.max_train_atoms is not None
                and natoms > args.max_train_atoms
            ):
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            optimizer = None
            try:
                prepare_start = time.perf_counter()
                data, nedges = synthetic_data(
                    natoms=natoms,
                    cutoff=cutoff,
                    seed=args.seed,
                    device=device,
                )
                prepare_s = time.perf_counter() - prepare_start
                for parameter in model.parameters():
                    parameter.requires_grad_(task == "train")
                model.train(task == "train")
                if task == "train":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

                def step():
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    output = model(data.copy())
                    if optimizer is not None:
                        loss = output[AtomicDataDict.FORCE_KEY].square().mean()
                        loss = loss + output[AtomicDataDict.TOTAL_ENERGY_KEY].square().mean()
                        loss.backward()
                        optimizer.step()
                    return output

                warmup_start = time.perf_counter()
                step()
                sync()
                for _ in range(3):
                    step()
                sync()
                warmup_s = time.perf_counter() - warmup_start
                torch.cuda.reset_peak_memory_stats()
                values: list[float] = []
                for _ in range(repeats_for(natoms, task)):
                    sync()
                    started = time.perf_counter()
                    step()
                    sync()
                    values.append(1.0e3 * (time.perf_counter() - started))
                row = {
                    "engine": args.label,
                    "task": task,
                    "natoms": natoms,
                    "nedges": nedges,
                    "neighbors_per_atom": nedges / natoms,
                    "parameters": nparams,
                    "prepare_s": prepare_s,
                    "compile_s": 0.0,
                    "warmup_s": warmup_s,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    **summarize(values, natoms),
                    "status": "ok",
                }
            except torch.cuda.OutOfMemoryError as exc:
                row = {
                    "engine": args.label,
                    "task": task,
                    "natoms": natoms,
                    "parameters": nparams,
                    "status": "oom",
                    "error": str(exc),
                }
            except Exception as exc:
                row = {
                    "engine": args.label,
                    "task": task,
                    "natoms": natoms,
                    "parameters": nparams,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "protocol": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": "float32",
            "tf32": False,
            "target_neighbors_per_atom": 32,
            "graph_build_in_timing": False,
            "steady_state_excludes_prepare": True,
            "max_train_atoms": args.max_train_atoms,
            "tasks": {
                "inference": "energy plus conservative forces",
                "train": "energy+force loss, force double backward, AdamW update",
            },
        },
        "engine": args.label,
        "configuration": str(args.config),
        "configuration_overrides": {
            "openequivariance_enabled": bool(
                config.get("openequivariance_enabled", False)
            ),
            "num_layers": int(config["num_layers"]),
        },
        "parameters": nparams,
        "openequivariance": oeq_metadata,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
