#!/usr/bin/env python3
"""Cross-model training and inference throughput on fixed-size atomic graphs.

Each invocation loads exactly one model family in its native environment.  The
timed inference quantity is energy plus conservative forces.  The timed
training quantity is a complete optimizer update with energy and force losses,
including the force double backward.  Compilation and graph/data preparation
are measured separately and excluded from steady-state latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms


SPECIES = (1, 6, 7, 8)


def strict_fp32() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync() -> None:
    torch.cuda.synchronize()


def make_atoms(natoms: int, cutoff: float, degree: float, seed: int) -> Atoms:
    """Deterministic periodic mixed-element structure with 32 lattice neighbors."""
    if int(degree) != 32:
        raise ValueError("The geometry benchmark currently supports degree=32 only")

    # A simple-cubic lattice contains 6 + 12 + 8 + 6 = 32 sites through
    # the |r|=2a shell.  Placing the cutoff between 2a and sqrt(5)a makes
    # the target degree exact rather than merely correct in expectation.
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
        np.meshgrid(
            np.arange(nx),
            np.arange(ny),
            np.arange(nz),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    positions = spacing * (
        grid.astype(np.float64)
        + 0.001 * rng.normal(size=(natoms, 3))
    )
    numbers = np.asarray(SPECIES, dtype=np.int64)[np.arange(natoms) % len(SPECIES)]
    return Atoms(
        numbers=numbers,
        positions=positions,
        cell=np.diag(np.asarray([nx, ny, nz], dtype=float) * spacing),
        pbc=True,
    )


def samples(fn: Callable[[], Any], warmup: int, repeats: int) -> tuple[list[float], float]:
    start = time.perf_counter()
    for _ in range(warmup):
        fn()
    sync()
    warmup_s = time.perf_counter() - start
    values: list[float] = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        values.append(1e3 * (time.perf_counter() - t0))
    return values, warmup_s


def summary(values: list[float], natoms: int) -> dict[str, float]:
    ordered = sorted(values)
    median = float(statistics.median(ordered))
    q10 = ordered[max(0, int(0.1 * (len(ordered) - 1)))]
    q90 = ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]
    return {
        "median_ms": median,
        "mean_ms": float(statistics.mean(ordered)),
        "p10_ms": float(q10),
        "p90_ms": float(q90),
        "steps_per_second": 1000.0 / median,
        "atoms_per_second": 1000.0 * float(natoms) / median,
    }


def repeats_for(natoms: int, task: str) -> int:
    if task == "train":
        return 12 if natoms <= 128 else (8 if natoms <= 256 else 5)
    return 30 if natoms <= 128 else (20 if natoms <= 256 else 12)


def build_ictc(args: argparse.Namespace):
    from chorus.bench.synthetic_workloads import (
        benchmark_ictc_training,
        make_graph,
    )
    from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
    from chorus.training.makefx_compile import trace_and_compile_force

    phase = args.engine.startswith("chorus-")
    loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.checkpoint,
        element_types=["H", "C", "N", "O"],
        device="cuda",
    )
    model = loaded.wrapper.model
    nparams = sum(p.numel() for p in model.parameters())

    def prepare(natoms: int, task: str):
        graph = make_graph(
            atoms=natoms,
            avg_degree=args.degree,
            dtype=torch.float32,
            device=torch.device("cuda"),
            seed=args.seed + natoms,
        )
        if task == "train":
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            # ForceTrainer owns the exact MakeFX force-double-backward path.
            def run_training_benchmark():
                return benchmark_ictc_training(
                    model,
                    graph,
                    device=torch.device("cuda"),
                    dtype=torch.float32,
                    lr=1e-3,
                    warmup=2,
                    iters=repeats_for(natoms, task),
                    makefx=True,
                    require_makefx=True,
                )

            return run_training_benchmark, natoms * args.degree, "aggregate"

        inputs = (
            graph.pos,
            graph.atomic_numbers,
            graph.batch,
            graph.edge_src,
            graph.edge_dst,
            graph.unit_shifts,
            graph.cell,
        )
        t0 = time.perf_counter()
        compiled = trace_and_compile_force(
            model.eval(),
            inputs,
            training=False,
            compile_dynamic_shapes=False,
        )
        compiled(*inputs)
        sync()
        compile_s = time.perf_counter() - t0

        def infer():
            return compiled(*inputs)

        return infer, natoms * args.degree, compile_s

    return model, nparams, prepare, {
        "backend": "MakeFX/Inductor",
        "graph_build_in_timing": False,
        "configuration": (
            f"loaded paper checkpoint; C128 L2 correlation3 2 interactions "
            f"{args.engine.removeprefix('chorus-')}"
            if phase
            else "loaded paper checkpoint; C128 L2 correlation3 2 interactions phase-off"
        ),
    }


def build_native_mace(args: argparse.Namespace):
    from e3nn import o3
    from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    from chorus.bench.synthetic_workloads import (
        benchmark_native_training,
        mace_data,
        make_graph,
    )

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ScaleShiftMACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
            num_interactions=2,
            num_elements=4,
            hidden_irreps=o3.Irreps("128x0e + 128x1o + 128x2e"),
            MLP_irreps=o3.Irreps("64x0e"),
            atomic_energies=np.zeros(4),
            avg_num_neighbors=10.7168550491333,
            atomic_numbers=list(SPECIES),
            correlation=3,
            gate=gate_dict["silu"],
            radial_type="bessel",
            radial_MLP=[64, 64, 64],
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            use_reduced_cg=True,
            cueq_config=CuEquivarianceConfig(
                enabled=True,
                layout="mul_ir",
                group="O3_e3nn",
                optimize_all=True,
                conv_fusion=True,
            ),
        )
    finally:
        torch.set_default_dtype(old_dtype)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
        # Older mace-torch checkpoints store the single-head E0 with a head axis.
        state = dict(state)
        state["atomic_energies_fn.atomic_energies"] = state[
            "atomic_energies_fn.atomic_energies"
        ].reshape(-1)
        model.load_state_dict(state, strict=True)
    model = model.to(device="cuda", dtype=torch.float32)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def prepare(natoms: int, task: str):
        graph = make_graph(
            atoms=natoms,
            avg_degree=args.degree,
            dtype=torch.float32,
            device=torch.device("cuda"),
            seed=args.seed + natoms,
        )
        if task == "train":
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            def run_training_benchmark():
                return benchmark_native_training(
                    model,
                    graph,
                    device=torch.device("cuda"),
                    lr=1e-3,
                    # CuEq may specialize a new large-graph kernel after the
                    # first few optimizer steps; keep that one-time work out
                    # of the steady-state timing.
                    warmup=5,
                    iters=repeats_for(natoms, task),
                )
            return run_training_benchmark, natoms * args.degree, "aggregate"

        data = mace_data(graph)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def infer():
            return model(data, training=False, compute_force=True)

        return infer, natoms * args.degree, 0.0

    return model, nparams, prepare, {
        "backend": "CuEq-only (native MACE has no compatible MakeFX force path)",
        "graph_build_in_timing": False,
        "configuration": "C128 L2 correlation3 2 interactions",
    }


def build_dpa(args: argparse.Namespace):
    from deepmd.pt.model.model import get_model
    from deepmd.utils.argcheck import normalize

    config = normalize(json.loads(Path(args.config).read_text()))
    if args.dpa_amp == "on":
        config["model"]["descriptor"]["use_amp"] = True
    elif args.dpa_amp == "off":
        config["model"]["descriptor"]["use_amp"] = False
    config["model"]["use_compile"] = True
    config["model"]["enable_tf32"] = False
    model = get_model(config["model"])
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    state = {
        key.removeprefix("model.Default."): value
        for key, value in raw.items()
        if key.startswith("model.Default.")
    }
    model.load_state_dict(state, strict=True)
    model = model.cuda()
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def prepare(natoms: int, task: str):
        atoms = make_atoms(natoms, args.cutoff, args.degree, args.seed)
        coord = torch.as_tensor(atoms.positions, dtype=torch.float32, device="cuda")[None]
        type_map = {symbol: i for i, symbol in enumerate(config["model"]["type_map"])}
        atype = torch.tensor(
            [[type_map[symbol] for symbol in atoms.get_chemical_symbols()]],
            dtype=torch.long,
            device="cuda",
        )
        box = torch.as_tensor(atoms.cell.array, dtype=torch.float32, device="cuda").reshape(1, 9)
        if task == "train":
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            def step():
                optimizer.zero_grad(set_to_none=True)
                out = model(coord, atype, box)
                loss = out["energy"].square().mean() + 100.0 * out["force"].square().mean()
                loss.backward()
                optimizer.step()
                return loss

            t0 = time.perf_counter()
            step()
            sync()
            return step, -1, time.perf_counter() - t0

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def infer():
            return model(coord, atype, box)

        t0 = time.perf_counter()
        infer()
        sync()
        return infer, -1, time.perf_counter() - t0

    return model, nparams, prepare, {
        "backend": "DPA-4 internal torch.compile (model.use_compile=true)",
        "graph_build_in_timing": True,
        "configuration": Path(args.config).as_posix(),
        "dpa_descriptor_amp": bool(config["model"]["descriptor"]["use_amp"]),
        "dpa_training_autocast_dtype": (
            "bfloat16" if config["model"]["descriptor"]["use_amp"] else None
        ),
        "dpa_inference_amp": False,
    }


def build_tece(args: argparse.Namespace):
    os.environ["TACE_USE_CUE"] = "0"
    os.environ["TACE_USE_OEQ"] = "1" if args.tece_backend == "openeq" else "0"
    os.environ["TACE_USE_EQT"] = "0"
    os.environ["TACE_USE_COMPILE"] = "0"

    from torch_geometric.loader import DataLoader
    from tace.dataset.graph import from_atoms
    from tace.dataset.quantity import KEYS, KeySpecification, update_keyspec_from_kwargs
    from tace.lightning import load_tace

    model = load_tace(args.checkpoint, "cuda", strict=True, use_ema=False).cuda()
    openeq_modules = [
        name
        for name, module in model.named_modules()
        if "._oeq." in type(module).__module__
        or "openequivariance" in type(module).__module__.lower()
    ]
    if args.tece_backend == "openeq" and not openeq_modules:
        raise RuntimeError(
            "TACE_USE_OEQ=1 was requested, but the loaded TECE model contains "
            "no OpenEquivariance modules"
        )
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    keyspec = KeySpecification()
    update_keyspec_from_kwargs(keyspec, KEYS)
    element = model.get_torch_element()
    cutoff = float(model.get_cutoff())

    def prepare(natoms: int, task: str):
        atoms = make_atoms(natoms, cutoff, args.degree, args.seed)
        atoms.info["fidelity_idx"] = model.get_fidelity_idx()
        graph = from_atoms(
            element,
            atoms,
            cutoff,
            max_neighbors=model.get_max_neighbors(),
            target_property=model.get_target_property(),
            embedding_property=model.get_embedding_property(),
            keyspec=keyspec,
            training=task == "train",
            neighborlist_backend="matscipy",
        )
        batch = next(iter(DataLoader([graph], batch_size=1, shuffle=False))).cuda()
        nedges = int(batch.edge_index.shape[1])
        if task == "train":
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            def step():
                optimizer.zero_grad(set_to_none=True)
                out = model(batch)
                energy = out.get("energy", out.get("total_energy"))
                force = out.get("forces", out.get("force"))
                loss = energy.square().mean() + 100.0 * force.square().mean()
                loss.backward()
                optimizer.step()
                return loss

            return step, nedges, 0.0
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def infer():
            return model(batch)

        return infer, nedges, 0.0

    return model, nparams, prepare, {
        "backend": (
            "TECE OpenEquivariance"
            if args.tece_backend == "openeq"
            else "TECE native eager"
        ),
        "graph_build_in_timing": False,
        "configuration": args.checkpoint,
        "tece_openeq_module_count": len(openeq_modules),
        "tece_openeq_modules": openeq_modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        required=True,
        choices=[
            "native-mace",
            "ictc-baseline",
            "chorus-r8",
            "chorus-r16",
            "chorus-r32",
            "chorus-final-r16",
            "chorus-persistent-r16",
            "dpa4",
            "tece",
        ],
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", default="128,256,512,1024,2048,4096")
    parser.add_argument("--tasks", default="inference,train")
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--dpa-amp",
        choices=("config", "on", "off"),
        default="config",
        help=(
            "DPA-4 descriptor AMP override. 'on' enables BF16 autocast during "
            "training, 'off' enforces strict FP32, and 'config' preserves input.json."
        ),
    )
    parser.add_argument(
        "--tece-backend",
        choices=("eager", "openeq"),
        default="eager",
        help="TECE tensor-product backend.",
    )
    args = parser.parse_args()
    strict_fp32()
    if args.engine == "ictc-baseline" or args.engine.startswith("chorus-"):
        _, nparams, prepare, metadata = build_ictc(args)
    elif args.engine == "native-mace":
        _, nparams, prepare, metadata = build_native_mace(args)
    elif args.engine == "dpa4":
        _, nparams, prepare, metadata = build_dpa(args)
    else:
        _, nparams, prepare, metadata = build_tece(args)
    strict_fp32()

    rows: list[dict[str, Any]] = []
    for natoms in [int(x) for x in args.sizes.split(",")]:
        for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                t0 = time.perf_counter()
                fn, nedges, compile_marker = prepare(natoms, task)
                prepare_s = time.perf_counter() - t0
                if compile_marker == "aggregate":
                    # The MACE-family helper performs its own warmup and timing.
                    t0 = time.perf_counter()
                    result = fn()
                    aggregate_s = time.perf_counter() - t0
                    median_ms = float(result[0])
                    compile_s = float(result[3]) if len(result) > 3 else 0.0
                    timing = {
                        "median_ms": median_ms,
                        "mean_ms": median_ms,
                        "p10_ms": median_ms,
                        "p90_ms": median_ms,
                        "steps_per_second": 1000.0 / median_ms,
                        "atoms_per_second": 1000.0 * natoms / median_ms,
                    }
                    warmup_s = max(0.0, aggregate_s - median_ms * repeats_for(natoms, task) / 1e3)
                else:
                    values, warmup_s = samples(
                        fn,
                        warmup=3,
                        repeats=repeats_for(natoms, task),
                    )
                    timing = summary(values, natoms)
                    compile_s = float(compile_marker)
                row = {
                    "engine": args.engine,
                    "task": task,
                    "natoms": natoms,
                    "nedges": nedges,
                    "neighbors_per_atom": None if nedges < 0 else nedges / natoms,
                    "parameters": nparams,
                    "prepare_s": prepare_s,
                    "compile_s": compile_s,
                    "warmup_s": warmup_s,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    **timing,
                    "status": "ok",
                }
            except torch.cuda.OutOfMemoryError as exc:
                row = {
                    "engine": args.engine,
                    "task": task,
                    "natoms": natoms,
                    "parameters": nparams,
                    "status": "oom",
                    "error": str(exc),
                }
            except Exception as exc:
                row = {
                    "engine": args.engine,
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
            "dtype": (
                "mixed-fp32-bfloat16"
                if args.engine == "dpa4"
                and metadata.get("dpa_descriptor_amp", False)
                else "float32"
            ),
            "tf32": False,
            "tasks": {
                "inference": "energy plus conservative forces",
                "train": "energy+force loss, backward, optimizer update",
            },
            "steady_state_excludes_compile_and_prepare": True,
            "target_neighbors_per_atom": args.degree,
        },
        "engine": args.engine,
        "checkpoint": args.checkpoint,
        "parameters": nparams,
        **metadata,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
