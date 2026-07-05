#!/usr/bin/env python3
"""Benchmark MACE-ICTC training/inference modes against native mace-torch.

The benchmark intentionally uses small synthetic fixed-shape graphs.  It is a
kernel/backend throughput harness, not a chemistry-quality validation run:
native MACE and MACE-ICTC are configured with comparable angular/radial sizes,
but they are not weight-converted to the same parameterization.

Rows are emitted with ``status=ok|skip|error`` so unsupported backend / angular
combinations do not stop the full sweep.  The mace-torch e3nn row is used as the
per-task, per-(hidden_lmax,max_ell) baseline for the speedup column.

Example on the 4090 box:

    PYTHONPATH=/tmp/mace_torch_0_3_16:$PWD \
      python -m mace_ictc.bench.bench_mace_ictc_vs_mace \
        --device cuda --dtype float32 --atoms 64 --channels 8 \
        --configs 1:1,1:2,2:2,2:3 --out-dir /tmp/mace_ictc_bench
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

from mace_ictc.cli.export_aoti_core import (
    _aoti_compile,
    _aoti_load,
    force_compute_fn_factory,
)
from mace_ictc.models.pure_cartesian_ictd_fix import PureCartesianICTDFix
from mace_ictc.training.makefx_compile import trace_and_compile_force
from mace_ictc.training.train_loop import ForceTrainer, disable_tf32


SPECIES = (1, 6, 7, 8)
ATOMIC_ENERGIES = np.array([-13.6, -1029.0, -1485.0, -2042.0], dtype=float)
R_MAX = 5.0
AVG_NUM_NEIGHBORS = 16.0


@dataclasses.dataclass(frozen=True)
class AngularConfig:
    hidden_lmax: int
    max_ell: int


@dataclasses.dataclass
class GraphBatch:
    pos: torch.Tensor
    atomic_numbers: torch.Tensor
    species_index: torch.Tensor
    node_attrs: torch.Tensor
    batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_index: torch.Tensor
    unit_shifts: torch.Tensor
    shifts: torch.Tensor
    cell: torch.Tensor
    ptr: torch.Tensor
    force_ref: torch.Tensor
    energy_ref: torch.Tensor
    stress_ref: torch.Tensor


@contextmanager
def default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def parse_configs(text: str) -> list[AngularConfig]:
    out: list[AngularConfig] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        sep = ":" if ":" in item else "/"
        left, right = item.split(sep, 1)
        out.append(AngularConfig(hidden_lmax=int(left), max_ell=int(right)))
    if not out:
        raise ValueError("--configs produced no entries")
    return out


def parse_int_list(text: str) -> list[int]:
    out = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def hidden_irreps(channels: int, hidden_lmax: int) -> o3.Irreps:
    return o3.Irreps(
        " + ".join(
            f"{int(channels)}x{ell}{'e' if ell % 2 == 0 else 'o'}"
            for ell in range(int(hidden_lmax) + 1)
        )
    )


def one_hot_species(species_index: torch.Tensor, num_species: int, dtype: torch.dtype) -> torch.Tensor:
    return F.one_hot(species_index.cpu(), num_classes=num_species).to(dtype=dtype)


def make_graph(
    *,
    atoms: int,
    avg_degree: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> GraphBatch:
    """Create a deterministic single-graph batch with random directed edges.

    The box is deliberately large and shifts are zero.  This keeps the graph
    static and isolates model/backend throughput from neighbor-list generation.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pos = torch.randn(int(atoms), 3, generator=gen, dtype=torch.float64) * 1.5
    species_index = torch.randint(0, len(SPECIES), (int(atoms),), generator=gen)
    atomic_numbers = torch.tensor(SPECIES, dtype=torch.long)[species_index]

    edges = int(atoms) * int(avg_degree)
    edge_src = torch.randint(0, int(atoms), (edges,), generator=gen)
    edge_dst = torch.randint(0, int(atoms), (edges,), generator=gen)
    loop = edge_src == edge_dst
    edge_dst[loop] = (edge_dst[loop] + 1) % int(atoms)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)

    unit_shifts = torch.zeros(edges, 3, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * 100.0
    shifts = unit_shifts @ cell[0]
    batch = torch.zeros(int(atoms), dtype=torch.long)
    ptr = torch.tensor([0, int(atoms)], dtype=torch.long)
    node_attrs = one_hot_species(species_index, len(SPECIES), dtype=torch.float64)

    return GraphBatch(
        pos=pos.to(device=device, dtype=dtype),
        atomic_numbers=atomic_numbers.to(device=device),
        species_index=species_index.to(device=device),
        node_attrs=node_attrs.to(device=device, dtype=dtype),
        batch=batch.to(device=device),
        edge_src=edge_src.to(device=device),
        edge_dst=edge_dst.to(device=device),
        edge_index=edge_index.to(device=device),
        unit_shifts=unit_shifts.to(device=device, dtype=dtype),
        shifts=shifts.to(device=device, dtype=dtype),
        cell=cell.to(device=device, dtype=dtype),
        ptr=ptr.to(device=device),
        force_ref=torch.zeros(int(atoms), 3, device=device, dtype=dtype),
        energy_ref=torch.zeros(1, device=device, dtype=dtype),
        stress_ref=torch.zeros(1, 3, 3, device=device, dtype=dtype),
    )


def mace_data(graph: GraphBatch) -> dict[str, torch.Tensor]:
    return {
        "positions": graph.pos,
        "node_attrs": graph.node_attrs,
        "edge_index": graph.edge_index,
        "shifts": graph.shifts,
        "unit_shifts": graph.unit_shifts,
        "cell": graph.cell,
        "batch": graph.batch,
        "ptr": graph.ptr,
    }


def ictd_batch_tuple(graph: GraphBatch) -> tuple[torch.Tensor, ...]:
    return (
        graph.pos,
        graph.atomic_numbers,
        graph.batch,
        graph.force_ref,
        graph.energy_ref,
        graph.edge_src,
        graph.edge_dst,
        graph.unit_shifts,
        graph.cell,
        graph.stress_ref,
    )


def build_native_mace(
    *,
    backend: str,
    cfg: AngularConfig,
    channels: int,
    num_interactions: int,
    correlation: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes

    cueq_config = None
    if backend == "cueq":
        from mace.modules.wrapper_ops import CuEquivarianceConfig

        cueq_config = CuEquivarianceConfig(
            enabled=True,
            layout="mul_ir",
            group="O3_e3nn",
            optimize_all=True,
            conv_fusion=True,
        )

    with default_dtype(dtype):
        model = ScaleShiftMACE(
            r_max=R_MAX,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=int(cfg.max_ell),
            interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
            num_interactions=int(num_interactions),
            num_elements=len(SPECIES),
            hidden_irreps=hidden_irreps(channels, int(cfg.hidden_lmax)),
            MLP_irreps=o3.Irreps(f"{int(channels)}x0e"),
            atomic_energies=ATOMIC_ENERGIES,
            avg_num_neighbors=AVG_NUM_NEIGHBORS,
            atomic_numbers=list(SPECIES),
            correlation=int(correlation),
            gate=gate_dict["silu"],
            radial_type="bessel",
            radial_MLP=[int(channels), int(channels), int(channels)],
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            use_reduced_cg=True,
            cueq_config=cueq_config,
        )
    return model.to(device=device, dtype=dtype)


def build_ictd(
    *,
    product_backend: str,
    cfg: AngularConfig,
    channels: int,
    num_interactions: int,
    correlation: int,
    dtype: torch.dtype,
    device: torch.device,
    use_reduced_cg: bool = True,
) -> PureCartesianICTDFix:
    with default_dtype(dtype):
        model = PureCartesianICTDFix(
            max_embed_radius=R_MAX,
            main_max_radius=R_MAX,
            main_number_of_basis=8,
            hidden_dim_conv=int(channels),
            hidden_dim_sh=int(channels),
            hidden_dim=int(channels),
            channel_in2=int(channels),
            embedding_dim=int(channels),
            max_atomvalue=10,
            atomic_numbers=list(SPECIES),
            output_size=8,
            embed_size=[int(channels), int(channels)],
            main_hidden_sizes3=[int(channels)],
            num_layers=1,
            num_interaction=int(num_interactions),
            function_type_main="bessel",
            lmax=int(cfg.hidden_lmax),
            ictd_fix_edge_lmax=int(cfg.max_ell),
            ictd_fix_route="baseline",
            ictd_fix_product_backend=str(product_backend),
            ictd_fix_use_reduced_cg=bool(use_reduced_cg),
            ictd_fix_fusion_scale_init=1.0,
            ictd_fix_fusion_heads=1,
            save_contraction_order=int(correlation),
            avg_num_neighbors=AVG_NUM_NEIGHBORS,
            radial_sqrt_num_basis=False,
            angular_basis="ictd",
            internal_compute_dtype=dtype,
            device=device,
        )
    return model.to(device=device, dtype=dtype)


def time_callable(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(int(warmup)):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(int(iters)):
        fn()
    sync(device)
    return (time.perf_counter() - start) * 1e3 / max(1, int(iters))


def benchmark_native_training(
    model: torch.nn.Module,
    graph: GraphBatch,
    *,
    device: torch.device,
    lr: float,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    data = mace_data(graph)
    atoms = float(graph.pos.shape[0])
    last_loss = 0.0

    def step() -> None:
        nonlocal last_loss
        opt.zero_grad(set_to_none=True)
        if data["positions"].grad is not None:
            data["positions"].grad = None
        out = model(data, training=True, compute_force=True)
        energy = out["energy"].reshape(-1) / atoms
        forces = out["forces"]
        loss = F.smooth_l1_loss(energy, torch.zeros_like(energy), beta=0.5)
        loss = loss + 10.0 * F.smooth_l1_loss(forces, torch.zeros_like(forces), beta=0.5)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())

    ms = time_callable(step, device=device, warmup=warmup, iters=iters)
    return ms, last_loss


def benchmark_ictd_training(
    model: torch.nn.Module,
    graph: GraphBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    warmup: int,
    iters: int,
    makefx: bool,
    require_makefx: bool,
) -> tuple[float, float, int, float]:
    batch = ictd_batch_tuple(graph)
    trainer = ForceTrainer(
        model,
        [batch],
        device=device,
        dtype=dtype,
        learning_rate=float(lr),
        lr_scheduler="none",
        train_makefx_compile=bool(makefx),
        require_train_makefx_compile=bool(require_makefx),
        makefx_max_slots=2,
        energy_weight=1.0,
        force_weight=10.0,
        stress_weight=0.0,
        atomic_energy_keys=SPECIES,
        atomic_energy_values=[0.0] * len(SPECIES),
        epochs=1,
        log_interval=0,
    )
    model.train()
    last_loss = 0.0

    def step() -> None:
        nonlocal last_loss
        trainer.optimizer.zero_grad(set_to_none=True)
        out = trainer._compute(batch, training=True)  # benchmark harness for one fixed batch
        loss = out["total_loss"]
        loss.backward()
        trainer.optimizer.step()
        last_loss = float(loss.detach())

    compile_s = 0.0
    if makefx:
        t0 = time.perf_counter()
        step()
        sync(device)
        compile_s = time.perf_counter() - t0
        timed_warmup = max(0, int(warmup) - 1)
    else:
        timed_warmup = int(warmup)

    ms = time_callable(step, device=device, warmup=timed_warmup, iters=iters)
    cache_size = len(getattr(getattr(trainer, "_makefx_cache", None), "_cache", {}))
    return ms, last_loss, cache_size if makefx else 0, compile_s


def ictd_force_fn(model: torch.nn.Module, graph: GraphBatch, *, training: bool = False):
    def run():
        pos = graph.pos.detach().requires_grad_(True)
        out = model(
            pos,
            graph.atomic_numbers,
            graph.batch,
            graph.edge_src,
            graph.edge_dst,
            graph.unit_shifts,
            graph.cell,
        )
        e_atom = out[0] if isinstance(out, tuple) else out
        grad = torch.autograd.grad(e_atom.sum(), pos, create_graph=training)[0]
        return e_atom, -grad

    return run


def benchmark_ictd_inference_eager(
    model: torch.nn.Module,
    graph: GraphBatch,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    model.eval()
    fn = ictd_force_fn(model, graph, training=False)
    return time_callable(fn, device=device, warmup=warmup, iters=iters)


def benchmark_native_inference(
    model: torch.nn.Module,
    graph: GraphBatch,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    model.eval()
    data = mace_data(graph)

    def fn():
        if data["positions"].grad is not None:
            data["positions"].grad = None
        return model(data, training=False, compute_force=True)

    return time_callable(fn, device=device, warmup=warmup, iters=iters)


def benchmark_ictd_inference_aoti(
    model: torch.nn.Module,
    graph: GraphBatch,
    *,
    device: torch.device,
    out_dir: Path,
    stem: str,
    warmup: int,
    iters: int,
    export_strict: bool,
) -> tuple[float, float, str]:
    model.eval()
    example_inputs = (
        graph.pos,
        graph.atomic_numbers,
        graph.batch,
        graph.edge_src,
        graph.edge_dst,
        graph.unit_shifts,
        graph.cell,
    )
    package_path = out_dir / f"{stem}.pt2"
    if package_path.exists():
        package_path.unlink()

    t0 = time.perf_counter()
    gm = trace_and_compile_force(
        model,
        example_inputs,
        training=False,
        compute_fn=force_compute_fn_factory(model, training=False),
        do_compile=False,
    )
    exported = torch.export.export(gm, tuple(example_inputs), strict=bool(export_strict))
    _aoti_compile(exported, str(package_path))
    runner = _aoti_load(str(package_path), device)
    sync(device)
    compile_s = time.perf_counter() - t0

    def fn():
        return runner(*example_inputs)

    ms = time_callable(fn, device=device, warmup=warmup, iters=iters)
    return ms, compile_s, str(package_path)


def short_error(exc: BaseException) -> tuple[str, str]:
    msg = f"{type(exc).__name__}: {exc}"
    return type(exc).__name__, msg.replace("\n", " ")[:500]


def add_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    row = {
        "task": kwargs.pop("task"),
        "mode": kwargs.pop("mode"),
        "atoms": kwargs.pop("atoms"),
        "hidden_lmax": kwargs.pop("hidden_lmax"),
        "max_ell": kwargs.pop("max_ell"),
        "status": kwargs.pop("status", "ok"),
        "time_ms": kwargs.pop("time_ms", ""),
        "compile_s": kwargs.pop("compile_s", ""),
        "baseline_ms": kwargs.pop("baseline_ms", ""),
        "speedup_vs_mace_e3nn": kwargs.pop("speedup_vs_mace_e3nn", ""),
        "loss": kwargs.pop("loss", ""),
        "cache_entries": kwargs.pop("cache_entries", ""),
        "artifact": kwargs.pop("artifact", ""),
        "note": kwargs.pop("note", ""),
        "error_type": kwargs.pop("error_type", ""),
        "error": kwargs.pop("error", ""),
    }
    row.update(kwargs)
    rows.append(row)


def annotate_speedups(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, int, int, int], float] = {}
    for row in rows:
        if (
            row["status"] == "ok"
            and row["mode"] == "mace_torch_e3nn"
            and row["time_ms"] != ""
        ):
            baselines[
                (row["task"], int(row["atoms"]), int(row["hidden_lmax"]), int(row["max_ell"]))
            ] = float(row["time_ms"])
    for row in rows:
        key = (row["task"], int(row["atoms"]), int(row["hidden_lmax"]), int(row["max_ell"]))
        baseline = baselines.get(key)
        if baseline is None:
            continue
        row["baseline_ms"] = f"{baseline:.6g}"
        if row["status"] == "ok" and row["time_ms"] != "":
            row["speedup_vs_mace_e3nn"] = f"{baseline / float(row['time_ms']):.6g}"


def write_outputs(rows: list[dict[str, Any]], meta: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"mace_ictc_vs_mace_bench_{stamp}.csv"
    json_path = out_dir / f"mace_ictc_vs_mace_bench_{stamp}.json"
    md_path = out_dir / f"mace_ictc_vs_mace_bench_{stamp}.md"

    fieldnames = [
        "task",
        "mode",
        "atoms",
        "hidden_lmax",
        "max_ell",
        "status",
        "time_ms",
        "compile_s",
        "baseline_ms",
        "speedup_vs_mace_e3nn",
        "loss",
        "cache_entries",
        "artifact",
        "note",
        "error_type",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    with md_path.open("w") as f:
        f.write("# MACE-ICTC vs mace-torch benchmark\n\n")
        f.write("## Metadata\n\n")
        for key, value in meta.items():
            f.write(f"- {key}: `{value}`\n")
        f.write("\n## OK rows\n\n")
        f.write("| task | mode | atoms | lmax | max_ell | ms | compile_s | speedup_vs_e3nn |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in ok_rows:
            f.write(
                f"| {row['task']} | {row['mode']} | {row['atoms']} | "
                f"{row['hidden_lmax']} | {row['max_ell']} | "
                f"{row['time_ms']} | {row['compile_s']} | {row['speedup_vs_mace_e3nn']} |\n"
            )
        non_ok = [r for r in rows if r["status"] != "ok"]
        if non_ok:
            f.write("\n## Skips and errors\n\n")
            f.write("| task | mode | atoms | lmax | max_ell | status | note/error |\n")
            f.write("|---|---:|---:|---:|---:|---|---|\n")
            for row in non_ok:
                detail = row["note"] or row["error"]
                f.write(
                    f"| {row['task']} | {row['mode']} | {row['atoms']} | "
                    f"{row['hidden_lmax']} | {row['max_ell']} | "
                    f"{row['status']} | {detail} |\n"
                )

    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def row_ok(
    rows: list[dict[str, Any]],
    *,
    task: str,
    mode: str,
    atoms: int,
    cfg: AngularConfig,
    time_ms: float,
    compile_s: float | str = "",
    loss: float | str = "",
    cache_entries: int | str = "",
    artifact: str = "",
    note: str = "",
) -> None:
    add_row(
        rows,
        task=task,
        mode=mode,
        atoms=int(atoms),
        hidden_lmax=cfg.hidden_lmax,
        max_ell=cfg.max_ell,
        status="ok",
        time_ms=f"{float(time_ms):.6g}",
        compile_s=f"{float(compile_s):.6g}" if compile_s != "" else "",
        loss=f"{float(loss):.6g}" if loss != "" else "",
        cache_entries=cache_entries,
        artifact=artifact,
        note=note,
    )


def row_error(
    rows: list[dict[str, Any]],
    *,
    task: str,
    mode: str,
    atoms: int,
    cfg: AngularConfig,
    exc: BaseException,
) -> None:
    etype, msg = short_error(exc)
    add_row(
        rows,
        task=task,
        mode=mode,
        atoms=int(atoms),
        hidden_lmax=cfg.hidden_lmax,
        max_ell=cfg.max_ell,
        status="error",
        error_type=etype,
        error=msg,
    )


def row_skip(
    rows: list[dict[str, Any]],
    *,
    task: str,
    mode: str,
    atoms: int,
    cfg: AngularConfig,
    note: str,
) -> None:
    add_row(
        rows,
        task=task,
        mode=mode,
        atoms=int(atoms),
        hidden_lmax=cfg.hidden_lmax,
        max_ell=cfg.max_ell,
        status="skip",
        note=note,
    )


def run_config(args: argparse.Namespace, cfg: AngularConfig, rows: list[dict[str, Any]], *, atoms: int) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    graph = make_graph(
        atoms=atoms,
        avg_degree=args.avg_degree,
        dtype=dtype,
        device=device,
        seed=args.seed + cfg.hidden_lmax * 100 + cfg.max_ell,
    )
    aoti_dir = Path(args.out_dir) / "aoti_packages"
    aoti_dir.mkdir(parents=True, exist_ok=True)

    native_modes = [
        ("mace_torch_e3nn", "e3nn"),
        ("mace_torch_cueq", "cueq"),
    ]
    for mode_name, backend in native_modes:
        try:
            model = build_native_mace(
                backend=backend,
                cfg=cfg,
                channels=args.channels,
                num_interactions=args.num_interactions,
                correlation=args.correlation,
                dtype=dtype,
                device=device,
            )
            ms, loss = benchmark_native_training(
                model,
                graph,
                device=device,
                lr=args.lr,
                warmup=args.train_warmup,
                iters=args.train_iters,
            )
            row_ok(rows, task="train", mode=mode_name, atoms=atoms, cfg=cfg, time_ms=ms, loss=loss)
        except Exception as exc:  # noqa: BLE001
            row_error(rows, task="train", mode=mode_name, atoms=atoms, cfg=cfg, exc=exc)
            if args.verbose_errors:
                traceback.print_exc()

        try:
            model = build_native_mace(
                backend=backend,
                cfg=cfg,
                channels=args.channels,
                num_interactions=args.num_interactions,
                correlation=args.correlation,
                dtype=dtype,
                device=device,
            )
            ms = benchmark_native_inference(
                model,
                graph,
                device=device,
                warmup=args.infer_warmup,
                iters=args.infer_iters,
            )
            row_ok(rows, task="inference", mode=mode_name, atoms=atoms, cfg=cfg, time_ms=ms)
        except Exception as exc:  # noqa: BLE001
            row_error(rows, task="inference", mode=mode_name, atoms=atoms, cfg=cfg, exc=exc)
            if args.verbose_errors:
                traceback.print_exc()

    for task_mode, makefx in [
        ("mace_ictc_bridge_u_eager", False),
        ("mace_ictc_bridge_u_makefx_train", True),
    ]:
        try:
            model = build_ictd(
                product_backend="ictd-bridge-u",
                cfg=cfg,
                channels=args.channels,
                num_interactions=args.num_interactions,
                correlation=args.correlation,
                dtype=dtype,
                device=device,
                use_reduced_cg=True,
            )
            ms, loss, cache_entries, compile_s = benchmark_ictd_training(
                model,
                graph,
                device=device,
                dtype=dtype,
                lr=args.lr,
                warmup=args.train_warmup,
                iters=args.train_iters,
                makefx=makefx,
                require_makefx=makefx,
            )
            row_ok(
                rows,
                task="train",
                mode=task_mode,
                atoms=atoms,
                cfg=cfg,
                time_ms=ms,
                compile_s=compile_s if makefx else "",
                loss=loss,
                cache_entries=cache_entries,
            )
        except Exception as exc:  # noqa: BLE001
            row_error(rows, task="train", mode=task_mode, atoms=atoms, cfg=cfg, exc=exc)
            if args.verbose_errors:
                traceback.print_exc()

    try:
        model = build_ictd(
            product_backend="ictd-bridge-u",
            cfg=cfg,
            channels=args.channels,
            num_interactions=args.num_interactions,
            correlation=args.correlation,
            dtype=dtype,
            device=device,
            use_reduced_cg=True,
        )
        ms = benchmark_ictd_inference_eager(
            model,
            graph,
            device=device,
            warmup=args.infer_warmup,
            iters=args.infer_iters,
        )
        row_ok(rows, task="inference", mode="mace_ictc_bridge_u_eager", atoms=atoms, cfg=cfg, time_ms=ms)
    except Exception as exc:  # noqa: BLE001
        row_error(rows, task="inference", mode="mace_ictc_bridge_u_eager", atoms=atoms, cfg=cfg, exc=exc)
        if args.verbose_errors:
            traceback.print_exc()

    if not args.no_aoti:
        try:
            model = build_ictd(
                product_backend="ictd-bridge-u",
                cfg=cfg,
                channels=args.channels,
                num_interactions=args.num_interactions,
                correlation=args.correlation,
                dtype=dtype,
                device=device,
                use_reduced_cg=True,
            )
            ms, compile_s, artifact = benchmark_ictd_inference_aoti(
                model,
                graph,
                device=device,
                out_dir=aoti_dir,
                stem=f"bridge_u_l{cfg.hidden_lmax}_e{cfg.max_ell}_n{atoms}",
                warmup=args.infer_warmup,
                iters=args.infer_iters,
                export_strict=True,
            )
            row_ok(
                rows,
                task="inference",
                mode="mace_ictc_bridge_u_aoti",
                atoms=atoms,
                cfg=cfg,
                time_ms=ms,
                compile_s=compile_s,
                artifact=artifact,
            )
        except Exception as exc:  # noqa: BLE001
            row_error(rows, task="inference", mode="mace_ictc_bridge_u_aoti", atoms=atoms, cfg=cfg, exc=exc)
            if args.verbose_errors:
                traceback.print_exc()

    pure_u_supported = cfg.hidden_lmax == cfg.max_ell and cfg.hidden_lmax <= 3
    pure_u_note = "ictd-pure-u supports only hidden_lmax == max_ell and lmax <= 3 in this build"

    if args.include_pure_u and pure_u_supported:
        for task_mode, makefx in [
            ("mace_ictc_pure_u_eager", False),
            ("mace_ictc_pure_u_makefx_train", True),
        ]:
            try:
                model = build_ictd(
                    product_backend="ictd-pure-u",
                    cfg=cfg,
                    channels=args.channels,
                    num_interactions=args.num_interactions,
                    correlation=args.correlation,
                    dtype=dtype,
                    device=device,
                    use_reduced_cg=False,
                )
                ms, loss, cache_entries, compile_s = benchmark_ictd_training(
                    model,
                    graph,
                    device=device,
                    dtype=dtype,
                    lr=args.lr,
                    warmup=args.train_warmup,
                    iters=args.train_iters,
                    makefx=makefx,
                    require_makefx=makefx,
                )
                row_ok(
                    rows,
                    task="train",
                    mode=task_mode,
                    atoms=atoms,
                    cfg=cfg,
                    time_ms=ms,
                    compile_s=compile_s if makefx else "",
                    loss=loss,
                    cache_entries=cache_entries,
                )
            except Exception as exc:  # noqa: BLE001
                row_error(rows, task="train", mode=task_mode, atoms=atoms, cfg=cfg, exc=exc)
                if args.verbose_errors:
                    traceback.print_exc()

        try:
            model = build_ictd(
                product_backend="ictd-pure-u",
                cfg=cfg,
                channels=args.channels,
                num_interactions=args.num_interactions,
                correlation=args.correlation,
                dtype=dtype,
                device=device,
                use_reduced_cg=False,
            )
            ms = benchmark_ictd_inference_eager(
                model,
                graph,
                device=device,
                warmup=args.infer_warmup,
                iters=args.infer_iters,
            )
            row_ok(rows, task="inference", mode="mace_ictc_pure_u_eager", atoms=atoms, cfg=cfg, time_ms=ms)
        except Exception as exc:  # noqa: BLE001
            row_error(rows, task="inference", mode="mace_ictc_pure_u_eager", atoms=atoms, cfg=cfg, exc=exc)
            if args.verbose_errors:
                traceback.print_exc()

        if not args.no_aoti:
            try:
                model = build_ictd(
                    product_backend="ictd-pure-u",
                    cfg=cfg,
                    channels=args.channels,
                    num_interactions=args.num_interactions,
                    correlation=args.correlation,
                    dtype=dtype,
                    device=device,
                    use_reduced_cg=False,
                )
                ms, compile_s, artifact = benchmark_ictd_inference_aoti(
                    model,
                    graph,
                    device=device,
                    out_dir=aoti_dir,
                    stem=f"pure_u_l{cfg.hidden_lmax}_e{cfg.max_ell}_n{atoms}",
                    warmup=args.infer_warmup,
                    iters=args.infer_iters,
                    export_strict=True,
                )
                row_ok(
                    rows,
                    task="inference",
                    mode="mace_ictc_pure_u_aoti",
                    atoms=atoms,
                    cfg=cfg,
                    time_ms=ms,
                    compile_s=compile_s,
                    artifact=artifact,
                )
            except Exception as exc:  # noqa: BLE001
                row_error(rows, task="inference", mode="mace_ictc_pure_u_aoti", atoms=atoms, cfg=cfg, exc=exc)
                if args.verbose_errors:
                    traceback.print_exc()
    elif args.include_pure_u:
        for task, mode in [
            ("train", "mace_ictc_pure_u_eager"),
            ("train", "mace_ictc_pure_u_makefx_train"),
            ("inference", "mace_ictc_pure_u_eager"),
            ("inference", "mace_ictc_pure_u_aoti"),
        ]:
            row_skip(rows, task=task, mode=mode, atoms=atoms, cfg=cfg, note=pure_u_note)

    for task_mode, makefx in [
        ("mace_ictc_cueq_product_eager", False),
        ("mace_ictc_cueq_product_makefx_train", True),
    ]:
        try:
            model = build_ictd(
                product_backend="cueq",
                cfg=cfg,
                channels=args.channels,
                num_interactions=args.num_interactions,
                correlation=args.correlation,
                dtype=dtype,
                device=device,
                use_reduced_cg=True,
            )
            ms, loss, cache_entries, compile_s = benchmark_ictd_training(
                model,
                graph,
                device=device,
                dtype=dtype,
                lr=args.lr,
                warmup=args.train_warmup,
                iters=args.train_iters,
                makefx=makefx,
                require_makefx=makefx,
            )
            row_ok(
                rows,
                task="train",
                mode=task_mode,
                atoms=atoms,
                cfg=cfg,
                time_ms=ms,
                compile_s=compile_s if makefx else "",
                loss=loss,
                cache_entries=cache_entries,
            )
        except Exception as exc:  # noqa: BLE001
            row_error(rows, task="train", mode=task_mode, atoms=atoms, cfg=cfg, exc=exc)
            if args.verbose_errors:
                traceback.print_exc()

    try:
        model = build_ictd(
            product_backend="cueq",
            cfg=cfg,
            channels=args.channels,
            num_interactions=args.num_interactions,
            correlation=args.correlation,
            dtype=dtype,
            device=device,
            use_reduced_cg=True,
        )
        ms = benchmark_ictd_inference_eager(
            model,
            graph,
            device=device,
            warmup=args.infer_warmup,
            iters=args.infer_iters,
        )
        row_ok(rows, task="inference", mode="mace_ictc_cueq_product_eager", atoms=atoms, cfg=cfg, time_ms=ms)
    except Exception as exc:  # noqa: BLE001
        row_error(rows, task="inference", mode="mace_ictc_cueq_product_eager", atoms=atoms, cfg=cfg, exc=exc)
        if args.verbose_errors:
            traceback.print_exc()

    if not args.no_aoti:
        try:
            model = build_ictd(
                product_backend="cueq",
                cfg=cfg,
                channels=args.channels,
                num_interactions=args.num_interactions,
                correlation=args.correlation,
                dtype=dtype,
                device=device,
                use_reduced_cg=True,
            )
            ms, compile_s, artifact = benchmark_ictd_inference_aoti(
                model,
                graph,
                device=device,
                out_dir=aoti_dir,
                stem=f"cueq_product_l{cfg.hidden_lmax}_e{cfg.max_ell}_n{atoms}",
                warmup=args.infer_warmup,
                iters=args.infer_iters,
                export_strict=False,
            )
            row_ok(
                rows,
                task="inference",
                mode="mace_ictc_cueq_product_aoti",
                atoms=atoms,
                cfg=cfg,
                time_ms=ms,
                compile_s=compile_s,
                artifact=artifact,
            )
        except Exception as exc:  # noqa: BLE001
            row_error(rows, task="inference", mode="mace_ictc_cueq_product_aoti", atoms=atoms, cfg=cfg, exc=exc)
            if args.verbose_errors:
                traceback.print_exc()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "fp32", "float64", "fp64", "bfloat16", "bf16"])
    p.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "medium"],
                   help="float32 matmul precision. Only 'highest' is allowed; TF32 is disallowed.")
    p.add_argument("--configs", default="1:1,1:2,2:2,2:3", help="comma list hidden_lmax:max_ell")
    p.add_argument("--atoms", type=int, default=64)
    p.add_argument("--atoms-list", default="", help="comma list of atom counts; overrides --atoms")
    p.add_argument("--avg-degree", type=int, default=16)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--num-interactions", type=int, default=2)
    p.add_argument("--correlation", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260614)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-warmup", type=int, default=1)
    p.add_argument("--train-iters", type=int, default=3)
    p.add_argument("--infer-warmup", type=int, default=2)
    p.add_argument("--infer-iters", type=int, default=10)
    p.add_argument("--no-aoti", action="store_true", help="skip MACE-ICTC AOTI inference rows")
    p.add_argument("--include-pure-u", action="store_true", help="also benchmark ictd-pure-u diagnostic backend")
    p.add_argument("--out-dir", default=str(Path(tempfile.gettempdir()) / "mace_ictc_bench"))
    p.add_argument("--verbose-errors", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.matmul_precision != "highest":
        raise ValueError("TF32 is not allowed; use --matmul-precision highest")
    disable_tf32()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    configs = parse_configs(args.configs)
    atoms_list = parse_int_list(args.atoms_list) if args.atoms_list else [int(args.atoms)]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    try:
        import mace

        mace_file = getattr(mace, "__file__", "")
        mace_version = getattr(mace, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        mace_file = f"import failed: {exc}"
        mace_version = "unavailable"

    meta = {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "mace_version": mace_version,
        "mace_file": mace_file,
        "device": str(device),
        "dtype": args.dtype,
        "matmul_precision": args.matmul_precision,
        "atoms": ",".join(str(x) for x in atoms_list),
        "avg_degree": args.avg_degree,
        "channels": args.channels,
        "num_interactions": args.num_interactions,
        "correlation": args.correlation,
        "configs": args.configs,
        "train_iters": args.train_iters,
        "infer_iters": args.infer_iters,
        "aoti": not args.no_aoti,
        "include_pure_u": bool(args.include_pure_u),
    }

    print(json.dumps({"meta": meta}, indent=2), flush=True)
    rows: list[dict[str, Any]] = []
    for atoms in atoms_list:
        for cfg in configs:
            print(
                f"## atoms={atoms} hidden_lmax={cfg.hidden_lmax} max_ell={cfg.max_ell}",
                flush=True,
            )
            before = len(rows)
            run_config(args, cfg, rows, atoms=atoms)
            annotate_speedups(rows)
            for row in rows[before:]:
                print(json.dumps(row, ensure_ascii=False), flush=True)

    annotate_speedups(rows)
    paths = write_outputs(rows, meta, Path(args.out_dir))
    print(json.dumps({"outputs": paths}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
