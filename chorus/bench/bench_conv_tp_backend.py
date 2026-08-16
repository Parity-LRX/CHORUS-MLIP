#!/usr/bin/env python3
"""Operator-level throughput: interaction conv-TP, pytorch vs cuEquivariance.

Times ONLY the message-passing tensor product of
``ICTDResidualInteractionBlock`` (the part that ``ictd_tp_backend='cueq'``
replaces), including the exact boundary work each backend performs in the
model forward:

  * ``pytorch``: node-feature gather ``x1[l][edge_src]`` + the reference
    ``EdgeWeightedPathPreservingTensorProduct`` (dense per-(l1, l2) projector
    matmuls in the ICTC basis);
  * ``cueq``: per-l Q basis bridges + fused
    ``cuet.ChannelWiseTensorProduct`` kernel (internal gather via
    ``indices_1``) + output reassembly into path-preserving ICTC blocks.

Both rows share one parameterization: a single block is built per backend with
the same seed and the cueq block loads the reference state_dict (the mirror
adds no parameters).  FP32, TF32 disabled.  Each iteration is synchronized and
timed individually; the median is reported.

Example on the 4090 box:

    python -m chorus.bench.bench_conv_tp_backend \
        --device cuda --channels 72 --lmax 3 --nodes 256,1024,4096 \
        --edges-per-node 32 --out-dir /tmp/conv_tp_bench
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from chorus.models.ictd_irreps import direction_harmonics_all
from chorus.models.pure_cartesian_ictd_fix import ICTDResidualInteractionBlock
from chorus.training.train_loop import disable_tf32


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_block(
    backend: str,
    *,
    channels: int,
    lmax: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> ICTDResidualInteractionBlock:
    torch.manual_seed(seed)
    block = ICTDResidualInteractionBlock(
        channels=int(channels),
        lmax=int(lmax),
        input_lmax=int(lmax),
        number_of_basis=8,
        num_elements=3,
        internal_compute_dtype=dtype,
        ictd_tp_backend=backend,
    )
    return block.to(device=device, dtype=dtype)


def make_operator_inputs(
    block: ICTDResidualInteractionBlock,
    *,
    num_nodes: int,
    num_edges: int,
    channels: int,
    lmax: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    """Node-level features, per-edge harmonics/gates and a fixed cotangent.

    Shapes match what the interaction block hands to the conv-TP in the model:
    x1 {l: (N, C, 2l+1)}, edge_attrs {l: (E, 1, 2l+1)}, gates (E, P*C).
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x1 = {
        l: (0.5 * torch.randn(num_nodes, channels, 2 * l + 1, generator=g)).to(
            device=device, dtype=dtype
        )
        for l in range(lmax + 1)
    }
    dirs = torch.randn(num_edges, 3, generator=g, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y = direction_harmonics_all(dirs, lmax)
    edge_attrs = {l: y[l].to(device=device, dtype=dtype).unsqueeze(-2) for l in range(lmax + 1)}
    num_paths = int(block.tp.num_paths)
    gates = (0.5 * torch.randn(num_edges, num_paths * channels, generator=g)).to(
        device=device, dtype=dtype
    )
    edge_src = torch.randint(0, num_nodes, (num_edges,), generator=g).to(device)
    cotangent = {
        l: (0.5 * torch.randn(num_edges, channels * int(block.tp.path_counts_by_l.get(l, 0)), 2 * l + 1, generator=g)).to(
            device=device, dtype=dtype
        )
        for l in range(lmax + 1)
        if int(block.tp.path_counts_by_l.get(l, 0)) > 0
    }
    return {
        "x1": x1,
        "edge_attrs": edge_attrs,
        "gates": gates,
        "edge_src": edge_src,
        "cotangent": cotangent,
    }


def mark_leaves(inputs: dict[str, Any], weight: torch.Tensor) -> list[torch.Tensor]:
    leaves: list[torch.Tensor] = []
    for t in inputs["x1"].values():
        t.requires_grad_(True)
        leaves.append(t)
    for t in inputs["edge_attrs"].values():
        t.requires_grad_(True)
        leaves.append(t)
    inputs["gates"].requires_grad_(True)
    leaves.append(inputs["gates"])
    weight.requires_grad_(True)
    leaves.append(weight)
    return leaves


def clear_grads(leaves: list[torch.Tensor]) -> None:
    for t in leaves:
        t.grad = None


def make_pytorch_fn(block: ICTDResidualInteractionBlock, inputs: dict[str, Any], *, backward: bool) -> Callable[[], None]:
    def fn() -> None:
        x1e = {l: inputs["x1"][l][inputs["edge_src"]] for l in range(block.input_lmax + 1)}
        blocks = block.tp(x1e, inputs["edge_attrs"], inputs["gates"])
        if backward:
            loss = sum((blocks[l] * inputs["cotangent"][l]).sum() for l in inputs["cotangent"])
            loss.backward()

    return fn


def make_cueq_fn(block: ICTDResidualInteractionBlock, inputs: dict[str, Any], *, backward: bool) -> Callable[[], None]:
    def fn() -> None:
        blocks = block.tp_cueq(
            inputs["x1"], inputs["edge_src"], inputs["edge_attrs"], inputs["gates"], block.tp.weight
        )
        if backward:
            loss = sum((blocks[l] * inputs["cotangent"][l]).sum() for l in inputs["cotangent"])
            loss.backward()

    return fn


def time_median(fn: Callable[[], None], *, device: torch.device, warmup: int, iters: int, leaves: list[torch.Tensor] | None = None) -> float:
    # leaves=None marks the fwd-only mode: run under no_grad so nn.Parameters
    # (requires_grad=True by default) do not make the reference TP save its
    # dense per-(l1, l2) intermediates for a backward nobody will call (OOM at
    # large edge counts).
    ctx = torch.no_grad() if leaves is None else contextlib.nullcontext()
    with ctx:
        for _ in range(int(warmup)):
            if leaves is not None:
                clear_grads(leaves)
            fn()
        sync(device)
        samples: list[float] = []
        for _ in range(int(iters)):
            if leaves is not None:
                clear_grads(leaves)
            start = time.perf_counter()
            fn()
            sync(device)
            samples.append((time.perf_counter() - start) * 1e3)
    return float(statistics.median(samples))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--channels", type=int, default=72)
    ap.add_argument("--lmax", type=int, default=3)
    ap.add_argument("--nodes", default="256,1024,4096")
    ap.add_argument("--edges-per-node", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--backends", default="pytorch,cueq")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.float32
    disable_tf32()
    want_backends = {x.strip() for x in str(args.backends).split(",") if x.strip()}

    nodes_list = [int(x) for x in str(args.nodes).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for num_nodes in nodes_list:
        num_edges = int(num_nodes) * int(args.edges_per_node)
        ref = build_block("pytorch", channels=args.channels, lmax=args.lmax, seed=args.seed, dtype=dtype, device=device)
        inputs = make_operator_inputs(
            ref,
            num_nodes=num_nodes,
            num_edges=num_edges,
            channels=args.channels,
            lmax=args.lmax,
            seed=args.seed + 1,
            dtype=dtype,
            device=device,
        )
        backends: list[tuple[str, ICTDResidualInteractionBlock]] = []
        if "pytorch" in want_backends:
            backends.append(("pytorch", ref))
        cueq_block = None
        if device.type == "cuda" and "cueq" in want_backends:
            try:
                cueq_block = build_block("cueq", channels=args.channels, lmax=args.lmax, seed=args.seed, dtype=dtype, device=device)
                cueq_block.load_state_dict(ref.state_dict())
                backends.append(("cueq", cueq_block))
            except Exception as exc:  # noqa: BLE001
                rows.append({"nodes": num_nodes, "edges": num_edges, "backend": "cueq", "mode": "fwd", "ms": "", "status": f"error: {exc}"})
        for name, block in backends:
            for mode, backward in [("fwd", False), ("fwd+bwd", True)]:
                leaves = mark_leaves(inputs, block.tp.weight) if backward else None
                if name == "cueq":
                    fn = make_cueq_fn(block, inputs, backward=backward)
                else:
                    fn = make_pytorch_fn(block, inputs, backward=backward)
                try:
                    ms = time_median(fn, device=device, warmup=args.warmup, iters=args.iters, leaves=leaves)
                except Exception as exc:  # noqa: BLE001 - one bad row must not kill the sweep
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    row = {"nodes": num_nodes, "edges": num_edges, "backend": name, "mode": mode, "ms": "", "status": f"error: {type(exc).__name__}: {exc}"}
                    rows.append(row)
                    print(f"nodes={num_nodes:>6} edges={num_edges:>7} backend={name:<8} mode={mode:<7} ERROR {type(exc).__name__}: {exc}", flush=True)
                    continue
                finally:
                    if leaves is not None:
                        clear_grads(leaves)
                row = {"nodes": num_nodes, "edges": num_edges, "backend": name, "mode": mode, "ms": f"{ms:.4f}", "status": "ok"}
                rows.append(row)
                print(f"nodes={num_nodes:>6} edges={num_edges:>7} backend={name:<8} mode={mode:<7} median_ms={ms:.4f}", flush=True)
        # per-config speedup summary line
        table = {(r["backend"], r["mode"]): float(r["ms"]) for r in rows if r["nodes"] == num_nodes and r["status"] == "ok"}
        for mode in ("fwd", "fwd+bwd"):
            if ("pytorch", mode) in table and ("cueq", mode) in table:
                print(f"nodes={num_nodes:>6} mode={mode:<7} speedup cueq vs pytorch: {table[('pytorch', mode)] / table[('cueq', mode)]:.3f}x", flush=True)
        del ref, cueq_block, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"conv_tp_backend_bench_{stamp}.csv"
        json_path = out_dir / f"conv_tp_backend_bench_{stamp}.json"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["nodes", "edges", "backend", "mode", "ms", "status"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        meta = {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "channels": int(args.channels),
            "lmax": int(args.lmax),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "dtype": "float32 (TF32 off)",
            "statistic": "median of per-iteration wall time",
        }
        with json_path.open("w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=2)
        print(f"wrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
