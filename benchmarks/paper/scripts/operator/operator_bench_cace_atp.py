#!/usr/bin/env python
"""Benchmark CACE's native AngularTensorProduct workload.

This script intentionally does NOT treat CACE as a drop-in MACE convolution
tensor-product baseline. CACE's AngularTensorProduct multiplies symmetric
Cartesian monomials and accumulates them in the truncated monomial basis; its
native output degree is l1+l2, not a CG path (l1,l2)->l3. The measurements are
therefore a reference Cartesian product workload, separate from the matched
e3nn/cartnn/ICTC tensor-product benchmark.

Usage examples:

  PYTHONPATH=/path/to/cace python benchmarks/paper/scripts/operator/operator_bench_cace_atp.py --out /tmp/cace_atp
  python benchmarks/paper/scripts/operator/operator_bench_cace_atp.py --cace-root /path/to/cace --out /tmp/cace_atp
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import sys
import time
from pathlib import Path

import torch


CSV_COLUMNS = [
    "backend",
    "package_url",
    "package_commit",
    "op_name",
    "semantic_equivalence",
    "max_l",
    "n_angular",
    "n_indices",
    "radial",
    "channels",
    "edges",
    "dtype",
    "mode",
    "warmup",
    "measured",
    "forward_ms",
    "backward_ms",
    "total_ms",
    "edges_per_s",
    "peak_mem_gb",
    "status",
    "error",
    "notes",
]

CACE_URL = "https://github.com/BingqingCheng/cace"
CACE_SEMANTICS = (
    "CACE native AngularTensorProduct: symmetric Cartesian monomial product in "
    "the truncated monomial basis; native output degree l1+l2, not a matched "
    "CG tensor-product path (l1,l2)->l3"
)


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def time_cace(op, x1, x2, mode: str, warmup: int, measured: int, device: torch.device):
    requires_grad = mode == "forward_backward"

    def zero_grads() -> None:
        if requires_grad:
            x1.grad = None
            x2.grad = None

    def forward_loss():
        out = op(x1, x2)
        return out, out.pow(2).sum()

    for _ in range(warmup):
        if mode == "forward_only":
            with torch.no_grad():
                _ = op(x1, x2)
        else:
            zero_grads()
            _, loss = forward_loss()
            loss.backward()
    cuda_sync(device)

    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    use_events = device.type == "cuda"
    for _ in range(measured):
        if mode == "forward_only":
            if use_events:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                with torch.no_grad():
                    e0.record()
                    _ = op(x1, x2)
                    e1.record()
                torch.cuda.synchronize(device)
                fwd_ms.append(e0.elapsed_time(e1))
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = op(x1, x2)
                fwd_ms.append((time.perf_counter() - t0) * 1e3)
            bwd_ms.append(0.0)
        else:
            zero_grads()
            if use_events:
                ef0 = torch.cuda.Event(enable_timing=True)
                ef1 = torch.cuda.Event(enable_timing=True)
                eb0 = torch.cuda.Event(enable_timing=True)
                eb1 = torch.cuda.Event(enable_timing=True)
                ef0.record()
                out = op(x1, x2)
                ef1.record()
                loss = out.pow(2).sum()
                eb0.record()
                loss.backward()
                eb1.record()
                torch.cuda.synchronize(device)
                fwd_ms.append(ef0.elapsed_time(ef1))
                bwd_ms.append(eb0.elapsed_time(eb1))
            else:
                t0 = time.perf_counter()
                out = op(x1, x2)
                t1 = time.perf_counter()
                out.pow(2).sum().backward()
                t2 = time.perf_counter()
                fwd_ms.append((t1 - t0) * 1e3)
                bwd_ms.append((t2 - t1) * 1e3)
    return statistics.median(fwd_ms), statistics.median(bwd_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cace-root", default="", help="Path to a CACE source checkout to add to PYTHONPATH")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-ls", default="1,2,3")
    parser.add_argument("--edges", type=int, default=100000)
    parser.add_argument("--radial", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=10)
    parser.add_argument("--slow-measured", type=int, default=5)
    args = parser.parse_args()

    if args.cace_root:
        sys.path.insert(0, args.cace_root)

    try:
        from cace.modules.angular import compute_length_lmax, make_lxlylz_list
        from cace.modules.product_basis import AngularTensorProduct
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Could not import CACE. Install CACE or pass --cace-root /path/to/cace. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "operator_cace_atp.csv")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    package_commit = ""
    if args.cace_root:
        git_dir = Path(args.cace_root)
        head = git_dir / ".git" / "HEAD"
        if head.exists():
            package_commit = "local checkout"

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        def emit(**kwargs) -> None:
            row = {col: kwargs.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(row)
            handle.flush()
            print(
                f"[{row['status']:5s}] cace_atp L{row['max_l']} "
                f"rad{row['radial']} C{row['channels']} E{row['edges']} {row['mode']:16s} "
                f"total={row['total_ms']} fwd={row['forward_ms']} bwd={row['backward_ms']} "
                f"mem={row['peak_mem_gb']} {row['error']}",
                flush=True,
            )

        for max_l in parse_ints(args.max_ls):
            try:
                l_list = make_lxlylz_list(max_l)
                op = AngularTensorProduct(max_l, l_list).to(device)
                n_angular = compute_length_lmax(max_l)
                n_indices = len(getattr(op, "indice_list", []))
            except Exception as exc:  # noqa: BLE001
                emit(
                    backend="cace_atp",
                    package_url=CACE_URL,
                    package_commit=package_commit,
                    op_name="cace.modules.product_basis.AngularTensorProduct",
                    semantic_equivalence=CACE_SEMANTICS,
                    max_l=max_l,
                    radial=args.radial,
                    channels=args.channels,
                    edges=args.edges,
                    dtype=args.dtype,
                    mode="build",
                    warmup=args.warmup,
                    measured=0,
                    status="error",
                    error=f"{type(exc).__name__}:{exc}"[:300],
                )
                continue

            for mode in ("forward_only", "forward_backward"):
                requires_grad = mode == "forward_backward"
                measured = args.measured if max_l < 3 else min(args.measured, args.slow_measured)
                try:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    x1 = torch.randn(
                        args.edges,
                        args.radial,
                        n_angular,
                        args.channels,
                        device=device,
                        dtype=dtype,
                        requires_grad=requires_grad,
                    )
                    x2 = torch.randn(
                        args.edges,
                        args.radial,
                        n_angular,
                        args.channels,
                        device=device,
                        dtype=dtype,
                        requires_grad=requires_grad,
                    )
                    fwd, bwd = time_cace(op, x1, x2, mode, args.warmup, measured, device)
                    total = fwd + (bwd if requires_grad else 0.0)
                    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
                    emit(
                        backend="cace_atp",
                        package_url=CACE_URL,
                        package_commit=package_commit,
                        op_name="cace.modules.product_basis.AngularTensorProduct",
                        semantic_equivalence=CACE_SEMANTICS,
                        max_l=max_l,
                        n_angular=n_angular,
                        n_indices=n_indices,
                        radial=args.radial,
                        channels=args.channels,
                        edges=args.edges,
                        dtype=args.dtype,
                        mode=mode,
                        warmup=args.warmup,
                        measured=measured,
                        forward_ms=round(fwd, 5),
                        backward_ms=round(bwd, 5),
                        total_ms=round(total, 5),
                        edges_per_s=round(args.edges / (total / 1e3), 1) if total > 0 else 0,
                        peak_mem_gb=round(peak, 4),
                        status="ok",
                        error="",
                        notes="CACE native monomial product workload; not matched MACE CG TP",
                    )
                    del x1, x2
                    free()
                except RuntimeError as exc:
                    status = "oom" if "out of memory" in str(exc).lower() else "error"
                    emit(
                        backend="cace_atp",
                        package_url=CACE_URL,
                        package_commit=package_commit,
                        op_name="cace.modules.product_basis.AngularTensorProduct",
                        semantic_equivalence=CACE_SEMANTICS,
                        max_l=max_l,
                        n_angular=n_angular,
                        n_indices=n_indices,
                        radial=args.radial,
                        channels=args.channels,
                        edges=args.edges,
                        dtype=args.dtype,
                        mode=mode,
                        warmup=args.warmup,
                        measured=0,
                        status=status,
                        error=f"{type(exc).__name__}:{exc}"[:300],
                        notes="CACE native monomial product workload; not matched MACE CG TP",
                    )
                    free()

    print(f"DONE -> {csv_path}")


if __name__ == "__main__":
    main()
