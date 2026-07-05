#!/usr/bin/env python
"""Flat-wrapper torch.compile forward+backward operator benchmark.

This is the training-mode companion to operator_bench_aoti.py.  It reuses the
same flat-I/O wrapper used for the matched-fusion forward-only table, but times
forward+backward instead of inference-only AOTI.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

import torch
import torch._dynamo
from cartnn import o3 as co3
from e3nn import o3 as e3o3

from operator_bench import (
    CARTNN_COMMIT,
    CARTNN_URL,
    CSV_COLUMNS,
    E3NN_COMMIT,
    E3NN_URL,
    ICTC_COMMIT,
    ICTC_URL,
    build_eo3,
    build_ictd,
    cuda_sync,
    eo3_forward,
    eo3_loss,
    eo3_make_inputs,
    free,
)
from operator_bench_aoti import FlatICTCTP, ictd_flat_inputs


torch._dynamo.config.cache_size_limit = 256


META = {
    "ictd_compile_fwbw": (
        ICTC_URL,
        ICTC_COMMIT,
        "ictd_tp_torchcompile_flat_fwbw",
        "ICTC flat-wrapper torch.compile tensor product, forward+backward",
    ),
    "e3nn": (
        E3NN_URL,
        E3NN_COMMIT,
        "e3nn_tp_fused_fwbw",
        "e3nn codegen-fused tensor product, forward+backward reference",
    ),
    "cartnn": (
        CARTNN_URL,
        CARTNN_COMMIT,
        "cartnn_tp_fused_fwbw",
        "cartnn codegen-fused tensor product, forward+backward reference",
    ),
}


def zero_grads(leaves):
    for tensor in leaves:
        tensor.grad = None


def flat_loss(out):
    return out.pow(2).sum()


def time_fwbw(call, loss_fn, inputs, leaves, warmup, measured, device):
    for _ in range(warmup):
        zero_grads(leaves)
        loss_fn(call(*inputs)).backward()
    cuda_sync(device)

    fwd_ms, bwd_ms = [], []
    for _ in range(measured):
        zero_grads(leaves)
        ef0 = torch.cuda.Event(enable_timing=True)
        ef1 = torch.cuda.Event(enable_timing=True)
        eb0 = torch.cuda.Event(enable_timing=True)
        eb1 = torch.cuda.Event(enable_timing=True)
        ef0.record()
        out = call(*inputs)
        ef1.record()
        loss = loss_fn(out)
        eb0.record()
        loss.backward()
        eb1.record()
        torch.cuda.synchronize(device)
        fwd_ms.append(ef0.elapsed_time(ef1))
        bwd_ms.append(eb0.elapsed_time(eb1))
    return statistics.median(fwd_ms), statistics.median(bwd_ms)


def make_ictd_inputs(tp, hidden_lmax, max_ell, channels, edges, dtype, device):
    x1, edge, gates = ictd_flat_inputs(tp, hidden_lmax, max_ell, channels, edges, dtype, device)
    x1.requires_grad_(True)
    gates.requires_grad_(True)
    return (x1, edge, gates), [x1, gates]


def emit(writer, backend, hidden_lmax, max_ell, channels, edges, dtype_name, warmup, measured,
         status, forward_ms=None, backward_ms=None, peak_mem_gb=None, error="", notes=""):
    url, commit, op_name, semeq = META[backend]
    row = {col: "" for col in CSV_COLUMNS}
    row.update(
        backend=backend,
        package_url=url,
        package_commit=commit,
        op_name=op_name,
        semantic_equivalence=semeq,
        hidden_lmax=hidden_lmax,
        max_ell=max_ell,
        correlation=2,
        channels=channels,
        edges=edges,
        dtype=dtype_name,
        mode="forward_backward",
        warmup=warmup,
        measured=measured,
        status=status,
        error=error[:300],
        notes=notes,
    )
    if forward_ms is not None and backward_ms is not None:
        total_ms = forward_ms + backward_ms
        row.update(
            forward_ms=round(forward_ms, 5),
            backward_ms=round(backward_ms, 5),
            total_ms=round(total_ms, 5),
            edges_per_s=round(edges / (total_ms / 1e3), 1) if total_ms > 0 else 0,
        )
    if peak_mem_gb is not None:
        row["peak_mem_gb"] = round(peak_mem_gb, 4)
    writer.writerow(row)
    print(
        f"[{status:5s}] {backend:18s} l{hidden_lmax}/{max_ell} C{channels} E{edges} "
        f"{dtype_name} fwd={row['forward_ms']} bwd={row['backward_ms']} "
        f"total={row['total_ms']} mem={row['peak_mem_gb']} {row['error']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--configs", default="1:1,1:2,2:2,2:3,3:3")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--edges", type=int, default=100000)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    torch.set_default_dtype(dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configs = [tuple(int(part) for part in cfg.split(":")) for cfg in args.configs.split(",")]

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "operator_compile_fwbw_flat.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for hidden_lmax, max_ell in configs:
            target_lmax = hidden_lmax
            torch._dynamo.reset()
            try:
                tp, paths = build_ictd(hidden_lmax, max_ell, target_lmax, args.channels, dtype, device)
                flat = FlatICTCTP(tp, hidden_lmax, max_ell).to(device).train()
                with torch.no_grad():
                    flat(*ictd_flat_inputs(tp, hidden_lmax, max_ell, args.channels, 64, dtype, device))
                cuda_sync(device)
                compiled = torch.compile(flat, dynamic=False, mode="max-autotune-no-cudagraphs")
                inputs, leaves = make_ictd_inputs(
                    tp, hidden_lmax, max_ell, args.channels, args.edges, dtype, device
                )
                torch.cuda.reset_peak_memory_stats(device)
                fwd, bwd = time_fwbw(
                    compiled, flat_loss, inputs, leaves, args.warmup + 10, args.measured, device
                )
                emit(
                    writer,
                    "ictd_compile_fwbw",
                    hidden_lmax,
                    max_ell,
                    args.channels,
                    args.edges,
                    args.dtype,
                    args.warmup + 10,
                    args.measured,
                    "ok",
                    fwd,
                    bwd,
                    torch.cuda.max_memory_allocated(device) / 1e9,
                    notes=f"paths={len(paths)}; flat wrapper; torch.compile(max-autotune-no-cudagraphs)",
                )
                del inputs, leaves, compiled, flat, tp
                free()
            except RuntimeError as exc:
                status = "oom" if "out of memory" in str(exc).lower() else "error"
                emit(
                    writer,
                    "ictd_compile_fwbw",
                    hidden_lmax,
                    max_ell,
                    args.channels,
                    args.edges,
                    args.dtype,
                    args.warmup + 10,
                    0,
                    status,
                    error=f"{type(exc).__name__}:{exc}",
                )
                free()
            except Exception as exc:
                emit(
                    writer,
                    "ictd_compile_fwbw",
                    hidden_lmax,
                    max_ell,
                    args.channels,
                    args.edges,
                    args.dtype,
                    args.warmup + 10,
                    0,
                    "error",
                    error=f"{type(exc).__name__}:{exc}",
                )
                free()

            for backend, module in (("e3nn", e3o3), ("cartnn", co3)):
                try:
                    tp, paths, in1, in2, _ = build_eo3(
                        module, hidden_lmax, max_ell, target_lmax, args.channels, dtype, device
                    )
                    inputs, leaves = eo3_make_inputs(tp, in1, in2, args.edges, dtype, device, True)
                    torch.cuda.reset_peak_memory_stats(device)
                    fwd, bwd = time_fwbw(
                        lambda x1, x2, w: eo3_forward(tp, (x1, x2, w)),
                        eo3_loss,
                        inputs,
                        leaves,
                        args.warmup,
                        args.measured,
                        device,
                    )
                    emit(
                        writer,
                        backend,
                        hidden_lmax,
                        max_ell,
                        args.channels,
                        args.edges,
                        args.dtype,
                        args.warmup,
                        args.measured,
                        "ok",
                        fwd,
                        bwd,
                        torch.cuda.max_memory_allocated(device) / 1e9,
                        notes=f"paths={len(paths)}; codegen-fused tensor product",
                    )
                    del inputs, leaves, tp
                    free()
                except RuntimeError as exc:
                    status = "oom" if "out of memory" in str(exc).lower() else "error"
                    emit(
                        writer,
                        backend,
                        hidden_lmax,
                        max_ell,
                        args.channels,
                        args.edges,
                        args.dtype,
                        args.warmup,
                        0,
                        status,
                        error=f"{type(exc).__name__}:{exc}",
                    )
                    free()
                except Exception as exc:
                    emit(
                        writer,
                        backend,
                        hidden_lmax,
                        max_ell,
                        args.channels,
                        args.edges,
                        args.dtype,
                        args.warmup,
                        0,
                        "error",
                        error=f"{type(exc).__name__}:{exc}",
                    )
                    free()

    print(f"DONE -> {csv_path}")


if __name__ == "__main__":
    main()
