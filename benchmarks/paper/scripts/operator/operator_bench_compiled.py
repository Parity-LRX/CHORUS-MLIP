#!/usr/bin/env python
"""Companion pass: time the MACE-ICTC conv tensor product under torch.compile
(the deployed form; the model ships an AOTI/compiled product path), so the operator
comparison against the codegen-fused e3nn/cartnn TensorProducts is fair.

Writes operator_ictd_compiled.csv with the SAME columns as operator_bench.py and an
extra backend label "ictd_compiled". Channels fixed (default 64) to bound runtime.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

import torch

from operator_bench import (
    CSV_COLUMNS, ICTC_URL, ICTC_COMMIT, build_ictd, ictd_make_inputs, ictd_loss,
    cuda_sync, free,
)

SEMEQ = ("operator-level comparable workload: MACE-ICTC conv tensor product (ICTC 2l+1 "
         "basis) under torch.compile (deployed form; eager has Python per-path overhead). "
         "Same (l1,l2,l3) path set as cartnn/e3nn; NOT exact apples-to-apples")


def time_compiled(fwd_c, loss_fn, inp, leaves, mode, warmup, measured, device):
    def zero_grads():
        for t in leaves:
            t.grad = None
    for _ in range(warmup):
        if mode == "forward_only":
            with torch.no_grad():
                _ = fwd_c(inp)
        else:
            zero_grads(); loss_fn(fwd_c(inp)).backward()
    cuda_sync(device)
    fwd, bwd = [], []
    for _ in range(measured):
        if mode == "forward_only":
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                e0.record(); _ = fwd_c(inp); e1.record()
            torch.cuda.synchronize(device); fwd.append(e0.elapsed_time(e1)); bwd.append(0.0)
        else:
            zero_grads()
            ef0 = torch.cuda.Event(enable_timing=True); ef1 = torch.cuda.Event(enable_timing=True)
            eb0 = torch.cuda.Event(enable_timing=True); eb1 = torch.cuda.Event(enable_timing=True)
            ef0.record(); out = fwd_c(inp); ef1.record()
            loss = loss_fn(out)
            eb0.record(); loss.backward(); eb1.record()
            torch.cuda.synchronize(device); fwd.append(ef0.elapsed_time(ef1)); bwd.append(eb0.elapsed_time(eb1))
    return statistics.median(fwd), statistics.median(bwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", default="1:1,1:2,2:2,2:3,3:3")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--edges", default="10000,50000,100000,500000")
    ap.add_argument("--dtypes", default="float32,float64")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measured", type=int, default=50)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]
    edges_list = [int(x) for x in args.edges.split(",")]
    dtypes = [{"float32": torch.float32, "float64": torch.float64}[d] for d in args.dtypes.split(",")]
    C = args.channels

    csv_path = os.path.join(args.out, "operator_ictd_compiled.csv")
    f = open(csv_path, "w", newline=""); w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    w.writeheader(); f.flush()

    def emit(**kw):
        row = {c: kw.get(c, "") for c in CSV_COLUMNS}; w.writerow(row); f.flush()
        print(f"[{row['status']:5s}] ictd_compiled l{row['hidden_lmax']}/{row['max_ell']} "
              f"C{row['channels']} E{row['edges']} {row['dtype']} {row['mode']:16s} "
              f"fwd={row['forward_ms']} bwd={row['backward_ms']} eps={row['edges_per_s']} "
              f"mem={row['peak_mem_gb']} {row['error']}", flush=True)

    for (hidden_lmax, max_ell) in configs:
        target_lmax = hidden_lmax
        for dtype in dtypes:
            dname = "float32" if dtype == torch.float32 else "float64"
            torch.set_default_dtype(dtype)
            try:
                tp, paths = build_ictd(hidden_lmax, max_ell, target_lmax, C, dtype, device)
                npaths = len(paths)
            except Exception as exc:  # noqa
                emit(backend="ictd_compiled", package_url=ICTC_URL, package_commit=ICTC_COMMIT,
                     op_name="ictd_edge_weighted_path_tp_compiled", semantic_equivalence=SEMEQ,
                     hidden_lmax=hidden_lmax, max_ell=max_ell, correlation=2, channels=C, edges="",
                     dtype=dname, mode="build", status="error", error=f"build:{exc}"[:300])
                continue
            base_note = f"paths={npaths}; torch.compile(default); binary product (body order 2)"

            def fwd_eager(inp):
                return tp(inp[0], inp[1], inp[2])
            try:
                fwd_c = torch.compile(fwd_eager, dynamic=False)
            except Exception as exc:  # noqa
                fwd_c = None

            for E in edges_list:
                for mode in ("forward_only", "forward_backward"):
                    rg = (mode == "forward_backward")
                    try:
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        inp, leaves = ictd_make_inputs(tp, hidden_lmax, max_ell, C, E, dtype, device, rg)
                        callable_fwd = fwd_c if fwd_c is not None else fwd_eager
                        note = base_note if fwd_c is not None else base_note + "; COMPILE-FAILED fallback eager"
                        # extra warmups for compile graph capture
                        fwd_ms, bwd_ms = time_compiled(callable_fwd, ictd_loss, inp, leaves, mode,
                                                       args.warmup + 5, args.measured, device)
                        total = fwd_ms + (bwd_ms if rg else 0.0)
                        eps = E / (total / 1e3) if total > 0 else 0.0
                        peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
                        emit(backend="ictd_compiled", package_url=ICTC_URL, package_commit=ICTC_COMMIT,
                             op_name="ictd_edge_weighted_path_tp_compiled", semantic_equivalence=SEMEQ,
                             hidden_lmax=hidden_lmax, max_ell=max_ell, correlation=2, channels=C, edges=E,
                             dtype=dname, mode=mode, warmup=args.warmup + 5, measured=args.measured,
                             forward_ms=round(fwd_ms, 5), backward_ms=round(bwd_ms, 5),
                             total_ms=round(total, 5), edges_per_s=round(eps, 1),
                             peak_mem_gb=round(peak, 4), status="ok", error="", notes=note)
                        del inp, leaves; free()
                    except RuntimeError as exc:
                        msg = str(exc); status = "oom" if "out of memory" in msg.lower() else "error"
                        emit(backend="ictd_compiled", package_url=ICTC_URL, package_commit=ICTC_COMMIT,
                             op_name="ictd_edge_weighted_path_tp_compiled", semantic_equivalence=SEMEQ,
                             hidden_lmax=hidden_lmax, max_ell=max_ell, correlation=2, channels=C, edges=E,
                             dtype=dname, mode=mode, status=status, error=f"{type(exc).__name__}:{msg}"[:300],
                             notes=base_note); free()
                    except Exception as exc:  # noqa
                        emit(backend="ictd_compiled", package_url=ICTC_URL, package_commit=ICTC_COMMIT,
                             op_name="ictd_edge_weighted_path_tp_compiled", semantic_equivalence=SEMEQ,
                             hidden_lmax=hidden_lmax, max_ell=max_ell, correlation=2, channels=C, edges=E,
                             dtype=dname, mode=mode, status="error", error=f"{type(exc).__name__}:{exc}"[:300],
                             notes=base_note); free()
            del tp; free()
    f.close()
    print(f"DONE -> {csv_path}")


if __name__ == "__main__":
    main()
