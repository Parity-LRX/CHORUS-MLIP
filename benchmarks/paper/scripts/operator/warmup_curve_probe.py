#!/usr/bin/env python
"""Verify warmup sufficiency: time EACH of the first N forward calls for ictd / cartnn / e3nn,
so we can SEE where the (device,dtype) cache populates and where the curve plateaus.
Confirms whether warmup=20 in operator_bench.py lands on the warm plateau.
"""
from __future__ import annotations
import argparse, statistics, torch
from operator_bench import (build_ictd, ictd_make_inputs, ictd_forward,
                             build_eo3, eo3_make_inputs, eo3_forward, cuda_sync)
from e3nn import o3 as e3o3
from cartnn import o3 as co3


def curve(forward_fn, inp, n, device):
    ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            e0.record(); _ = forward_fn(inp); e1.record()
        torch.cuda.synchronize(device); ts.append(e0.elapsed_time(e1))
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="2:2")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--edges", type=int, default=100000)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    hidden_lmax, max_ell = (int(x) for x in args.config.split(":"))
    target = hidden_lmax
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    torch.set_default_dtype(dtype)
    C, E, N = args.channels, args.edges, args.n
    print(f"# config l{hidden_lmax}/{max_ell} C{C} E{E} {args.dtype}  (per-call forward ms)")

    # ICTC (fresh instance -> cold cache)
    tp, _ = build_ictd(hidden_lmax, max_ell, target, C, dtype, device)
    inp, _ = ictd_make_inputs(tp, hidden_lmax, max_ell, C, E, dtype, device, False)
    cuda_sync(device)
    ti = curve(lambda i: ictd_forward(tp, i), inp, N, device)

    # cartnn
    tpc, _, in1c, in2c, _ = build_eo3(co3, hidden_lmax, max_ell, target, C, dtype, device)
    inpc, _ = eo3_make_inputs(tpc, in1c, in2c, E, dtype, device, False)
    cuda_sync(device)
    tc = curve(lambda i: eo3_forward(tpc, i), inpc, N, device)

    # e3nn
    tpe, _, in1e, in2e, _ = build_eo3(e3o3, hidden_lmax, max_ell, target, C, dtype, device)
    inpe, _ = eo3_make_inputs(tpe, in1e, in2e, E, dtype, device, False)
    cuda_sync(device)
    te = curve(lambda i: eo3_forward(tpe, i), inpe, N, device)

    def fmt(ts):
        return " ".join(f"{t:6.2f}" for t in ts)
    print("call#:   ", " ".join(f"{i+1:6d}" for i in range(N)))
    print("ictd:    ", fmt(ti))
    print("cartnn:  ", fmt(tc))
    print("e3nn:    ", fmt(te))
    for name, ts in (("ictd", ti), ("cartnn", tc), ("e3nn", te)):
        first = ts[0]
        warm_21_40 = statistics.median(ts[20:]) if len(ts) > 20 else float("nan")
        warm_6_40 = statistics.median(ts[5:]) if len(ts) > 5 else float("nan")
        print(f"# {name:7s} call1={first:.2f}  median[6:]={warm_6_40:.3f}  "
              f"median[21:]={warm_21_40:.3f}  ratio call1/warm={first/warm_21_40:.1f}x  "
              f"plateau_drift[6:]vs[21:]={warm_6_40/warm_21_40:.3f}")


if __name__ == "__main__":
    main()
