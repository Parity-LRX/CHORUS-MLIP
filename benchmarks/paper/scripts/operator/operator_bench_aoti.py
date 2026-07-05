#!/usr/bin/env python
"""Matched-fusion operator comparison (forward-only / inference).

The earlier operator bench timed the ICTC conv-TP EAGER (torch.compile graph-broke on its dict I/O)
against e3nn/cartnn which are opt_einsum_fx codegen-FUSED -> unfair to ICTC. Here the ICTC operator is
put at the SAME fusion level the model deploys: torch.compile and AOTInductor of a thin flat-I/O wrapper
around the *unmodified* repo tp. Timed vs e3nn / cartnn fused + eager-ICTC ref. Forward-only (AOTI is
inference-only).  No repo source is modified.

Ordering matters: the repo tp lazily builds a (device,dtype) CG/projector cache on first forward. We WARM
it with a real forward first, run eager/compile/e3nn/cartnn, and do the torch.export+AOTI LAST (a cold
export traces _cg_for under fake tensors and corrupts the cache).
"""
from __future__ import annotations

import argparse, csv, os, statistics, traceback
import torch
from torch.export import Dim

from operator_bench import (
    build_ictd, build_eo3, eo3_make_inputs, eo3_forward, cuda_sync, free,
    ICTC_URL, ICTC_COMMIT, CARTNN_URL, CARTNN_COMMIT, E3NN_URL, E3NN_COMMIT, CSV_COLUMNS,
)
from e3nn import o3 as e3o3
from cartnn import o3 as co3
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256  # avoid cross-config recompile-limit fallback to eager


class FlatICTCTP(torch.nn.Module):
    """flat tensors <-> repo EdgeWeightedPathPreservingTensorProduct dict I/O (export-friendly)."""
    def __init__(self, tp, hidden_lmax: int, max_ell: int):
        super().__init__()
        self.tp = tp
        self.in1_ls = list(range(int(hidden_lmax) + 1))
        self.in2_ls = list(range(int(max_ell) + 1))

    def forward(self, x1_flat, edge_flat, gates):
        x1 = {}; off = 0
        for l in self.in1_ls:
            w = 2 * l + 1; x1[l] = x1_flat[:, :, off:off + w]; off += w
        edge = {}; off = 0
        for l in self.in2_ls:
            w = 2 * l + 1; edge[l] = edge_flat[:, :, off:off + w]; off += w
        out = self.tp(x1, edge, gates)
        return torch.cat([out[l].reshape(out[l].shape[0], -1) for l in sorted(out.keys())], dim=-1)


def ictd_flat_inputs(tp, hl, me, C, E, dtype, device):
    din1 = sum(2 * l + 1 for l in range(hl + 1))
    din2 = sum(2 * l + 1 for l in range(me + 1))
    return (torch.randn(E, C, din1, device=device, dtype=dtype),
            torch.randn(E, 1, din2, device=device, dtype=dtype),
            torch.randn(E, tp.num_paths * C, device=device, dtype=dtype))


def time_fwd(call, inp, warmup, measured, device):
    for _ in range(warmup):
        with torch.no_grad():
            _ = call(*inp)
    cuda_sync(device)
    ts = []
    for _ in range(measured):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            e0.record(); _ = call(*inp); e1.record()
        torch.cuda.synchronize(device); ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


META = {
    "ictd_eager": (ICTC_URL, ICTC_COMMIT, "ictd_tp_eager", "ICTC eager (ref)"),
    "ictd_compile": (ICTC_URL, ICTC_COMMIT, "ictd_tp_torchcompile", "ICTC torch.compile(flat wrapper)"),
    "ictd_aoti": (ICTC_URL, ICTC_COMMIT, "ictd_tp_aoti", "ICTC AOTInductor (flat wrapper) = deployment fusion level"),
    "e3nn": (E3NN_URL, E3NN_COMMIT, "e3nn_tp_fused", "e3nn codegen-fused TP (ref)"),
    "cartnn": (CARTNN_URL, CARTNN_COMMIT, "cartnn_tp_fused", "cartnn codegen-fused TP"),
}


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
    ap.add_argument("--no-aoti", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]
    edges_list = [int(x) for x in args.edges.split(",")]
    dtypes = [{"float32": torch.float32, "float64": torch.float64}[d] for d in args.dtypes.split(",")]
    C = args.channels

    f = open(os.path.join(args.out, "operator_aoti_fwd.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_COLUMNS); w.writeheader(); f.flush()

    def emit(be, hl, me, dn, E, status, ms=None, peak=None, err="", note="", warmup=0, measured=0):
        url, commit, op, sem = META[be]
        row = {c: "" for c in CSV_COLUMNS}
        row.update(backend=be, package_url=url, package_commit=commit, op_name=op,
                   semantic_equivalence=sem, hidden_lmax=hl, max_ell=me, correlation=2, channels=C,
                   edges=E, dtype=dn, mode="forward_only", warmup=warmup, measured=measured, status=status,
                   error=err[:250], notes=note)
        if ms is not None:
            row.update(forward_ms=round(ms, 5), backward_ms=0, total_ms=round(ms, 5),
                       edges_per_s=round(E / (ms / 1e3), 1) if ms else 0)
        if peak is not None:
            row["peak_mem_gb"] = round(peak, 4)
        w.writerow(row); f.flush()
        print(f"[{status:5s}] {be:13s} l{hl}/{me} C{C} E{E} {dn:7s} "
              f"fwd={row['forward_ms']} eps={row['edges_per_s']} mem={row['peak_mem_gb']} {err[:70]}", flush=True)

    for (hl, me) in configs:
        target = hl
        for dtype in dtypes:
            dn = "float32" if dtype == torch.float32 else "float64"
            torch.set_default_dtype(dtype)
            torch._dynamo.reset()  # fresh compile state per (config,dtype): no eager-fallback taint
            try:
                tp, paths = build_ictd(hl, me, target, C, dtype, device)
                flat = FlatICTCTP(tp, hl, me).to(device).eval()
                with torch.no_grad():  # WARM the lazy (device,dtype) cache with real tensors
                    _ = flat(*ictd_flat_inputs(tp, hl, me, C, 64, dtype, device))
                cuda_sync(device)
            except Exception as e:
                emit("ictd_eager", hl, me, dn, "", "error", err=f"build:{e}"); continue
            note = f"paths={len(paths)}; forward-only"

            eo = {}
            for be, mod in (("e3nn", e3o3), ("cartnn", co3)):
                try:
                    tpx, _, in1, in2, _o = build_eo3(mod, hl, me, target, C, dtype, device)
                    eo[be] = (tpx, in1, in2)
                except Exception as e:
                    emit(be, hl, me, dn, "", "error", err=f"build:{e}", note=note)
            try:
                tc = torch.compile(flat, dynamic=False, mode="max-autotune-no-cudagraphs")
            except Exception:
                tc = None

            # ---- phase A: eager / compile / e3nn / cartnn (cache stays warm-real) ----
            for E in edges_list:
                inp = ictd_flat_inputs(tp, hl, me, C, E, dtype, device)
                for be, call in (("ictd_eager", flat), ("ictd_compile", tc)):
                    if call is None:
                        emit(be, hl, me, dn, E, "error", err="torch.compile unavailable", note=note); continue
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                        wu = args.warmup + (10 if be == "ictd_compile" else 0)
                        ms = time_fwd(call, inp, wu, args.measured, device)
                        emit(be, hl, me, dn, E, "ok", ms=ms,
                             peak=torch.cuda.max_memory_allocated(device) / 1e9, note=note, warmup=wu, measured=args.measured)
                    except RuntimeError as e:
                        emit(be, hl, me, dn, E, "oom" if "out of memory" in str(e).lower() else "error",
                             err=f"{type(e).__name__}:{e}", note=note); free()
                for be in ("e3nn", "cartnn"):
                    if be not in eo: continue
                    tpx, in1, in2 = eo[be]
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                        (xi, xj, ww), _ = eo3_make_inputs(tpx, in1, in2, E, dtype, device, False)
                        ms = time_fwd(lambda a, b, c: eo3_forward(tpx, (a, b, c)), (xi, xj, ww),
                                      args.warmup, args.measured, device)
                        emit(be, hl, me, dn, E, "ok", ms=ms,
                             peak=torch.cuda.max_memory_allocated(device) / 1e9, note=note, warmup=args.warmup, measured=args.measured)
                    except RuntimeError as e:
                        emit(be, hl, me, dn, E, "oom" if "out of memory" in str(e).lower() else "error",
                             err=f"{type(e).__name__}:{e}", note=note); free()
                del inp; free()

            # ---- phase B: AOTI export (LAST; cache is warm so export hits it) ----
            if not args.no_aoti:
                runner = None; aerr = ""
                try:
                    x1, ed, gt = ictd_flat_inputs(tp, hl, me, C, 64, dtype, device)
                    bd = Dim("E", min=2, max=max(edges_list) * 4)
                    ep = torch.export.export(flat, (x1, ed, gt),
                                             dynamic_shapes=({0: bd}, {0: bd}, {0: bd}), strict=False)
                    from torch._inductor import aoti_compile_and_package, aoti_load_package
                    pkg = os.path.join(args.out, f"ictd_op_l{hl}{me}_{dn}.pt2")
                    if os.path.exists(pkg): os.remove(pkg)
                    aoti_compile_and_package(ep, package_path=pkg)
                    runner = aoti_load_package(pkg)
                except Exception as e:
                    aerr = f"{type(e).__name__}:{e}"; traceback.print_exc()
                for E in edges_list:
                    if runner is None:
                        emit("ictd_aoti", hl, me, dn, E, "error", err=aerr or "aoti export failed", note=note); continue
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                        inp = ictd_flat_inputs(tp, hl, me, C, E, dtype, device)
                        ms = time_fwd(lambda a, b, c: runner(a, b, c), inp, args.warmup, args.measured, device)
                        emit("ictd_aoti", hl, me, dn, E, "ok", ms=ms,
                             peak=torch.cuda.max_memory_allocated(device) / 1e9, note=note, warmup=args.warmup, measured=args.measured)
                        del inp; free()
                    except RuntimeError as e:
                        emit("ictd_aoti", hl, me, dn, E, "oom" if "out of memory" in str(e).lower() else "error",
                             err=f"{type(e).__name__}:{e}", note=note); free()
            del flat, tp, eo; free()
    f.close()
    print("DONE -> operator_aoti_fwd.csv")


if __name__ == "__main__":
    main()
