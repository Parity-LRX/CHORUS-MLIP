#!/usr/bin/env python
"""Plot operator-level throughput + speedup: ICTC vs Cartesian references vs e3nn(reference).

Reads operator_cartnn_vs_ictd.csv (written by operator_bench.py) and produces
  figures/operator_throughput.png   (edges/s vs directed edges)
  figures/operator_speedup.png      (ICTC- and e3nn-vs-cartnn speedup)

Grid: rows = (hidden_lmax/max_ell) configs, cols = {fp32,fp64} x {forward_only,forward_backward}.
Fixed channels (default 64). forward-only and forward+backward are kept in separate columns,
fp32/fp64 in separate columns, per the benchmark spec. OOM/error rows are dropped from the
curves (they are reported separately in the summary).
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKENDS = ["e3nn", "cartnn", "ictd", "cace_sym", "cart3l"]
COLORS = {
    "e3nn": "#888888",
    "cartnn": "#d62728",
    "ictd": "#1f77b4",
    "cace_sym": "#2ca02c",
    "cart3l": "#9467bd",
}
MARKERS = {"e3nn": "s", "cartnn": "o", "ictd": "^", "cace_sym": "D", "cart3l": "v"}
LABELS = {
    "e3nn": "e3nn (spherical, ref)",
    "cartnn": "cartnn (Cartesian 3^l)",
    "ictd": "MACE-ICTC",
    "cace_sym": "CACE-style sym. monomial",
    "cart3l": "native dense 3^l diagnostic",
}


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fnum(x, d=None):
    try:
        return float(x)
    except Exception:
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--channels", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load(args.csv)
    present_backends = [be for be in BACKENDS if any(r["backend"] == be for r in rows)]
    # config order as they appear
    configs = []
    for r in rows:
        key = (int(r["hidden_lmax"]), int(r["max_ell"]))
        if key not in configs:
            configs.append(key)
    configs.sort()

    col_specs = [("float32", "forward_only"), ("float32", "forward_backward"),
                 ("float64", "forward_only"), ("float64", "forward_backward")]
    col_titles = ["fp32 fwd-only", "fp32 fwd+bwd", "fp64 fwd-only", "fp64 fwd+bwd"]

    # index ok rows: (config, dtype, mode, backend) -> {edges: total_ms, edges_per_s}
    idx = defaultdict(dict)
    for r in rows:
        if r["status"] != "ok" or int(r["channels"]) != args.channels:
            continue
        key = ((int(r["hidden_lmax"]), int(r["max_ell"])), r["dtype"], r["mode"], r["backend"])
        idx[key][int(r["edges"])] = dict(total_ms=fnum(r["total_ms"]), eps=fnum(r["edges_per_s"]))

    # ---------------- throughput ----------------
    nrow, ncol = len(configs), len(col_specs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    for ri, cfg in enumerate(configs):
        for ci, (dt, mode) in enumerate(col_specs):
            ax = axes[ri][ci]
            any_data = False
            for be in present_backends:
                d = idx.get((cfg, dt, mode, be), {})
                if not d:
                    continue
                xs = sorted(d)
                ys = [d[x]["eps"] for x in xs]
                ax.plot(xs, ys, marker=MARKERS[be], color=COLORS[be], label=LABELS[be], lw=1.8, ms=5)
                any_data = True
            ax.set_xscale("log"); ax.set_yscale("log")
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"l{cfg[0]}/{cfg[1]}\nedges/s", fontsize=10)
            if ri == nrow - 1:
                ax.set_xlabel("directed edges", fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
            if ri == 0 and ci == 0 and any_data:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Operator throughput (channels={args.channels}): MACE-ICTC vs Cartesian references vs e3nn",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    p1 = os.path.join(args.outdir, "operator_throughput.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p1)

    # ---------------- speedup vs cartnn ----------------
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    for ri, cfg in enumerate(configs):
        for ci, (dt, mode) in enumerate(col_specs):
            ax = axes[ri][ci]
            cart = idx.get((cfg, dt, mode, "cartnn"), {})
            any_speedup = False
            for be in ("ictd", "e3nn", "cart3l"):
                d = idx.get((cfg, dt, mode, be), {})
                xs = sorted(set(d) & set(cart))
                if not xs:
                    continue
                # speedup of `be` relative to cartnn: cartnn_total / be_total  (>1 => be faster)
                ys = [cart[x]["total_ms"] / d[x]["total_ms"] for x in xs if d[x]["total_ms"]]
                xs = [x for x in xs if d[x]["total_ms"]]
                ax.plot(xs, ys, marker=MARKERS[be], color=COLORS[be],
                        label=f"{LABELS[be]} / cartnn", lw=1.8, ms=5)
                any_speedup = True
            ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
            ax.set_xscale("log")
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"l{cfg[0]}/{cfg[1]}\nspeedup vs cartnn", fontsize=10)
            if ri == nrow - 1:
                ax.set_xlabel("directed edges", fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
            if ri == 0 and ci == 0 and any_speedup:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Speedup vs cartnn (channels={args.channels}); >1 means faster than cartnn",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    p2 = os.path.join(args.outdir, "operator_speedup.png")
    fig.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
