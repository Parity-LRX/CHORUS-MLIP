#!/usr/bin/env python
"""Plot the matched-fusion (forward-only) operator comparison: eager vs fused ICTC vs e3nn/cartnn.
Shows that the ICTC operator's standing vs e3nn flips once it is fused (torch.compile/AOTI).
figures/operator_matched_fusion.png
"""
import argparse, csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["ictd_eager", "ictd_compile", "ictd_aoti", "e3nn", "cartnn"]
COLOR = {"ictd_eager": "#9ecae1", "ictd_compile": "#1f77b4", "ictd_aoti": "#08306b",
         "e3nn": "#888888", "cartnn": "#d62728"}
LABEL = {"ictd_eager": "ICTC eager", "ictd_compile": "ICTC torch.compile", "ictd_aoti": "ICTC AOTI",
         "e3nn": "e3nn (fused ref)", "cartnn": "cartnn (fused)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--edges", type=int, default=100000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = [r for r in csv.DictReader(open(args.csv)) if r["status"] == "ok"]
    configs = []
    for r in rows:
        c = (int(r["hidden_lmax"]), int(r["max_ell"]))
        if c not in configs:
            configs.append(c)
    configs.sort()
    val = defaultdict(dict)  # (dtype, cfg) -> backend -> ms
    for r in rows:
        if int(r["edges"]) != args.edges:
            continue
        val[(r["dtype"], (int(r["hidden_lmax"]), int(r["max_ell"])))][r["backend"]] = float(r["total_ms"])

    dts = ["float32", "float64"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    xs = range(len(configs))
    xlabels = [f"l{a}/{b}" for a, b in configs]
    for j, dt in enumerate(dts):
        # absolute ms (log)
        ax = axes[0][j]
        nb = len(ORDER); w = 0.16
        for bi, be in enumerate(ORDER):
            ys = [val[(dt, c)].get(be) for c in configs]
            xpos = [x + (bi - nb / 2) * w + w / 2 for x in xs]
            ax.bar([xp for xp, y in zip(xpos, ys) if y], [y for y in ys if y],
                   width=w, color=COLOR[be], label=LABEL[be])
        ax.set_yscale("log"); ax.set_xticks(list(xs)); ax.set_xticklabels(xlabels)
        ax.set_title(f"{dt} forward-only, E={args.edges}: forward ms (log)")
        ax.set_ylabel("forward ms"); ax.grid(True, axis="y", alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, ncol=2)
        # speedup vs e3nn (e3nn_time / backend_time; >1 = faster than e3nn)
        ax2 = axes[1][j]
        for be in ("ictd_eager", "ictd_compile", "ictd_aoti", "cartnn"):
            ys = []
            for c in configs:
                e = val[(dt, c)].get("e3nn"); v = val[(dt, c)].get(be)
                ys.append(e / v if (e and v) else None)
            xp = [x for x, y in zip(xs, ys) if y]
            yp = [y for y in ys if y]
            ax2.plot(xp, yp, marker="o", color=COLOR[be], label=LABEL[be], lw=1.8)
        ax2.axhline(1.0, color="k", ls="--", lw=0.8)
        ax2.set_xticks(list(xs)); ax2.set_xticklabels(xlabels)
        ax2.set_title(f"{dt}: speedup vs e3nn (>1 = faster than e3nn)")
        ax2.set_ylabel("e3nn_ms / backend_ms"); ax2.grid(True, alpha=0.3)
        if j == 0:
            ax2.legend(fontsize=8)
    fig.suptitle("Matched-fusion operator comparison (forward-only, channels=64): "
                 "ICTC competitiveness vs e3nn flips once fused", fontsize=13)
    fig.tight_layout()
    p = os.path.join(args.outdir, "operator_matched_fusion.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
