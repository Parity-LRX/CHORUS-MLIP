#!/usr/bin/env python
"""Publication fp32 operator figure (forward-only, matched fusion, channels=64).
Outputs vector PDF + 300-dpi PNG + SVG to figures/.
Left: per-config forward time (ms, log) for ICTC eager / ICTC torch.compile / e3nn / cartnn.
Right: speedup vs e3nn (e3nn_ms / backend_ms; >1 = faster than e3nn) for fused ICTC and cartnn.
"""
import argparse, csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.linewidth": 0.9, "figure.dpi": 150, "svg.fonttype": "none",
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})

C_EAGER, C_COMP, C_E3NN, C_CART = "#bdd7e7", "#08519c", "#636363", "#cb181d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--edges", type=int, default=100000)
    ap.add_argument("--channels", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = [r for r in csv.DictReader(open(args.csv))
            if r["status"] == "ok" and r["dtype"] == "float32"
            and int(r["edges"]) == args.edges and int(r["channels"]) == args.channels]
    configs = []
    for r in rows:
        c = (int(r["hidden_lmax"]), int(r["max_ell"]))
        if c not in configs:
            configs.append(c)
    configs.sort()
    V = defaultdict(dict)
    for r in rows:
        V[(int(r["hidden_lmax"]), int(r["max_ell"]))][r["backend"]] = float(r["total_ms"])
    labels = [f"{a}/{b}" for a, b in configs]
    xs = range(len(configs))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # ---- left: grouped bars (log ms) ----
    series = [("ictd_eager", "ICTC (eager)", C_EAGER),
              ("ictd_compile", "ICTC (torch.compile)", C_COMP),
              ("e3nn", "e3nn (spherical, ref)", C_E3NN),
              ("cartnn", "cartnn (Cartesian 3$^\\ell$)", C_CART)]
    n = len(series); w = 0.20
    for i, (be, lab, col) in enumerate(series):
        ys = [V[c].get(be) for c in configs]
        xpos = [x + (i - n / 2) * w + w / 2 for x in xs]
        axL.bar([xp for xp, y in zip(xpos, ys) if y], [y for y in ys if y],
                width=w, color=col, label=lab, edgecolor="white", linewidth=0.4)
    axL.set_yscale("log")
    axL.set_xticks(list(xs)); axL.set_xticklabels(labels)
    axL.set_xlabel("angular config  (hidden $L$ / edge $L$)")
    axL.set_ylabel("forward time per call (ms)")
    axL.set_title(f"(a) Operator forward time  (fp32, C={args.channels}, E={args.edges:,})")
    axL.grid(True, axis="y", which="both", alpha=0.25)
    axL.legend(frameon=False, loc="upper left", ncol=1)

    # ---- right: speedup vs e3nn ----
    for be, lab, col, mk in (("ictd_compile", "ICTC (torch.compile)", C_COMP, "o"),
                             ("cartnn", "cartnn", C_CART, "s")):
        ys = []
        for c in configs:
            e = V[c].get("e3nn"); v = V[c].get(be)
            ys.append(e / v if (e and v) else None)
        xp = [x for x, y in zip(xs, ys) if y]; yp = [y for y in ys if y]
        axR.plot(xp, yp, marker=mk, color=col, label=lab, lw=2.2, ms=8)
        for x, y in zip(xp, yp):
            axR.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 8 if be == "ictd_compile" else -14), ha="center", fontsize=9, color=col)
    axR.axhline(1.0, color="k", ls="--", lw=1.0)
    axR.text(len(configs) - 1, 1.02, "e3nn baseline", ha="right", va="bottom", fontsize=9, color="k")
    axR.set_xticks(list(xs)); axR.set_xticklabels(labels)
    axR.set_xlabel("angular config  (hidden $L$ / edge $L$)")
    axR.set_ylabel("speedup vs e3nn  ($t_{e3nn}/t$)")
    axR.set_title("(b) Speedup vs e3nn   (>1 = faster than e3nn)")
    axR.grid(True, alpha=0.25)
    axR.legend(frameon=False, loc="upper left")
    axR.set_ylim(0, max(2.0, max([V[c].get("e3nn", 0) / V[c]["ictd_compile"]
                 for c in configs if V[c].get("ictd_compile")] + [1.4]) * 1.15))

    fig.suptitle("Equivariant tensor-product operator on RTX 4090 (fp32, forward-only, matched fusion)",
                 fontsize=13.5, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(args.outdir, f"operator_fp32_matched_fusion.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
