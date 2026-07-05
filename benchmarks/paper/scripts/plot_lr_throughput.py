#!/usr/bin/env python3
"""Long-range cost / throughput figure (RTX 4090 atom-count sweep).

Matches the paper figure style (setup_style + log-log throughput panels, save_all
-> pdf/png/svg). Reads the bench_lr JSON written by bench_lr_throughput.py:
    {backend, channels, lmax, num_interaction, degree, dtype,
     results: [{mode, cond, N, ms, atoms_s, sane, note}, ...]}

Usage: plot_lr_throughput.py <results.json> <out_stem> [--metric atoms_s|ms]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

# condition palette drawn from the paper's WHOLE_MODEL_COLORS family
COND_COLORS = {"none": "#4a4a4a", "elec": "#08519c", "elec-mp": "#6a51a3", "disp": "#d95f02", "both": "#41ab5d",
               "disp-respa": "#d95f02", "disp-mf": "#fd8d3c", "disp-c6": "#c51b8a"}
COND_MARKERS = {"none": "o", "elec": "s", "elec-mp": "*", "disp": "^", "both": "D",
                "disp-respa": "v", "disp-mf": ">", "disp-c6": "P"}
# rRESPA-accelerated dispersion (deployed MBD every K steps): dashed, same orange as disp.
COND_LINESTYLE = {"disp-respa": "--"}
COND_LABELS = {
    "none": "no long-range",
    "elec": "electrostatics (scalar)",
    "elec-mp": "electrostatics (multipole)",
    "disp": "dispersion (MBD, in-graph)",
    "both": "elec + dispersion",
    "disp-mf": "dispersion (MBD, matrix-free)",
    "disp-respa": "dispersion (MBD) + rRESPA, K=20",
    "disp-c6": "dispersion (pairwise C6)",
}
COND_ORDER = ["none", "elec", "elec-mp", "disp", "disp-respa", "disp-c6", "both"]

MODE_TITLES = {
    "train": "Training (E+F+backward)",
    "makefx-train": "Training (make_fx fused)",
    "infer": "Inference (E+F)",
    "aoti-infer": "Inference (AOTI deploy)",
    "md": "MD step (eager E+F)",
    "aoti-md": "MD step (AOTI + LAMMPS)",
}
MODE_ORDER = ["train", "makefx-train", "infer", "aoti-infer", "md", "aoti-md"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("out_stem", type=Path)
    ap.add_argument("--metric", choices=["atoms_s", "ms"], default="atoms_s")
    args = ap.parse_args()

    d = json.loads(args.json.read_text())
    res = d["results"]
    setup_style()

    modes = [m for m in MODE_ORDER if any(c["mode"] == m for c in res)]
    modes += sorted({c["mode"] for c in res} - set(modes))  # any unknown modes last

    ncol = 3 if len(modes) > 4 else 2
    nrow = (len(modes) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.5 * nrow), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    metric = args.metric
    ylabel = "throughput (atoms/s, log)" if metric == "atoms_s" else "time per step (ms, log)"

    for i, mode in enumerate(modes):
        ax = axes[i]
        for cond in COND_ORDER:
            cells = [c for c in res if c["mode"] == mode and c["cond"] == cond and c.get(metric) is not None]
            cells = sorted(cells, key=lambda c: c["N"])
            xs = [c["N"] for c in cells]
            ys = [c[metric] for c in cells]
            if not xs:
                continue
            # flag eager-equivalent cells (disp/both make_fx-train) with a hollow marker
            eager_eq = mode == "makefx-train" and any(
                "eager" in (c.get("note") or "").lower() for c in cells
            )
            ax.plot(
                xs, ys,
                marker=COND_MARKERS[cond],
                markerfacecolor="white" if eager_eq else COND_COLORS[cond],
                color=COND_COLORS[cond],
                label=COND_LABELS[cond],
                linewidth=1.5, markersize=5.5, markeredgewidth=1.2,
                linestyle=COND_LINESTYLE.get(cond, "-"),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(MODE_TITLES.get(mode, mode))
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xticks(sorted({c["N"] for c in res}))
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
        ax.get_yaxis().set_minor_formatter(mticker.NullFormatter())  # only decade (10^n) labels on the shared y

    for j in range(len(modes), len(axes)):
        axes[j].axis("off")

    for r in range(nrow):
        axes[r * ncol].set_ylabel(ylabel)
    for cc in range(ncol):
        idx = (nrow - 1) * ncol + cc
        if idx < len(modes):
            axes[idx].set_xlabel("atoms")
        # also label the bottom-most visible axis in each column
    # ensure every column's lowest populated panel carries the x-label
    for cc in range(ncol):
        for r in range(nrow - 1, -1, -1):
            idx = r * ncol + cc
            if idx < len(modes):
                axes[idx].set_xlabel("atoms")
                break

    handles, labels, _seen = [], [], set()
    for _ax in axes:
        for _h, _l in zip(*_ax.get_legend_handles_labels()):
            if _l not in _seen:
                _seen.add(_l); handles.append(_h); labels.append(_l)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))

    cfg = f"fp32, C={d['channels']}, $\\ell$={d['lmax']}, {d['num_interaction']} interactions, degree {d['degree']}"
    fig.suptitle(f"Long-range cost on RTX 4090 ({cfg})", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    save_all(fig, args.out_stem)


if __name__ == "__main__":
    main()
