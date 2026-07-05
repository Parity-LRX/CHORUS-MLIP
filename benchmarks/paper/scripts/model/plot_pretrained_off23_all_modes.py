#!/usr/bin/env python3
"""Plot OFF23-small pretrained timings in the same five-mode style as the main throughput plots."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

DEGREE = 50
MODES = [
    "MACE e3nn",
    "MACE cuEq",
    "ICTC eager",
    "ICTC compiled",
    "ICTC+cuEq compiled",
]
STYLE = {
    "MACE e3nn": {"color": "#4C78A8", "marker": "o"},
    "MACE cuEq": {"color": "#F58518", "marker": "s"},
    "ICTC eager": {"color": "#54A24B", "marker": "^"},
    "ICTC compiled": {"color": "#B279A2", "marker": "D"},
    "ICTC+cuEq compiled": {"color": "#E45756", "marker": "P"},
}

RAW_ROWS = [
    # 4090 OFF23-small fixed-edge force timings. Degree is exactly 50, so
    # 50-neighbor-equivalent atoms equals the actual atom count.
    {"atoms": 128, "mode": "MACE e3nn", "time_ms": 12.232},
    {"atoms": 128, "mode": "MACE cuEq", "time_ms": 17.944},
    {"atoms": 128, "mode": "ICTC eager", "time_ms": 11.391},
    {"atoms": 128, "mode": "ICTC compiled", "time_ms": 1.503},
    {"atoms": 128, "mode": "ICTC+cuEq compiled", "time_ms": 3.990},
    {"atoms": 512, "mode": "MACE e3nn", "time_ms": 13.617},
    {"atoms": 512, "mode": "MACE cuEq", "time_ms": 17.899},
    {"atoms": 512, "mode": "ICTC eager", "time_ms": 13.905},
    {"atoms": 512, "mode": "ICTC compiled", "time_ms": 4.488},
    {"atoms": 512, "mode": "ICTC+cuEq compiled", "time_ms": 4.692},
]


def fmt_x(value: float, _pos: int) -> str:
    return f"{int(value):d}" if abs(value - round(value)) < 1e-6 else f"{value:g}"


def rows_with_metrics() -> list[dict[str, str]]:
    baseline = {r["atoms"]: r["time_ms"] for r in RAW_ROWS if r["mode"] == "MACE e3nn"}
    out: list[dict[str, str]] = []
    for r in RAW_ROWS:
        atoms = int(r["atoms"])
        time_ms = float(r["time_ms"])
        equiv_atoms50 = atoms * DEGREE / 50.0
        throughput_equiv_atoms50_s = equiv_atoms50 / (time_ms / 1000.0)
        speedup = baseline[atoms] / time_ms
        out.append(
            {
                "model": "OFF23-small",
                "atoms": str(atoms),
                "directed_edges": str(atoms * DEGREE),
                "equiv_atoms50": f"{equiv_atoms50:.6g}",
                "mode": str(r["mode"]),
                "time_ms": f"{time_ms:.6g}",
                "throughput_equiv_atoms50_s": f"{throughput_equiv_atoms50_s:.6g}",
                "speedup_vs_mace_e3nn": f"{speedup:.6g}",
            }
        )
    return out


def write_csv(rows: list[dict[str, str]]) -> Path:
    out = ROOT / "pretrained_off23_all_modes_4090.csv"
    fields = [
        "model",
        "atoms",
        "directed_edges",
        "equiv_atoms50",
        "mode",
        "time_ms",
        "throughput_equiv_atoms50_s",
        "speedup_vs_mace_e3nn",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"selected: {out}")
    return out


def plot(rows: list[dict[str, str]]) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15), sharex=True)
    panels = [
        (axes[0], "throughput_equiv_atoms50_s", "Throughput", "Throughput (10$^3$ equiv. atoms s$^{-1}$)"),
        (axes[1], "speedup_vs_mace_e3nn", "Speedup", "Speedup vs MACE e3nn"),
    ]
    for pidx, (ax, field, title, ylabel) in enumerate(panels):
        ax.text(
            -0.13,
            1.08,
            chr(ord("a") + pidx),
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )
        for mode in MODES:
            vals = [
                r
                for r in rows
                if r["mode"] == mode
            ]
            vals.sort(key=lambda r: float(r["equiv_atoms50"]))
            y = [float(r[field]) for r in vals]
            if field == "throughput_equiv_atoms50_s":
                y = [v / 1000.0 for v in y]
            style = STYLE[mode]
            ax.plot(
                [float(r["equiv_atoms50"]) for r in vals],
                y,
                label=mode,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.7,
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
        if field == "speedup_vs_mace_e3nn":
            ax.axhline(1.0, color="#4C78A8", linestyle="--", linewidth=0.85, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Equivalent atoms per graph (50 directed neighbors)")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xticks([128, 512])
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.7)
        ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(
        0.012,
        0.02,
        "RTX 4090, FP32, OFF23-small, fixed 50 directed edges/atom. "
        "Native MACE modes used isolated e3nn 0.4.4 for old OFF pickle compatibility; ICTC modes used current MACE-ICTC.",
        ha="left",
        fontsize=7.2,
        color="#333333",
    )
    fig.tight_layout(rect=(0.035, 0.095, 1.0, 0.89), w_pad=1.25)

    paths: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        out = FIG / f"pretrained_off23_all_modes_4090.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"figure: {out}")
        paths.append(out)
    plt.close(fig)
    return paths


def main() -> None:
    rows = rows_with_metrics()
    write_csv(rows)
    plot(rows)


if __name__ == "__main__":
    main()
