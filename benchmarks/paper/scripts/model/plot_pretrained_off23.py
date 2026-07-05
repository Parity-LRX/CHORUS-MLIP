#!/usr/bin/env python3
"""Plot the OFF23-small pretrained conversion and deployment checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#B279A2"
GRID = "#D9D9D9"


def fmt_k(value: float, _pos: int) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k".replace(".0k", "k")
    return f"{value:g}"


def style_axis(ax, *, ylabel: str, xlabel: str | None = None, logy: bool = False) -> None:
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.17,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    atoms = [128, 512]
    edges = [6400, 25600]
    native_ms = [12.232, 13.617]
    ictd_ms = [11.391, 13.905]
    rel_energy = [7.729e-7, 3.079e-7]
    rel_force = [1.118e-6, 1.126e-6]

    cores = ["bridge-U", "cuEq product"]
    eager = [11.228, 13.757]
    aoti = [4.343, 4.703]
    speedup = [2.59, 2.93]

    lmp_atoms = [128, 512]
    lmp_bridge_pair = [1.459, 5.817]
    lmp_cueq_pair = [4.318, 5.979]
    lmp_bridge_steps = [679.342, 149.400]
    lmp_cueq_steps = [230.814, 145.825]

    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.25))
    ax = axes[0, 0]
    add_panel_label(ax, "a")
    ax.plot(edges, native_ms, marker="o", color=BLUE, label="native MACE")
    ax.plot(edges, ictd_ms, marker="s", color=ORANGE, label="converted ICTC")
    for x, y, n in zip(edges, native_ms, atoms):
        ax.annotate(f"{n} atoms", (x, y), textcoords="offset points", xytext=(3, 5), fontsize=7.2)
    style_axis(ax, ylabel="fwd+force time (ms)", xlabel="Directed edges")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.legend(frameon=False, loc="best")
    ax.set_title("OFF23 small parity timing")

    ax = axes[0, 1]
    add_panel_label(ax, "b")
    ax.plot(atoms, rel_energy, marker="o", color=GREEN, label="energy")
    ax.plot(atoms, rel_force, marker="D", color=RED, label="force")
    ax.axhline(3e-3, color="#777777", linestyle="--", linewidth=0.8, label="FP32 export tol")
    style_axis(ax, ylabel="relative difference", xlabel="Atoms at 50 directed neighbors", logy=True)
    ax.legend(frameon=False, loc="best")
    ax.set_title("native MACE vs converted ICTC")

    ax = axes[1, 0]
    add_panel_label(ax, "c")
    x = range(len(cores))
    width = 0.34
    ax.bar([i - width / 2 for i in x], eager, width=width, color=PURPLE, label="eager")
    ax.bar([i + width / 2 for i in x], aoti, width=width, color=GREEN, label="AOTI .pt2")
    for i, s in enumerate(speedup):
        ax.text(i + width / 2, aoti[i] + 0.35, f"{s:.2f}x", ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(list(x), cores)
    style_axis(ax, ylabel="fwd+force time (ms)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Python AOTI export, 512 atoms x 50 edges")

    ax = axes[1, 1]
    add_panel_label(ax, "d")
    ax.plot(lmp_atoms, lmp_bridge_pair, marker="o", color=ORANGE, label="bridge-U pair ms")
    ax.plot(lmp_atoms, lmp_cueq_pair, marker="s", color=RED, label="cuEq pair ms")
    ax2 = ax.twinx()
    ax2.plot(lmp_atoms, lmp_bridge_steps, marker="o", linestyle="--", color=BLUE, label="bridge-U steps/s")
    ax2.plot(lmp_atoms, lmp_cueq_steps, marker="s", linestyle="--", color="#72B7B2", label="cuEq steps/s")
    style_axis(ax, ylabel="LAMMPS pair time (ms/step)", xlabel="Atoms")
    ax2.set_ylabel("LAMMPS steps/s")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper center", ncol=2)
    ax.set_title("LAMMPS mff/torch AOTI smoke")

    fig.suptitle(
        "OFF23 small pretrained MACE to MACE-ICTC on RTX 4090",
        y=0.985,
        fontsize=11.5,
    )
    fig.text(
        0.012,
        0.014,
        "Native OFF reference used isolated e3nn 0.4.4; conversion/export used current env. "
        "Synthetic fixed-edge graphs are parity/kernel checks, not physical MD trajectories.",
        ha="left",
        fontsize=7.4,
        color="#333333",
    )
    fig.tight_layout(rect=(0.035, 0.055, 0.995, 0.95), w_pad=1.25, h_pad=1.45)

    for ext in ("pdf", "svg", "png"):
        out = FIG / f"pretrained_off23_summary_4090.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"figure: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
