#!/usr/bin/env python3
"""Render the declared MACE Pearl causal graph without a Graphviz dependency."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def box(ax, xy, width, height, text, *, face="#F4F6F8", edge="#6B7680", size=8.6):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size, zorder=4)
    return patch


def arrow(ax, start, end, *, dashed=False, colour="#68737D", width=1.0):
    style = "--" if dashed else "-"
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=width,
        linestyle=style,
        color=colour,
        shrinkA=2,
        shrinkB=2,
        zorder=2,
    )
    ax.add_patch(patch)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(11.8, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(0.15, 6.92, "MACE causal graph", fontsize=13, weight="semibold", ha="left")
    ax.text(0.15, 6.60, "Pearl's ladder separates association, retraining interventions, and fixed-model counterfactuals", color="#57616B", ha="left")

    # Rung 2: algorithm-level intervention and validation-only selection.
    ax.add_patch(FancyBboxPatch((0.12, 3.53), 11.72, 2.72, boxstyle="round,pad=0.025", facecolor="#FBFCFD", edgecolor="#CBD2D9", linewidth=0.9))
    ax.text(0.33, 6.00, "Rung 2 · Doing", color="#356C8C", weight="semibold", ha="left")
    ax.text(0.33, 5.73, "independently retrained dataset-by-seed models", color="#57616B", ha="left", fontsize=8.5)

    confounders = [
        ("Dataset / split", 0.35, 5.08),
        ("Seed", 0.35, 4.45),
        ("Fixed protocol", 0.35, 3.82),
    ]
    for label, x, y in confounders:
        box(ax, (x, y), 1.55, 0.42, label, face="#FFFFFF", size=8.2)
    box(ax, (2.35, 4.61), 1.75, 0.56, "Scope\nFinal / Persistent", face="#F7EDE9", edge="#B25B45")
    box(ax, (2.35, 3.82), 1.75, 0.56, "Policy\nOff / Diagonal / Aligned / Full", face="#EAF1F5", edge="#356C8C", size=8.0)
    box(ax, (4.75, 4.22), 1.70, 0.64, "Optimisation\ntrajectory", size=8.5)
    box(ax, (7.08, 4.22), 1.78, 0.64, "Validation-selected\ncheckpoint", face="#F4F2EB", edge="#8A7650", size=8.3)
    box(ax, (9.54, 4.22), 1.78, 0.64, "Held-out factual\nerror  Y", face="#F2F0E8", edge="#7A7153", size=8.5)

    for _, _, y in confounders:
        arrow(ax, (1.90, y + 0.21), (4.75, 4.54), colour="#98A1AA", width=0.8)
    arrow(ax, (4.10, 4.89), (4.75, 4.62))
    arrow(ax, (4.10, 4.10), (4.75, 4.42))
    arrow(ax, (6.45, 4.54), (7.08, 4.54))
    arrow(ax, (8.86, 4.54), (9.54, 4.54))

    # Rung 3: fixed-checkpoint potential outcomes.
    ax.add_patch(FancyBboxPatch((0.12, 0.55), 11.72, 2.45, boxstyle="round,pad=0.025", facecolor="#FBFCFD", edgecolor="#CBD2D9", linewidth=0.9))
    ax.text(0.33, 2.75, "Rung 3 · Imagining", color="#B25B45", weight="semibold", ha="left")
    ax.text(0.33, 2.48, "same selected predictor and held-out structures under phase interventions", color="#57616B", ha="left", fontsize=8.5)

    box(ax, (0.35, 1.31), 1.56, 0.62, "Phase intervention\nI", face="#F7EDE9", edge="#B25B45")
    box(ax, (2.34, 1.31), 1.52, 0.62, "Relative phase\nΦ(I)")
    box(ax, (4.30, 1.31), 1.62, 0.62, "Hermitian density\nR(I)", face="#EAF1F5", edge="#356C8C")
    box(ax, (6.40, 1.31), 1.68, 0.62, "Frozen prediction\nf_K(I, X)")
    box(ax, (8.60, 1.31), 1.78, 0.62, "Potential outcome\nY_K(I)", face="#F2F0E8", edge="#7A7153")
    box(ax, (6.85, 0.67), 0.78, 0.42, "Same X", face="#FFFFFF", size=8.2)
    arrow(ax, (1.91, 1.62), (2.34, 1.62))
    arrow(ax, (3.86, 1.62), (4.30, 1.62))
    arrow(ax, (5.92, 1.62), (6.40, 1.62))
    arrow(ax, (8.08, 1.62), (8.60, 1.62))
    arrow(ax, (7.24, 1.09), (7.24, 1.31), colour="#98A1AA", width=0.8)
    arrow(ax, (7.97, 4.22), (7.18, 1.93), dashed=True, colour="#8A7650")
    ax.text(7.74, 3.13, "fixed weights", rotation=-71, fontsize=7.8, color="#7A6B50", ha="center")

    # Rung 1 is explicitly downstream and descriptive.
    box(ax, (9.13, 5.20), 2.18, 0.62, "Rung 1 · Seeing\nfitted-model associations", face="#FFFFFF", edge="#8B949E", size=8.4)
    arrow(ax, (8.42, 4.86), (9.48, 5.20), dashed=True, colour="#8B949E", width=0.9)
    arrow(ax, (9.49, 1.93), (10.08, 5.20), dashed=True, colour="#8B949E", width=0.9)

    ax.text(
        0.35,
        0.24,
        "Causal unit: one independently trained dataset × seed model.  Atomic and frame outcomes remain nested diagnostics.",
        fontsize=8.2,
        color="#57616B",
        ha="left",
    )

    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
