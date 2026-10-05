#!/usr/bin/env python3
"""Render compact journal-style causal figures from audited CHORUS exports."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


# Match the serif, low-contrast visual system used by the main performance
# figures.  The causal panels retain their blue/orange semantics but no longer
# introduce a second typographic hierarchy inside the paper.
INK = "#272727"
MUTED = "#747474"
HAIRLINE = "#E6E6E3"
PAPER = "#FFFFFF"
CONTEXT = "#F5F5F3"
BLUE = "#205381"
BLUE_LIGHT = "#DDE7EF"
ORANGE = "#C96A3D"
ORANGE_LIGHT = "#F2E2DA"

BACKBONE_COLOR = {"MACE-ICTC": BLUE, "NequIP-SH": ORANGE}
DATASET_STYLE = {
    "mal": ("o", "MAL"),
    "bucky": ("s", "Buckyball"),
    "3bpa": ("^", "3BPA"),
    "zro": ("D", "Zr–O"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.55,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": PAPER,
            "figure.facecolor": PAPER,
        }
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=420,
            bbox_inches="tight" if tight else None,
            facecolor=PAPER,
        )
    plt.close(fig)


def add_node(
    ax: plt.Axes,
    centre: tuple[float, float],
    text: str,
    *,
    width: float = 0.18,
    height: float = 0.19,
    facecolor: str = PAPER,
    edgecolor: str = MUTED,
    weight: str = "normal",
    fontsize: float = 8.0,
) -> FancyBboxPatch:
    x, y = centre
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, zorder=4)
    return box


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.0,
    linestyle: str = "-",
    rad: float = 0.0,
    shrink: float = 1.0,
    patch_a: FancyBboxPatch | None = None,
    patch_b: FancyBboxPatch | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=shrink,
        shrinkB=shrink,
        patchA=patch_a,
        patchB=patch_b,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arrow)


def plot_identification_map(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.75))
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions = {
        "X": (0.095, 0.78),
        "T": (0.095, 0.47),
        "I": (0.095, 0.16),
        "W": (0.315, 0.68),
        "K": (0.515, 0.68),
        "Phi": (0.515, 0.22),
        "Rho": (0.735, 0.43),
        "Y": (0.925, 0.43),
    }
    node_specs = {
        "X": ("design context  X", 0.150, 0.095, CONTEXT, MUTED, 7.0),
        "T": ("operator policy  T", 0.145, 0.095, BLUE_LIGHT, BLUE, 7.2),
        "I": ("frozen intervention  I", 0.165, 0.095, ORANGE_LIGHT, ORANGE, 7.0),
        "W": ("optimised weights  W", 0.155, 0.095, PAPER, MUTED, 7.0),
        "K": ("selected checkpoint  K", 0.165, 0.095, PAPER, MUTED, 7.0),
        "Phi": ("relative-phase order  Φ", 0.165, 0.095, PAPER, MUTED, 6.9),
        "Rho": ("Hermitian density  ρ", 0.155, 0.095, PAPER, MUTED, 7.0),
        "Y": ("held-out Force MAE  Y", 0.155, 0.095, PAPER, INK, 7.0),
    }

    nodes: dict[str, FancyBboxPatch] = {}
    for name, centre in positions.items():
        label, width, height, face, edge, fontsize = node_specs[name]
        nodes[name] = add_node(
            ax,
            centre,
            label,
            width=width,
            height=height,
            facecolor=face,
            edgecolor=edge,
            weight="normal",
            fontsize=fontsize,
        )

    context_edges = (
        ("X", "W", 0.00),
        ("X", "Rho", -0.16),
        ("X", "Y", -0.22),
    )
    training_edges = (
        ("T", "W", 0.0),
        ("T", "Rho", 0.16),
    )
    frozen_edges = (("I", "Phi", 0.0), ("Phi", "Rho", 0.0))
    shared_edges = (
        ("W", "K", 0.0),
        ("K", "Phi", 0.0),
        ("K", "Rho", 0.06),
        ("K", "Y", -0.20),
        ("Rho", "Y", 0.0),
    )
    for source, target, rad in context_edges:
        add_arrow(
            ax, positions[source], positions[target], color="#A9B0B6", linewidth=0.75,
            linestyle=(0, (2.0, 2.0)), rad=rad, patch_a=nodes[source], patch_b=nodes[target]
        )
    for source, target, rad in training_edges:
        add_arrow(
            ax, positions[source], positions[target], color=BLUE, linewidth=1.15, rad=rad,
            patch_a=nodes[source], patch_b=nodes[target]
        )
    for source, target, rad in frozen_edges:
        add_arrow(
            ax, positions[source], positions[target], color=ORANGE, linewidth=1.15, rad=rad,
            patch_a=nodes[source], patch_b=nodes[target]
        )
    for source, target, rad in shared_edges:
        add_arrow(
            ax, positions[source], positions[target], color=INK, linewidth=0.9, rad=rad,
            patch_a=nodes[source], patch_b=nodes[target]
        )

    legend = [
        Line2D([0], [0], color=BLUE, lw=1.35, label="retraining intervention"),
        Line2D([0], [0], color=ORANGE, lw=1.35, label="frozen-checkpoint intervention"),
        Line2D([0], [0], color="#A9B0B6", lw=0.9, linestyle=(0, (2.0, 2.0)), label="design conditioning"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False, fontsize=6.25)
    ax.text(
        0.50, 0.095, "causal unit: dataset × seed trained model",
        ha="center", va="center", fontsize=6.4, color=MUTED
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.98, bottom=0.02)
    save_figure(fig, output_dir, "causal_identification_map_v2")


def interval(value: object) -> tuple[float, float]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    return float(parsed[0]), float(parsed[1])


def component_name(treatment: str, control: str, scope: str) -> str:
    lookup = {
        ("diagonal", "off"): "Hermitian self-density · Final",
        ("aligned", "diagonal"): "Cross-neighbour completion · Final",
        ("full", "aligned"): f"Learned phase · {scope}",
    }
    return lookup[(treatment, control)]


def dataset_macro_points(
    frame: pd.DataFrame,
    *,
    backbone: str,
    scope: str,
    treatment: str,
    control: str,
    value_column: str,
    transform,
) -> dict[str, float]:
    subset = frame[
        (frame["backbone"] == backbone)
        & (frame["scope"] == scope)
        & (frame["treatment"] == treatment)
        & (frame["control"] == control)
        & (frame["metric"] == "test_force_mae")
    ]
    return {
        dataset: float(transform(group[value_column].astype(float).mean()))
        for dataset, group in subset.groupby("dataset", sort=False)
    }


def style_effect_axis(ax: plt.Axes, title: str, rows: list[str]) -> None:
    ax.set_title(title, loc="left", weight="normal", pad=7)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", color=HAIRLINE, linewidth=0.5, zorder=0)


def draw_macro_interval(
    ax: plt.Axes,
    *,
    y: float,
    low: float,
    high: float,
    estimate: float,
    color: str,
) -> None:
    """Draw a quiet forest-plot interval and its macro estimate."""
    ax.hlines(y, low, high, color=color, linewidth=1.25, alpha=0.55, zorder=1)
    ax.vlines(
        [low, high],
        y - 0.055,
        y + 0.055,
        color=color,
        linewidth=0.75,
        alpha=0.55,
        zorder=1,
    )
    ax.scatter(
        estimate,
        y,
        marker="D",
        s=29,
        facecolor=color,
        edgecolor=PAPER,
        linewidth=0.6,
        zorder=4,
    )


def dataset_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=PAPER,
            markeredgecolor=MUTED,
            markeredgewidth=0.8,
            markersize=4.4,
            label=label,
        )
        for marker, label in DATASET_STYLE.values()
    ]


def plot_trained_effects(input_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(input_dir / "trained_effect_summary.csv")
    pairs = pd.read_csv(input_dir / "trained_paired_effects.csv")
    summary = summary[
        (summary["metric"] == "test_force_mae")
        & ~((summary["treatment"] == "full") & (summary["control"] == "off"))
    ].copy()
    order = [
        ("Final", "diagonal", "off"),
        ("Final", "aligned", "diagonal"),
        ("Final", "full", "aligned"),
        ("Persistent", "full", "aligned"),
    ]
    row_labels = [component_name(treatment, control, scope) for scope, treatment, control in order]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.90), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    for panel, ax, backbone in zip(("a", "b"), axes, ("MACE-ICTC", "NequIP-SH")):
        color = BACKBONE_COLOR[backbone]
        style_effect_axis(ax, f"({panel})  {backbone}", row_labels)
        for y, (scope, treatment, control) in enumerate(order):
            row = summary[
                (summary["backbone"] == backbone)
                & (summary["scope"] == scope)
                & (summary["treatment"] == treatment)
                & (summary["control"] == control)
            ].iloc[0]
            estimate = float(row["macro_relative_error_reduction_percent"])
            low, high = interval(row["hierarchical_bootstrap_95pct_reduction_ci_percent"])
            points = dataset_macro_points(
                pairs,
                backbone=backbone,
                scope=scope,
                treatment=treatment,
                control=control,
                value_column="log_benefit",
                transform=lambda value: 100.0 * (1.0 - np.exp(-value)),
            )
            offsets = np.linspace(-0.12, 0.12, len(DATASET_STYLE))
            for offset, (dataset, (marker, _)) in zip(offsets, DATASET_STYLE.items()):
                ax.scatter(
                    points[dataset],
                    y + offset,
                    marker=marker,
                    s=18,
                    facecolor=PAPER,
                    edgecolor=MUTED,
                    linewidth=0.65,
                    zorder=3,
                )
            draw_macro_interval(
                ax,
                y=y,
                low=low,
                high=high,
                estimate=estimate,
                color=color,
            )
        ax.axvline(0.0, color=INK, linewidth=0.7, zorder=0)
        ax.set_xlim(-6.5, 33.5)
        ax.set_xticks([-5, 0, 10, 20, 30])
    axes[0].set_ylabel("")
    fig.supxlabel("Force-MAE reduction under paired retraining (%)", y=0.075)
    fig.legend(handles=dataset_legend(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(bottom=0.25, left=0.22, right=0.985, top=0.89)
    save_figure(fig, output_dir, "trained_effect_profile_v2", tight=False)


def plot_frozen_effects(input_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(input_dir / "frozen_effect_summary.csv")
    pairs = pd.read_csv(input_dir / "frozen_paired_effects.csv")
    pairs = pairs.rename(columns={"intervention": "control", "log_harm": "log_benefit"})
    pairs["treatment"] = "factual"
    pairs["metric"] = "test_force_mae"
    order = [
        ("Final", "aligned", "Final · aligned"),
        ("Final", "permute_mean", "Final · permuted"),
        ("Persistent", "aligned", "Persistent · aligned"),
        ("Persistent", "permute_mean", "Persistent · permuted"),
    ]
    row_labels = [label for _, _, label in order]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.90), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    for panel, ax, backbone in zip(("c", "d"), axes, ("MACE-ICTC", "NequIP-SH")):
        color = BACKBONE_COLOR[backbone]
        style_effect_axis(ax, f"({panel})  {backbone}", row_labels)
        for y, (scope, control, _) in enumerate(order):
            row = summary[
                (summary["backbone"] == backbone)
                & (summary["scope"] == scope)
                & (summary["control"] == control)
            ].iloc[0]
            reduction = float(row["macro_relative_error_reduction_percent"])
            low_reduction, high_reduction = interval(row["hierarchical_bootstrap_95pct_reduction_ci_percent"])
            estimate = 1.0 / (1.0 - reduction / 100.0)
            low = 1.0 / (1.0 - low_reduction / 100.0)
            high = 1.0 / (1.0 - high_reduction / 100.0)
            points = dataset_macro_points(
                pairs,
                backbone=backbone,
                scope=scope,
                treatment="factual",
                control=control,
                value_column="log_benefit",
                transform=np.exp,
            )
            offsets = np.linspace(-0.12, 0.12, len(DATASET_STYLE))
            for offset, (dataset, (marker, _)) in zip(offsets, DATASET_STYLE.items()):
                ax.scatter(
                    points[dataset],
                    y + offset,
                    marker=marker,
                    s=18,
                    facecolor=PAPER,
                    edgecolor=MUTED,
                    linewidth=0.65,
                    zorder=3,
                )
            draw_macro_interval(
                ax,
                y=y,
                low=low,
                high=high,
                estimate=estimate,
                color=color,
            )
        ax.axvline(1.0, color=INK, linewidth=0.7, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.88, 32.0)
        ax.set_xticks([1, 2, 4, 8, 16, 32], ["1×", "2×", "4×", "8×", "16×", "32×"])
    axes[0].set_ylabel("")
    fig.supxlabel(r"Force MAE after intervention / factual Force MAE ($\log_2$ scale)", y=0.075)
    fig.legend(handles=dataset_legend(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(bottom=0.25, left=0.22, right=0.985, top=0.89)
    save_figure(fig, output_dir, "frozen_effect_profile_v2", tight=False)


def plot_combined_effects(input_dir: Path, output_dir: Path) -> None:
    """Render the four causal panels as one compact, scale-consistent figure."""
    trained_summary = pd.read_csv(input_dir / "trained_effect_summary.csv")
    trained_pairs = pd.read_csv(input_dir / "trained_paired_effects.csv")
    trained_summary = trained_summary[
        (trained_summary["metric"] == "test_force_mae")
        & ~(
            (trained_summary["treatment"] == "full")
            & (trained_summary["control"] == "off")
        )
    ].copy()
    trained_order = [
        ("Final", "diagonal", "off"),
        ("Final", "aligned", "diagonal"),
        ("Final", "full", "aligned"),
        ("Persistent", "full", "aligned"),
    ]
    trained_labels = [
        component_name(treatment, control, scope)
        for scope, treatment, control in trained_order
    ]

    frozen_summary = pd.read_csv(input_dir / "frozen_effect_summary.csv")
    frozen_pairs = pd.read_csv(input_dir / "frozen_paired_effects.csv")
    frozen_pairs = frozen_pairs.rename(
        columns={"intervention": "control", "log_harm": "log_benefit"}
    )
    frozen_pairs["treatment"] = "factual"
    frozen_pairs["metric"] = "test_force_mae"
    frozen_order = [
        ("Final", "aligned", "Final · aligned"),
        ("Final", "permute_mean", "Final · permuted"),
        ("Persistent", "aligned", "Persistent · aligned"),
        ("Persistent", "permute_mean", "Persistent · permuted"),
    ]
    frozen_labels = [label for _, _, label in frozen_order]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.0),
        sharex="row",
        sharey="row",
        gridspec_kw={"wspace": 0.08, "hspace": 0.52},
    )

    for panel, ax, backbone in zip(
        ("a", "b"), axes[0], ("MACE-ICTC", "NequIP-SH")
    ):
        color = BACKBONE_COLOR[backbone]
        style_effect_axis(ax, f"({panel})  {backbone}", trained_labels)
        for y, (scope, treatment, control) in enumerate(trained_order):
            row = trained_summary[
                (trained_summary["backbone"] == backbone)
                & (trained_summary["scope"] == scope)
                & (trained_summary["treatment"] == treatment)
                & (trained_summary["control"] == control)
            ].iloc[0]
            estimate = float(row["macro_relative_error_reduction_percent"])
            low, high = interval(
                row["hierarchical_bootstrap_95pct_reduction_ci_percent"]
            )
            points = dataset_macro_points(
                trained_pairs,
                backbone=backbone,
                scope=scope,
                treatment=treatment,
                control=control,
                value_column="log_benefit",
                transform=lambda value: 100.0 * (1.0 - np.exp(-value)),
            )
            offsets = np.linspace(-0.12, 0.12, len(DATASET_STYLE))
            for offset, (dataset, (marker, _)) in zip(
                offsets, DATASET_STYLE.items()
            ):
                ax.scatter(
                    points[dataset],
                    y + offset,
                    marker=marker,
                    s=18,
                    facecolor=PAPER,
                    edgecolor=MUTED,
                    linewidth=0.65,
                    zorder=3,
                )
            draw_macro_interval(
                ax,
                y=y,
                low=low,
                high=high,
                estimate=estimate,
                color=color,
            )
        ax.axvline(0.0, color=INK, linewidth=0.7, zorder=0)
        ax.set_xlim(-6.5, 33.5)
        ax.set_xticks([-5, 0, 10, 20, 30])

    for panel, ax, backbone in zip(
        ("c", "d"), axes[1], ("MACE-ICTC", "NequIP-SH")
    ):
        color = BACKBONE_COLOR[backbone]
        style_effect_axis(ax, f"({panel})  {backbone}", frozen_labels)
        for y, (scope, control, _) in enumerate(frozen_order):
            row = frozen_summary[
                (frozen_summary["backbone"] == backbone)
                & (frozen_summary["scope"] == scope)
                & (frozen_summary["control"] == control)
            ].iloc[0]
            reduction = float(row["macro_relative_error_reduction_percent"])
            low_reduction, high_reduction = interval(
                row["hierarchical_bootstrap_95pct_reduction_ci_percent"]
            )
            estimate = 1.0 / (1.0 - reduction / 100.0)
            low = 1.0 / (1.0 - low_reduction / 100.0)
            high = 1.0 / (1.0 - high_reduction / 100.0)
            points = dataset_macro_points(
                frozen_pairs,
                backbone=backbone,
                scope=scope,
                treatment="factual",
                control=control,
                value_column="log_benefit",
                transform=np.exp,
            )
            offsets = np.linspace(-0.12, 0.12, len(DATASET_STYLE))
            for offset, (dataset, (marker, _)) in zip(
                offsets, DATASET_STYLE.items()
            ):
                ax.scatter(
                    points[dataset],
                    y + offset,
                    marker=marker,
                    s=18,
                    facecolor=PAPER,
                    edgecolor=MUTED,
                    linewidth=0.65,
                    zorder=3,
                )
            draw_macro_interval(
                ax,
                y=y,
                low=low,
                high=high,
                estimate=estimate,
                color=color,
            )
        ax.axvline(1.0, color=INK, linewidth=0.7, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.88, 32.0)
        ax.set_xticks(
            [1, 2, 4, 8, 16, 32],
            ["1×", "2×", "4×", "8×", "16×", "32×"],
        )

    axes[0, 0].set_ylabel("")
    axes[1, 0].set_ylabel("")
    fig.text(
        0.603,
        0.535,
        "Force-MAE reduction under paired retraining (%)",
        ha="center",
        va="center",
        fontsize=7.8,
    )
    fig.text(
        0.603,
        0.080,
        r"Force MAE after intervention / factual Force MAE ($\log_2$ scale)",
        ha="center",
        va="center",
        fontsize=7.8,
    )
    fig.legend(
        handles=dataset_legend(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.603, 0.005),
    )
    fig.subplots_adjust(bottom=0.15, left=0.22, right=0.985, top=0.96)
    save_figure(fig, output_dir, "causal_effect_profile_v3", tight=False)


def plot_dose(input_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(input_dir / "phase_dose_response.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    for panel, ax, backbone in zip(("a", "b"), axes, ("MACE-ICTC", "NequIP-SH")):
        color = BACKBONE_COLOR[backbone]
        for scope, linestyle, filled in (("Final", (0, (3, 2)), False), ("Persistent", "-", True)):
            group = frame[(frame["backbone"] == backbone) & (frame["scope"] == scope)].sort_values("lambda")
            ax.plot(
                group["lambda"],
                group["force_mae_ratio_to_factual"],
                color=color,
                linewidth=1.7,
                linestyle=linestyle,
                marker="o",
                markersize=4.2,
                markerfacecolor=color if filled else PAPER,
                markeredgecolor=color,
                markeredgewidth=0.9,
                zorder=3,
            )
            left_row = group.iloc[0]
            ax.annotate(
                scope,
                (float(left_row["lambda"]), float(left_row["force_mae_ratio_to_factual"])),
                xytext=(7, 4 if scope == "Persistent" else 6),
                textcoords="offset points",
                fontsize=7.1,
                color=color,
                weight="normal",
            )
        ax.set_title(f"({panel})  {backbone}", loc="left", weight="normal", pad=7)
        ax.axhline(1.0, color=INK, linewidth=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.92, 3.70)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color=HAIRLINE, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Force MAE / factual Force MAE")
    fig.supxlabel("Relative-phase dose, λ", y=0.05, fontsize=8.5)
    fig.subplots_adjust(bottom=0.23, left=0.10, right=0.985, top=0.88)
    save_figure(fig, output_dir, "phase_dose_profile_v2")


def main() -> None:
    args = parse_args()
    configure_style()
    plot_identification_map(args.output_dir)
    plot_trained_effects(args.input_dir, args.output_dir)
    plot_frozen_effects(args.input_dir, args.output_dir)
    plot_combined_effects(args.input_dir, args.output_dir)
    plot_dose(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
