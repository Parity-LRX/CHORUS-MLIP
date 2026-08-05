#!/usr/bin/env python3
"""Combine per-dataset CHORUS phase diagnostics into paper-ready artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ORDER = ("buckyball", "mal", "sti", "3bpa", "t1x")
DISPLAY = {
    "buckyball": "MD22 Buckyball",
    "mal": "xxMD MAL",
    "sti": "xxMD STI",
    "3bpa": "3BPA 300 K",
    "t1x": "Transition1x subset",
}
FIG_DISPLAY = {
    "buckyball": "Buckyball",
    "mal": "MAL",
    "sti": "STI",
    "3bpa": "3BPA (300 K)",
    "t1x": "T1x-50k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--final-root",
        default=None,
        help="Optional matched CHORUS-Final diagnostics root used for scope comparison",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_payloads(root: Path) -> dict[str, dict[str, Any]]:
    payloads = {}
    for name in ORDER:
        path = root / name / "phase_interference.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads[name] = json.loads(path.read_text())
    return payloads


def percent_change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def channel_label(key: str) -> str:
    fields = key.split(".")
    layer = fields[0].replace("layer", "layer ")
    return f"{layer}, {fields[1]}"


def phase_small_multiples(payloads: dict[str, dict[str, Any]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    first_channels = sorted(
        payloads[ORDER[0]]["phase_histograms"]["channels"]
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.06, 0.9, len(first_channels)))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.5), sharex=True, constrained_layout=True)
    flat = axes.ravel()
    for index, name in enumerate(ORDER):
        axis = flat[index]
        histogram = payloads[name]["phase_histograms"]
        edges = np.asarray(histogram["bin_edges_radians"], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        channels = histogram["channels"]
        axis.axvspan(-math.pi, -math.pi / 2, color="#D55E00", alpha=0.07, lw=0)
        axis.axvspan(math.pi / 2, math.pi, color="#D55E00", alpha=0.07, lw=0)
        for color, key in zip(colors, first_channels):
            axis.plot(
                centers,
                channels[key]["relative_amplitude_weighted_probability"],
                color=color,
                lw=1.35,
                label=channel_label(key),
            )
        axis.set_title(DISPLAY[name], loc="left", fontweight="semibold")
        axis.set_xlim(-math.pi, math.pi)
        axis.set_xticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
        axis.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        axis.set_ylabel("Probability")
        axis.spines[["top", "right"]].set_visible(False)
    flat[3].set_xlabel(r"Relative phase $\Delta\theta_{jk}$")
    flat[4].set_xlabel(r"Relative phase $\Delta\theta_{jk}$")
    flat[5].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[5].legend(
        handles,
        labels,
        frameon=False,
        loc="center left",
        title="Persistent stream",
        title_fontproperties={"weight": "semibold", "size": 9},
    )
    fig.suptitle(
        "Gauge-invariant phase differences across validation environments",
        fontsize=11,
        fontweight="semibold",
    )
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(output / f"relative_phase_histograms.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def evidence_figure(payloads: dict[str, dict[str, Any]], output: Path) -> None:
    first_channels = sorted(
        payloads[ORDER[0]]["phase_histograms"]["channels"]
    )
    destructive = np.asarray(
        [
            [
                100.0
                * payloads[name]["phase_histograms"]["channels"][key]["summary"][
                    "destructive_kernel_fraction"
                ]
                for key in first_channels
            ]
            for name in ORDER
        ]
    )
    ratios = np.asarray(
        [
            [
                payloads[name]["interventions"][mode]["force_error"]["mae"]
                / payloads[name]["interventions"]["native"]["force_error"]["mae"]
                for mode in ("zero", "permute")
            ]
            for name in ORDER
        ]
    )
    global_shift = np.asarray(
        [
            1000.0
            * payloads[name]["interventions"]["global-shift"]["force_change"]["mae"]
            for name in ORDER
        ]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.0, 3.35),
        gridspec_kw={"width_ratios": [1.65, 1.0, 0.8]},
        constrained_layout=True,
    )
    image = axes[0].imshow(destructive, cmap="Oranges", vmin=0.0, vmax=50.0, aspect="auto")
    axes[0].set_xticks(range(len(first_channels)), [channel_label(k) for k in first_channels], rotation=35, ha="right")
    axes[0].set_yticks(range(len(ORDER)), [DISPLAY[name] for name in ORDER])
    axes[0].set_title("Destructive pair-kernel mass", loc="left", fontweight="semibold")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.05, pad=0.03)
    colorbar.set_label("Fraction (%)")
    for row in range(destructive.shape[0]):
        for column in range(destructive.shape[1]):
            value = destructive[row, column]
            if value >= 0.05:
                axes[0].text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value > 30.0 else "0.2",
                )

    x = np.arange(len(ORDER))
    width = 0.36
    axes[1].bar(x - width / 2, ratios[:, 0], width, color="#0072B2", label=r"$\theta\!\to\!0$")
    axes[1].bar(x + width / 2, ratios[:, 1], width, color="#D55E00", label="within-atom permutation")
    axes[1].axhline(1.0, color="0.25", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, [DISPLAY[name] for name in ORDER], rotation=35, ha="right")
    axes[1].set_ylabel("Force MAE / native Force MAE")
    axes[1].set_title("Same-checkpoint interventions", loc="left", fontweight="semibold")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)

    global_shift_scaled = 1.0e4 * global_shift
    axes[2].barh(
        np.arange(len(ORDER)), global_shift_scaled, color="#009E73", height=0.68
    )
    axes[2].set_yticks(np.arange(len(ORDER)), [DISPLAY[name] for name in ORDER])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0.0, 1.15 * float(global_shift_scaled.max()))
    axes[2].set_xlabel(r"Mean $|\Delta F|$ ($10^{-4}$ meV $\AA^{-1}$)")
    axes[2].set_title("Common-phase shift", loc="left", fontweight="semibold")
    axes[2].spines[["top", "right", "left"]].set_visible(False)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(output / f"phase_interference_evidence.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def readable_mechanism_figure(
    payloads: dict[str, dict[str, Any]], output: Path
) -> None:
    """Make the mechanism legible without requiring the reader to decode a heatmap."""
    blue = "#246B9E"
    blue_light = "#70A6C7"
    orange = "#D76A2D"
    ink = "#252A30"
    grey = "#AAB2BA"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.55),
        gridspec_kw={"width_ratios": [1.35, 0.88, 1.35]},
        constrained_layout=True,
    )

    # (a) A direct view of the dataset/layer where explicit cancellation appears.
    payload = payloads["3bpa"]
    histogram = payload["phase_histograms"]
    edges = np.asarray(histogram["bin_edges_radians"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    selected = (
        ("layer1.L0.H0", blue_light, r"layer 1, $L=0$"),
        ("layer1.L1.H0", blue, r"layer 1, $L=1$"),
        ("layer1.L2.H0", ink, r"layer 1, $L=2$"),
    )
    axes[0].axvspan(-math.pi, -math.pi / 2, color=orange, alpha=0.11, lw=0)
    axes[0].axvspan(math.pi / 2, math.pi, color=orange, alpha=0.11, lw=0)
    axes[0].axvline(-math.pi / 2, color=orange, lw=0.8, ls="--")
    axes[0].axvline(math.pi / 2, color=orange, lw=0.8, ls="--")
    for key, color, label in selected:
        values = histogram["channels"][key]
        axes[0].plot(
            centers,
            values["relative_amplitude_weighted_probability"],
            color=color,
            lw=1.7,
            label=label,
        )
    axes[0].text(
        -2.35,
        0.47,
        "destructive\n$\\cos\\Delta\\theta<0$",
        color=orange,
        ha="center",
        va="top",
        fontsize=8,
    )
    axes[0].text(
        2.35,
        0.47,
        "destructive\n$\\cos\\Delta\\theta<0$",
        color=orange,
        ha="center",
        va="top",
        fontsize=8,
    )
    axes[0].set_xlim(-math.pi, math.pi)
    axes[0].set_ylim(0.0, 0.5)
    axes[0].set_xticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
    axes[0].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axes[0].set_xlabel(r"Relative phase $\Delta\theta_{jk}$")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("a  Learned relative phases", loc="left", fontweight="semibold")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    axes[0].spines[["top", "right"]].set_visible(False)

    # (b) Replace the dense heatmap by a direct, labeled comparison.
    maximum_destructive = []
    for name in ORDER:
        channels = payloads[name]["phase_histograms"]["channels"]
        maximum_destructive.append(
            100.0
            * max(
                value["summary"]["destructive_kernel_fraction"]
                for value in channels.values()
            )
        )
    y = np.arange(len(ORDER))
    bar_colors = [orange if value >= 10.0 else blue_light for value in maximum_destructive]
    axes[1].barh(y, maximum_destructive, color=bar_colors, height=0.62)
    axes[1].set_yticks(y, [DISPLAY[name] for name in ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 53.0)
    axes[1].set_xlabel("Largest destructive mass (%)")
    axes[1].set_title("b  Explicit cancellation", loc="left", fontweight="semibold")
    for row, value in enumerate(maximum_destructive):
        axes[1].text(
            max(value, 0.6) + 0.8,
            row,
            f"{value:.1f}%",
            ha="left",
            va="center",
            color=ink,
            fontsize=8,
        )
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)

    # (c) Same-checkpoint interventions: native CHORUS is the explicit reference.
    zero_ratio = []
    permute_ratio = []
    for name in ORDER:
        interventions = payloads[name]["interventions"]
        native = interventions["native"]["force_error"]["mae"]
        zero_ratio.append(interventions["zero"]["force_error"]["mae"] / native)
        permute_ratio.append(interventions["permute"]["force_error"]["mae"] / native)
    zero_ratio = np.asarray(zero_ratio)
    permute_ratio = np.asarray(permute_ratio)
    axes[2].axvline(1.0, color=grey, lw=1.0, ls="--", zorder=0)
    axes[2].scatter(
        zero_ratio,
        y - 0.13,
        s=38,
        color=blue,
        marker="o",
        label=r"set $\theta=0$",
        zorder=3,
    )
    axes[2].scatter(
        permute_ratio,
        y + 0.13,
        s=38,
        color=orange,
        marker="s",
        label="permute phases within atom",
        zorder=3,
    )
    for row, value in enumerate(zero_ratio):
        axes[2].annotate(
            f"×{value:.2g}",
            (value, row - 0.13),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.3,
            color=blue,
        )
    for row, value in enumerate(permute_ratio):
        axes[2].annotate(
            f"×{value:.2g}",
            (value, row + 0.13),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.3,
            color=orange,
        )
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.82, 105.0)
    axes[2].set_yticks(y, [DISPLAY[name] for name in ORDER])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Force MAE relative to native CHORUS")
    axes[2].set_title("c  Same-checkpoint intervention", loc="left", fontweight="semibold")
    axes[2].legend(frameon=False, loc="lower right")
    axes[2].spines[["top", "right", "left"]].set_visible(False)
    axes[2].tick_params(axis="y", length=0)

    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(output / f"phase_interference_evidence.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _terminal_channels(payload: dict[str, Any]) -> dict[str, Any]:
    """Return channels from the deepest captured interaction only."""
    channels = payload["phase_histograms"]["channels"]
    layers = [int(key.split(".")[0].replace("layer", "")) for key in channels]
    terminal = max(layers)
    return {key: value for key, value in channels.items() if key.startswith(f"layer{terminal}.")}


def _intervention_ratios(payload: dict[str, Any]) -> tuple[float, float]:
    interventions = payload["interventions"]
    native = interventions["native"]["force_error"]["mae"]
    return (
        interventions["zero"]["force_error"]["mae"] / native,
        interventions["permute"]["force_error"]["mae"] / native,
    )


def scope_comparison_figure(
    final_payloads: dict[str, dict[str, Any]],
    persistent_payloads: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    """Compact paper figure comparing Final and Persistent phase mechanisms."""
    blue = "#205381"       # manuscript chorusblue
    orange = "#DC7520"     # manuscript chargedorange
    ink = "#25282C"
    mid_grey = "#7D858D"
    light_grey = "#D7DCE0"
    destructive_fill = "#F8E8DB"

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.15, 5.05), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.965,
        wspace=0.31,
        hspace=0.34,
        height_ratios=(0.96, 1.04),
    )
    ax_final = fig.add_subplot(grid[0, 0])
    ax_persistent = fig.add_subplot(grid[0, 1], sharex=ax_final, sharey=ax_final)
    ax_mass = fig.add_subplot(grid[1, 0])
    ax_intervention = fig.add_subplot(grid[1, 1])

    # Representative relative-phase distributions.  Terminal-layer channels are
    # matched between scopes, avoiding a layer-count advantage for Persistent.
    hist_styles = (
        ("L0", "#8AAFC7", (0, (3.0, 1.5))),
        ("L1", blue, "-"),
        ("L2", ink, "-"),
    )
    peak = 0.0
    for axis, payload, title in (
        (ax_final, final_payloads["3bpa"], "a  CHORUS-Final (3BPA)"),
        (ax_persistent, persistent_payloads["3bpa"], "b  CHORUS-Persistent (3BPA)"),
    ):
        histogram = payload["phase_histograms"]
        channels = _terminal_channels(payload)
        edges = np.asarray(histogram["bin_edges_radians"], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.axvspan(-math.pi, -math.pi / 2, color=destructive_fill, lw=0)
        axis.axvspan(math.pi / 2, math.pi, color=destructive_fill, lw=0)
        axis.axvline(-math.pi / 2, color=orange, lw=0.72, ls=(0, (2.5, 2.0)))
        axis.axvline(math.pi / 2, color=orange, lw=0.72, ls=(0, (2.5, 2.0)))
        for angular, color, line_style in hist_styles:
            matching = [key for key in channels if f".{angular}." in key]
            if not matching:
                continue
            values = np.asarray(
                channels[matching[0]]["relative_amplitude_weighted_probability"],
                dtype=float,
            )
            peak = max(peak, float(values.max()))
            axis.plot(
                centers,
                values,
                color=color,
                lw=1.45 if angular != "L2" else 1.65,
                ls=line_style,
                label=f"${angular[0]}={angular[1:]}$",
            )
        axis.set_title(title, loc="left", color=ink, fontweight="bold", pad=4)
        axis.set_xlim(-math.pi, math.pi)
        axis.set_xticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
        axis.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        axis.set_xlabel(r"Relative phase $\Delta\theta_{jk}$")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=2.8, width=0.65)
    y_top = max(0.10, math.ceil((peak * 1.17) / 0.05) * 0.05)
    ax_final.set_ylim(0.0, y_top)
    ax_final.set_ylabel("Amplitude-weighted probability")
    ax_persistent.tick_params(labelleft=False)
    ax_persistent.legend(
        frameon=False,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        handlelength=1.7,
        columnspacing=0.9,
        borderaxespad=0.25,
    )
    ax_final.text(
        0.095,
        0.955,
        "destructive\npairs",
        transform=ax_final.transAxes,
        color=orange,
        fontsize=7.2,
        ha="center",
        va="top",
        linespacing=0.9,
    )
    ax_persistent.text(
        0.095,
        0.955,
        "destructive\npairs",
        transform=ax_persistent.transAxes,
        color=orange,
        fontsize=7.2,
        ha="center",
        va="top",
        linespacing=0.9,
    )

    # Maximum destructive mass across the active phase channels in each scope.
    y = np.arange(len(ORDER))
    final_mass = []
    persistent_mass = []
    for name in ORDER:
        for payloads, target in (
            (final_payloads, final_mass),
            (persistent_payloads, persistent_mass),
        ):
            channels = payloads[name]["phase_histograms"]["channels"]
            target.append(
                100.0
                * max(
                    value["summary"]["destructive_kernel_fraction"]
                    for value in channels.values()
                )
            )
    bar_height = 0.26
    for row, (final_value, persistent_value) in enumerate(zip(final_mass, persistent_mass)):
        for value, offset, scope_label in (
            (final_value, -bar_height / 1.7, "F"),
            (persistent_value, bar_height / 1.7, "P"),
        ):
            constructive = 100.0 - value
            ax_mass.barh(
                row + offset,
                constructive,
                height=bar_height,
                color="#E4EBF0",
                edgecolor="white",
                linewidth=0.4,
            )
            ax_mass.barh(
                row + offset,
                value,
                left=constructive,
                height=bar_height,
                color=orange,
                edgecolor="white",
                linewidth=0.4,
            )
            ax_mass.text(
                -3.6,
                row + offset,
                scope_label,
                ha="center",
                va="center",
                fontsize=6.9,
                color=blue if scope_label == "F" else ink,
                fontweight="bold",
            )
            ax_mass.text(
                101.2,
                row + offset,
                f"{value:.1f}",
                ha="left",
                va="center",
                fontsize=6.8,
                color=orange if value >= 0.05 else mid_grey,
            )
    ax_mass.set_yticks(y, [FIG_DISPLAY[name] for name in ORDER])
    ax_mass.invert_yaxis()
    ax_mass.set_xlim(-6.5, 108.0)
    ax_mass.set_xticks([0, 25, 50, 75, 100])
    ax_mass.set_xlabel("Share of absolute pair-kernel mass (%)")
    ax_mass.set_title("c  Pair-kernel composition", loc="left", color=ink, fontweight="bold", pad=4)
    ax_mass.spines[["top", "right", "left"]].set_visible(False)
    ax_mass.tick_params(axis="y", length=0)
    ax_mass.grid(axis="x", color=light_grey, lw=0.55, zorder=0)

    # Same-checkpoint intervention.  Colour encodes intervention and marker fill
    # encodes scope, so the comparison remains readable in greyscale.
    final_ratios = np.asarray([_intervention_ratios(final_payloads[name]) for name in ORDER])
    persistent_ratios = np.asarray(
        [_intervention_ratios(persistent_payloads[name]) for name in ORDER]
    )
    ax_intervention.axvline(1.0, color=mid_grey, lw=0.8, ls=(0, (3.0, 2.0)), zorder=0)
    offsets = (-0.18, -0.06, 0.06, 0.18)
    series = (
        (final_ratios[:, 0], offsets[0], blue, "o", "Final, set phase to zero", "white"),
        (final_ratios[:, 1], offsets[1], orange, "s", "Final, permute phase", "white"),
        (persistent_ratios[:, 0], offsets[2], blue, "o", "Persistent, set phase to zero", blue),
        (persistent_ratios[:, 1], offsets[3], orange, "s", "Persistent, permute phase", orange),
    )
    for values, offset, color, marker, label, face in series:
        ax_intervention.scatter(
            values,
            y + offset,
            s=25,
            marker=marker,
            facecolors=face,
            edgecolors=color,
            linewidths=0.9,
            label=label,
            zorder=3,
        )
    ax_intervention.set_xscale("log")
    maximum_ratio = float(max(final_ratios.max(), persistent_ratios.max()))
    ax_intervention.set_xlim(0.86, max(2.0, maximum_ratio * 1.32))
    ax_intervention.set_yticks(y, [FIG_DISPLAY[name] for name in ORDER])
    ax_intervention.invert_yaxis()
    ax_intervention.set_xlabel("Force MAE / native Force MAE")
    ax_intervention.set_title("d  Phase intervention", loc="left", color=ink, fontweight="bold", pad=4)
    ax_intervention.spines[["top", "right", "left"]].set_visible(False)
    ax_intervention.tick_params(axis="y", length=0)
    ax_intervention.grid(axis="x", which="major", color=light_grey, lw=0.55, zorder=0)
    ax_intervention.text(
        0.99,
        0.965,
        r"$\circ$ Final   $\bullet$ Persistent   blue: $\theta\!\to\!0$   orange: permuted",
        transform=ax_intervention.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
        color=mid_grey,
    )
    for values, offset, color, _marker, _label, _face in series:
        for row, value in enumerate(values):
            if value >= 1.25 or value <= 0.92:
                ax_intervention.annotate(
                    f"×{value:.1f}" if value < 10 else f"×{value:.0f}",
                    (value, row + offset),
                    xytext=(3.5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=6.6,
                )

    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 360} if suffix == "png" else {}
        fig.savefig(
            output / f"phase_scope_mechanism_evidence.{suffix}",
            bbox_inches="tight",
            pad_inches=0.025,
            **kwargs,
        )
    plt.close(fig)


def write_scope_comparison_summary(
    final_payloads: dict[str, dict[str, Any]],
    persistent_payloads: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    rows: dict[str, Any] = {}
    for name in ORDER:
        rows[name] = {"label": DISPLAY[name]}
        for scope, payloads in (
            ("final", final_payloads),
            ("persistent", persistent_payloads),
        ):
            payload = payloads[name]
            channels = payload["phase_histograms"]["channels"]
            zero_ratio, permute_ratio = _intervention_ratios(payload)
            rows[name][scope] = {
                "checkpoint": payload["checkpoint"],
                "histogram_validation_batches": payload["num_batches"],
                "intervention_validation_batches": payload["num_intervention_batches"],
                "largest_destructive_kernel_fraction": max(
                    value["summary"]["destructive_kernel_fraction"]
                    for value in channels.values()
                ),
                "zero_phase_force_mae_ratio": zero_ratio,
                "within_atom_phase_permutation_force_mae_ratio": permute_ratio,
            }
    payload = {
        "selection": "validation Force-MAE-selected R16 checkpoint for each dataset and scope",
        "histogram_scope": "complete validation split",
        "intervention_scope": "leading validation batches; test data are not used",
        "datasets": rows,
    }
    (output / "phase_scope_mechanism_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def build_summary(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    datasets = {}
    for name in ORDER:
        payload = payloads[name]
        channels = payload["phase_histograms"]["channels"]
        destructive = [
            value["summary"]["destructive_kernel_fraction"]
            for value in channels.values()
        ]
        native_force = payload["interventions"]["native"]["force_error"]["mae"]
        datasets[name] = {
            "label": DISPLAY[name],
            "validation_batches": payload["num_batches"],
            "intervention_batches": payload["num_intervention_batches"],
            "relative_phase_pair_count_per_channel": int(
                next(iter(channels.values()))["summary"]["pair_count"]
            ),
            "destructive_kernel_fraction_min": min(destructive),
            "destructive_kernel_fraction_max": max(destructive),
            "zero_phase_force_mae_change_percent": percent_change(
                payload["interventions"]["zero"]["force_error"]["mae"], native_force
            ),
            "permuted_phase_force_mae_change_percent": percent_change(
                payload["interventions"]["permute"]["force_error"]["mae"], native_force
            ),
            "global_shift_force_prediction_change_mev_per_angstrom": 1000.0
            * payload["interventions"]["global-shift"]["force_change"]["mae"],
        }
    return {
        "selection": "validation Force-MAE-selected checkpoint for each dataset",
        "histogram_scope": "complete validation split",
        "intervention_scope": "leading validation batches; test data are not used",
        "datasets": datasets,
    }


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# CHORUS phase-interference diagnostics",
        "",
        "Relative-phase histograms use the complete validation split. Intervention metrics use the leading validation batches of the same split and the same trained checkpoint.",
        "",
        "| Dataset | Relative pairs / channel | Destructive mass range | Force MAE change, zero phase | Force MAE change, local permutation | Common-shift mean |ΔF| |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for name in ORDER:
        row = summary["datasets"][name]
        lines.append(
            "| {label} | {pairs:,} | {low:.1f}–{high:.1f}% | {zero:+.1f}% | {perm:+.1f}% | {shift:.4f} meV Å⁻¹ |".format(
                label=row["label"],
                pairs=row["relative_phase_pair_count_per_channel"],
                low=100.0 * row["destructive_kernel_fraction_min"],
                high=100.0 * row["destructive_kernel_fraction_max"],
                zero=row["zero_phase_force_mae_change_percent"],
                perm=row["permuted_phase_force_mae_change_percent"],
                shift=row["global_shift_force_prediction_change_mev_per_angstrom"],
            )
        )
    lines.extend(
        [
            "",
            "Absolute-phase histograms are descriptive because a common U(1) shift changes θ but leaves the Hermitian density invariant. The relative-phase histograms and the intervention tests carry the mechanistic interpretation.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads(input_root)
    summary = build_summary(payloads)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, output)
    phase_small_multiples(payloads, output)
    readable_mechanism_figure(payloads, output)
    if args.final_root is not None:
        final_payloads = load_payloads(Path(args.final_root))
        write_scope_comparison_summary(final_payloads, payloads, output)
        scope_comparison_figure(final_payloads, payloads, output)


if __name__ == "__main__":
    main()
