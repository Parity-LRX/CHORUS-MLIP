#!/usr/bin/env python3
"""Summarize CHORUS phase organization across MACE-ICTC and NequIP.

The main figure deliberately carries only the mechanism-defining evidence:
a representative relative-phase distribution and same-checkpoint phase
interventions for each base model.  A full five-system histogram plate and a
machine-readable summary are emitted for the appendix.
"""

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
LABEL = {
    "buckyball": "Buckyball",
    "mal": "MAL",
    "sti": "STI",
    "3bpa": "3BPA",
    "t1x": "T1x-50k",
}
NEQUIP_NAME = {
    "buckyball": "md22_buckyball",
    "mal": "xxmd_mal",
    "sti": "xxmd_sti",
    "3bpa": "3bpa",
    "t1x": "transition1x",
}

BLUE = "#205381"
ORANGE = "#DC7520"
INK = "#25282C"
MID_GREY = "#7D858D"
LIGHT_GREY = "#D7DCE0"
DESTRUCTIVE = "#F8E8DB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-final-root", required=True)
    parser.add_argument("--mace-persistent-root", required=True)
    parser.add_argument("--nequip-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_mace(root: Path) -> dict[str, dict[str, Any]]:
    return {
        name: load_json(root / name / "phase_interference.json")
        for name in ORDER
    }


def load_nequip(
    root: Path, depth: int, scope: str
) -> dict[str, dict[str, Any]]:
    return {
        name: load_json(
            root
            / f"depth{depth}"
            / scope
            / NEQUIP_NAME[name]
            / "phase_interference.json"
        )
        for name in ORDER
    }


def pooled_relative_histogram(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    histogram = payload["phase_histograms"]
    edges = np.asarray(histogram["bin_edges_radians"], dtype=float)
    pooled = np.zeros(edges.size - 1, dtype=float)
    for channel in histogram["channels"].values():
        probability = np.asarray(
            channel["relative_amplitude_weighted_probability"], dtype=float
        )
        weight = float(channel["summary"]["pair_amplitude_sum"])
        pooled += probability * weight
    total = float(pooled.sum())
    if total > 0.0:
        pooled /= total
    return 0.5 * (edges[:-1] + edges[1:]), pooled


def intervention_ratios(payload: dict[str, Any]) -> tuple[float, float]:
    interventions = payload["interventions"]
    native = float(interventions["native"]["force_error"]["mae"])
    return (
        float(interventions["zero"]["force_error"]["mae"]) / native,
        float(interventions["permute"]["force_error"]["mae"]) / native,
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.1,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_histogram_axis(axis: plt.Axes) -> None:
    axis.axvspan(-math.pi, -math.pi / 2, color=DESTRUCTIVE, lw=0, zorder=0)
    axis.axvspan(math.pi / 2, math.pi, color=DESTRUCTIVE, lw=0, zorder=0)
    axis.axvline(-math.pi / 2, color=ORANGE, lw=0.65, ls=(0, (2.4, 2.0)))
    axis.axvline(math.pi / 2, color=ORANGE, lw=0.65, ls=(0, (2.4, 2.0)))
    axis.set_xlim(-math.pi, math.pi)
    axis.set_xticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
    axis.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=2.7, width=0.65)


def draw_histogram_ridges(
    axis: plt.Axes,
    final: dict[str, dict[str, Any]],
    persistent: dict[str, dict[str, Any]],
    title: str,
) -> None:
    style_histogram_axis(axis)
    for row, name in enumerate(ORDER):
        axis.axhline(row, color=LIGHT_GREY, lw=0.45, zorder=0)
        for payloads, color, line_style, label in (
            (final, BLUE, (0, (3.0, 1.8)), "Final"),
            (persistent, ORANGE, "-", "Persistent"),
        ):
            centers, probability = pooled_relative_histogram(payloads[name])
            maximum = max(float(probability.max()), 1.0e-30)
            ridge = row - 0.72 * probability / maximum
            axis.plot(
                centers,
                ridge,
                color=color,
                ls=line_style,
                lw=1.45,
                label=label if row == 0 else None,
                zorder=3,
            )
    axis.set_ylim(len(ORDER) - 0.45, -0.82)
    axis.set_yticks(np.arange(len(ORDER)), [LABEL[name] for name in ORDER])
    axis.set_xlabel(r"Relative phase $\Delta\theta_{jk}$")
    axis.set_ylabel("Validation system")
    axis.set_title(title, loc="left", color=INK, fontweight="bold", pad=4)
    axis.legend(frameon=False, loc="upper right", handlelength=2.2)
    axis.text(
        0.99,
        0.02,
        "each curve peak-normalized",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=MID_GREY,
    )


def draw_interventions(
    axis: plt.Axes,
    final: dict[str, dict[str, Any]],
    persistent: dict[str, dict[str, Any]],
    title: str,
) -> None:
    x = np.arange(len(ORDER), dtype=float)
    final_values = np.asarray([intervention_ratios(final[name]) for name in ORDER])
    persistent_values = np.asarray(
        [intervention_ratios(persistent[name]) for name in ORDER]
    )
    axis.axhline(1.0, color=MID_GREY, lw=0.8, ls=(0, (3.0, 2.0)), zorder=0)
    series = (
        (final_values[:, 0], -0.18, BLUE, "o", "white"),
        (final_values[:, 1], -0.06, ORANGE, "s", "white"),
        (persistent_values[:, 0], 0.06, BLUE, "o", BLUE),
        (persistent_values[:, 1], 0.18, ORANGE, "s", ORANGE),
    )
    for values, offset, color, marker, face in series:
        axis.scatter(
            x + offset,
            values,
            marker=marker,
            s=24,
            facecolors=face,
            edgecolors=color,
            linewidths=0.9,
            zorder=3,
        )
    axis.set_yscale("log")
    lower = min(0.82, 0.92 * float(min(final_values.min(), persistent_values.min())))
    upper = max(2.0, 1.28 * float(max(final_values.max(), persistent_values.max())))
    axis.set_ylim(lower, upper)
    axis.set_xticks(x, [LABEL[name] for name in ORDER], rotation=24, ha="right")
    axis.set_ylabel("Force MAE / native Force MAE")
    axis.set_title(title, loc="left", color=INK, fontweight="bold", pad=4)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", which="major", color=LIGHT_GREY, lw=0.55, zorder=0)
    axis.tick_params(direction="out", length=2.7, width=0.65)


def main_figure(
    mace_final: dict[str, dict[str, Any]],
    mace_persistent: dict[str, dict[str, Any]],
    nequip_final: dict[str, dict[str, Any]],
    nequip_persistent: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 4.25),
        gridspec_kw={"height_ratios": (0.93, 1.07)},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.97, wspace=0.31, hspace=0.39)
    draw_histogram_ridges(
        axes[0, 0],
        mace_final,
        mace_persistent,
        "a  MACE-ICTC: relative phase",
    )
    draw_histogram_ridges(
        axes[0, 1],
        nequip_final,
        nequip_persistent,
        "b  NequIP-SH: relative phase",
    )
    for axis in axes[0]:
        axis.text(
            0.105,
            0.94,
            r"$\cos\Delta\theta<0$",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=6.8,
            color=ORANGE,
        )
    draw_interventions(
        axes[1, 0],
        mace_final,
        mace_persistent,
        "c  MACE-ICTC: same-checkpoint intervention",
    )
    draw_interventions(
        axes[1, 1],
        nequip_final,
        nequip_persistent,
        "d  NequIP-SH: same-checkpoint intervention",
    )
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 360} if suffix == "png" else {}
        fig.savefig(
            output / f"cross_backbone_phase_mechanism.{suffix}",
            bbox_inches="tight",
            pad_inches=0.025,
            **kwargs,
        )
    plt.close(fig)


def appendix_histograms(
    mace_final: dict[str, dict[str, Any]],
    mace_persistent: dict[str, dict[str, Any]],
    nequip_final: dict[str, dict[str, Any]],
    nequip_persistent: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        len(ORDER),
        2,
        figsize=(7.15, 8.15),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.07, top=0.965, wspace=0.28, hspace=0.27)
    for row, name in enumerate(ORDER):
        for column, (final, persistent, backbone) in enumerate(
            (
                (mace_final, mace_persistent, "MACE-ICTC"),
                (nequip_final, nequip_persistent, "NequIP-SH (3 layers)"),
            )
        ):
            axis = axes[row, column]
            style_histogram_axis(axis)
            for payloads, color, line_style, label in (
                (final, BLUE, (0, (3.0, 1.8)), "Final"),
                (persistent, ORANGE, "-", "Persistent"),
            ):
                centers, probability = pooled_relative_histogram(payloads[name])
                maximum = max(float(probability.max()), 1.0e-30)
                axis.plot(
                    centers,
                    probability / maximum,
                    color=color,
                    ls=line_style,
                    lw=1.45,
                    label=label,
                )
            axis.set_ylim(0.0, 1.06)
            axis.set_ylabel("Peak-normalized")
            axis.set_title(
                f"{LABEL[name]}  |  {backbone}",
                loc="left",
                fontsize=8.2,
                color=INK,
                fontweight="bold",
                pad=3,
            )
            if row < len(ORDER) - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(r"Relative phase $\Delta\theta_{jk}$")
            if row == 0:
                axis.legend(frameon=False, loc="upper right", ncol=2)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 360} if suffix == "png" else {}
        fig.savefig(
            output / f"cross_backbone_phase_histograms_appendix.{suffix}",
            bbox_inches="tight",
            pad_inches=0.025,
            **kwargs,
        )
    plt.close(fig)


def setting_summary(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    destructive = []
    zero = []
    permute = []
    global_shift = []
    for name in ORDER:
        payload = payloads[name]
        destructive.append(
            max(
                float(channel["summary"]["destructive_kernel_fraction"])
                for channel in payload["phase_histograms"]["channels"].values()
            )
        )
        zero_ratio, permute_ratio = intervention_ratios(payload)
        zero.append(zero_ratio)
        permute.append(permute_ratio)
        global_shift.append(
            1000.0
            * float(
                payload["interventions"]["global-shift"]["force_change"]["mae"]
            )
        )
    return {
        "largest_destructive_kernel_fraction_range": [min(destructive), max(destructive)],
        "zero_phase_force_mae_ratio_range": [min(zero), max(zero)],
        "phase_permutation_force_mae_ratio_range": [min(permute), max(permute)],
        "maximum_common_shift_force_change_mev_per_angstrom": max(global_shift),
    }


def write_summary(
    all_settings: dict[str, dict[str, dict[str, Any]]], output: Path
) -> None:
    payload = {
        "selection": "validation Force-MAE-selected checkpoint",
        "histograms": "validation frames; MACE complete split and NequIP at most 256 evenly spaced frames",
        "interventions": "same checkpoint; leading validation batches; test data not used",
        "interpretation": (
            "Relative phase controls coherent cross-neighbour coupling. Negative cosine terms "
            "are one regime, not a requirement: non-uniform constructive phase organization "
            "is also causally identified by zero-phase and within-atom permutation interventions."
        ),
        "settings": {
            key: setting_summary(value) for key, value in all_settings.items()
        },
    }
    (output / "cross_backbone_phase_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Cross-backbone CHORUS phase diagnostics",
        "",
        "| Setting | Destructive-kernel range | Zero-phase Force-MAE ratio | Permuted-phase Force-MAE ratio | Max common-shift |ΔF| |",
        "|:--|--:|--:|--:|--:|",
    ]
    for key, value in payload["settings"].items():
        destructive = value["largest_destructive_kernel_fraction_range"]
        zero = value["zero_phase_force_mae_ratio_range"]
        permute = value["phase_permutation_force_mae_ratio_range"]
        lines.append(
            f"| {key} | {100*destructive[0]:.2f}--{100*destructive[1]:.2f}% "
            f"| {zero[0]:.3f}--{zero[1]:.3f}× | {permute[0]:.3f}--{permute[1]:.3f}× "
            f"| {value['maximum_common_shift_force_change_mev_per_angstrom']:.4g} meV Å⁻¹ |"
        )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    configure_style()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mace_final = load_mace(Path(args.mace_final_root))
    mace_persistent = load_mace(Path(args.mace_persistent_root))
    nequip2_final = load_nequip(Path(args.nequip_root), 2, "final")
    nequip2_persistent = load_nequip(Path(args.nequip_root), 2, "persistent")
    nequip3_final = load_nequip(Path(args.nequip_root), 3, "final")
    nequip3_persistent = load_nequip(Path(args.nequip_root), 3, "persistent")

    main_figure(
        mace_final,
        mace_persistent,
        nequip3_final,
        nequip3_persistent,
        output,
    )
    appendix_histograms(
        mace_final,
        mace_persistent,
        nequip3_final,
        nequip3_persistent,
        output,
    )
    write_summary(
        {
            "MACE-ICTC Final": mace_final,
            "MACE-ICTC Persistent": mace_persistent,
            "NequIP-SH 2L Final": nequip2_final,
            "NequIP-SH 2L Persistent": nequip2_persistent,
            "NequIP-SH 3L Final": nequip3_final,
            "NequIP-SH 3L Persistent": nequip3_persistent,
        },
        output,
    )


if __name__ == "__main__":
    main()
