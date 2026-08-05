#!/usr/bin/env python3
"""Plot strict-fp32 CHORUS–TECE atom-count scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402


BLUE = "#2F6B9A"
BLUE_DARK = "#1E4667"
ORANGE = "#D9772B"
INK = "#22272E"
MUTED = "#667085"
GRID = "#D9DEE5"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_record(path: Path) -> dict:
    record = json.loads(path.read_text())
    protocol = record["protocol"]
    if (
        protocol["float32_matmul_precision"] != "highest"
        or protocol["allow_tf32"]
        or protocol["cudnn_allow_tf32"]
    ):
        raise ValueError(f"{path} is not a strict-fp32 result")
    if any(row["status"] != "ok" for row in record["rows"]):
        raise ValueError(f"{path} contains an incomplete row")
    return record


def save_all(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        output = out_stem.with_suffix(suffix)
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        print(f"wrote {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out_stem", type=Path)
    args = parser.parse_args()

    chorus = load_record(args.raw_dir / "chorus.json")
    eager = load_record(args.raw_dir / "tece_eager.json")
    cue = load_record(args.raw_dir / "tece_cue.json")

    atoms = np.asarray([row["natoms"] for row in chorus["rows"]])
    edges = np.asarray([row["nedges"] for row in chorus["rows"]])
    chorus_ms = np.asarray([row["median_ms"] for row in chorus["rows"]])
    eager_ms = np.asarray([row["median_ms"] for row in eager["rows"]])
    cue_ms = np.asarray([row["median_ms"] for row in cue["rows"]])
    if not (
        atoms.tolist() == [row["natoms"] for row in eager["rows"]]
        == [row["natoms"] for row in cue["rows"]]
    ):
        raise ValueError("atom-count grids do not match")
    if not (
        edges.tolist() == [row["nedges"] for row in eager["rows"]]
        == [row["nedges"] for row in cue["rows"]]
    ):
        raise ValueError("edge-count grids do not match")

    speedup = eager_ms / chorus_ms
    setup_style()
    fig, (ax_latency, ax_speedup) = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.25),
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.32},
    )

    ax_latency.plot(
        atoms,
        chorus_ms,
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLUE_DARK,
        markeredgewidth=0.8,
        linewidth=1.8,
        markersize=4.8,
        label="CHORUS · MakeFX",
        zorder=3,
    )
    ax_latency.plot(
        atoms,
        eager_ms,
        color=ORANGE,
        marker="s",
        markerfacecolor=ORANGE,
        markeredgecolor="#8D4818",
        markeredgewidth=0.8,
        linewidth=1.55,
        linestyle="--",
        markersize=4.5,
        label="TECE · eager",
        zorder=2,
    )
    ax_latency.plot(
        atoms,
        cue_ms,
        color=ORANGE,
        marker="^",
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=1.0,
        linewidth=1.35,
        linestyle=":",
        markersize=4.8,
        label="TECE · CUE first layer",
        zorder=2,
    )
    ax_latency.set_xscale("log", base=2)
    ax_latency.set_yscale("log")
    ax_latency.set_xlabel("Atoms")
    ax_latency.set_ylabel("Energy + force latency (ms)")
    ax_latency.set_title("(a) Model-core latency", loc="left", fontweight="semibold")
    ax_latency.grid(True, which="major")
    ax_latency.grid(False, which="minor")
    ax_latency.spines[["top", "right"]].set_visible(False)
    ax_latency.legend(loc="upper left", frameon=False, handlelength=2.5)

    ax_speedup.plot(
        atoms,
        speedup,
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLUE_DARK,
        markeredgewidth=0.8,
        linewidth=1.8,
        markersize=4.8,
        zorder=3,
    )
    ax_speedup.fill_between(
        atoms,
        1.0,
        speedup,
        color=BLUE,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    ax_speedup.axhline(1.0, color=MUTED, linewidth=0.9, linestyle="--")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_ylim(0.75, 7.15)
    ax_speedup.set_yticks(np.arange(1, 8))
    ax_speedup.set_xlabel("Atoms")
    ax_speedup.set_ylabel("Speedup vs TECE eager (×)")
    ax_speedup.set_title("(b) CHORUS speedup", loc="left", fontweight="semibold")
    ax_speedup.grid(True, axis="y")
    ax_speedup.grid(False, axis="x")
    ax_speedup.spines[["top", "right"]].set_visible(False)
    ax_speedup.annotate(
        f"{speedup[0]:.2f}×",
        (atoms[0], speedup[0]),
        xytext=(7, -2),
        textcoords="offset points",
        color=BLUE_DARK,
        fontsize=8,
        fontweight="semibold",
    )
    ax_speedup.annotate(
        f"{speedup[-1]:.2f}×",
        (atoms[-1], speedup[-1]),
        xytext=(-7, 8),
        textcoords="offset points",
        ha="right",
        color=BLUE_DARK,
        fontsize=8,
        fontweight="semibold",
    )

    shown_ticks = np.asarray([64, 216, 512, 1000, 4096])
    for axis in (ax_latency, ax_speedup):
        axis.set_xticks(shown_ticks)
        axis.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{int(value):,}")
        )
        axis.xaxis.set_minor_locator(mticker.NullLocator())
    ax_latency.set_yticks([5, 10, 20, 50, 100])
    ax_latency.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    ax_latency.yaxis.set_minor_locator(mticker.NullLocator())

    fig.suptitle(
        "CHORUS and TECE atom-count scaling on RTX 4090",
        x=0.08,
        y=1.045,
        ha="left",
        fontsize=12,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.08,
        0.985,
        "Strict FP32 (TF32 disabled) · ~18 directed neighbors/atom · graph construction excluded",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    fig.text(
        0.08,
        -0.035,
        "Median of 5–15 timed calls after 3 warm-ups. Periodic jittered carbon; 5 Å cutoff. "
        "CHORUS uses atom-count-specific MakeFX buckets.",
        ha="left",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )
    save_all(fig, args.out_stem)


if __name__ == "__main__":
    main()
