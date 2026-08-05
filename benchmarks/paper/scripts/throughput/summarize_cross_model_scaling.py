#!/usr/bin/env python3
"""Merge atom-scaling runs and emit paper tables plus a static figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SERIES = {
    "Native MACE (CuEq)": ("native_mace.json",),
    "MACE-ICTC": ("mace_ictc_baseline.json", "mace_ictc_baseline_large.json"),
    "CHORUS R8": ("chorus_r8.json",),
    "CHORUS R16": ("chorus_r16.json", "chorus_r16_large.json"),
    "CHORUS R32": ("chorus_r32.json",),
    "DPA-4 C32 (FP32 compiled)": ("dpa4_c32_compiled_fp32.json",),
    "DPA-4 C48 (FP32 compiled)": ("dpa4_c48_compiled_fp32.json",),
    "TECE C36 (OpenEq)": ("tece_c36.json",),
    "TECE C48 (OpenEq)": ("tece_c48.json",),
}

# Keep the complete rank sweep in the machine-readable tables, but plot only
# the representative rank selected by the accuracy--cost ablation.
FIGURE_SERIES = (
    "Native MACE (CuEq)",
    "MACE-ICTC",
    "CHORUS R16",
    "DPA-4 C32 (FP32 compiled)",
    "DPA-4 C48 (FP32 compiled)",
    "TECE C36 (OpenEq)",
    "TECE C48 (OpenEq)",
)

DISPLAY_LABELS = {
    "Native MACE (CuEq)": "Native MACE · CuEq",
    "MACE-ICTC": "MACE-ICTC",
    "CHORUS R8": "CHORUS R8",
    "CHORUS R16": "CHORUS R16",
    "CHORUS R32": "CHORUS R32",
    "DPA-4 C32 (FP32 compiled)": "DPA-4 C32 · FP32",
    "DPA-4 C48 (FP32 compiled)": "DPA-4 C48 · FP32",
    "TECE C36 (OpenEq)": "TECE C36 · OpenEq",
    "TECE C48 (OpenEq)": "TECE C48 · OpenEq",
}

STYLES = {
    "Native MACE (CuEq)": {
        "color": "#A7A7A7",
        "marker": "o",
        "markerfacecolor": "white",
        "linestyle": ":",
    },
    "MACE-ICTC": {"color": "#424242", "marker": "s", "linestyle": "--"},
    "CHORUS R8": {
        "color": "#9DBBE0",
        "marker": "o",
        "markerfacecolor": "white",
        "linestyle": "-",
    },
    "CHORUS R16": {"color": "#4B7FB5", "marker": "s", "linestyle": "-"},
    "CHORUS R32": {"color": "#173F68", "marker": "^", "linestyle": "-"},
    "DPA-4 C32 (FP32 compiled)": {
        "color": "#D89A42",
        "marker": "o",
        "markerfacecolor": "white",
        "linestyle": "--",
    },
    "DPA-4 C48 (FP32 compiled)": {
        "color": "#9B5A16",
        "marker": "s",
        "linestyle": "--",
    },
    "TECE C36 (OpenEq)": {
        "color": "#98A989",
        "marker": "o",
        "markerfacecolor": "white",
        "linestyle": "-.",
    },
    "TECE C48 (OpenEq)": {
        "color": "#4E6740",
        "marker": "s",
        "linestyle": "-.",
    },
}


def load_rows(source: Path, names: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    for name in names:
        path = source / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if not metadata:
            metadata = payload
        for row in payload["rows"]:
            merged[(row["task"], int(row["natoms"]))] = row
    return [merged[key] for key in sorted(merged)], metadata


def fmt_rate(row: dict[str, Any] | None) -> str:
    if row is None:
        return "—"
    if row["status"] != "ok":
        return row["status"].upper()
    return f"{row['atoms_per_second']:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows_by_series: dict[str, list[dict[str, Any]]] = {}
    metadata_by_series: dict[str, dict] = {}
    flat_rows: list[dict[str, Any]] = []
    for label, files in SERIES.items():
        rows, metadata = load_rows(args.source, files)
        rows_by_series[label] = rows
        metadata_by_series[label] = metadata
        flat_rows.extend({"model": label, **row} for row in rows)

    fields = sorted({key for row in flat_rows for key in row})
    with (args.output / "throughput_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    (args.output / "throughput_merged.json").write_text(
        json.dumps(
            {
                "series": metadata_by_series,
                "rows": flat_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    sizes = sorted({int(row["natoms"]) for row in flat_rows})
    lines = [
        "# Cross-model atom-scaling benchmark",
        "",
        "One RTX 4090; every model uses strict FP32, with TF32, AMP, and EMA disabled. "
        "Inference computes energy and conservative forces. Training "
        "includes the energy-force loss, backward pass, and optimizer update. "
        "Compile/preparation time is excluded from steady-state throughput.",
        "",
        "Native MACE uses CuEq-only tensor products; TECE uses OpenEquivariance. "
        "MACE-family and TECE graph construction is excluded. DPA-4 uses its standard "
        "compiled model interface, whose internal neighbor-list work remains inside "
        "the timed call; this interface difference must be retained as a caveat.",
        "",
        "## Configurations",
        "",
        "| Model | Trainable parameters | Backend |",
        "|---|---:|---|",
    ]
    for label in SERIES:
        meta = metadata_by_series[label]
        parameters = meta.get("parameters")
        backend = meta.get("backend", "—")
        lines.append(
            f"| {DISPLAY_LABELS[label]} | {parameters:,} | {backend} |"
            if parameters is not None
            else f"| {DISPLAY_LABELS[label]} | — | {backend} |"
        )

    lines.extend(
        [
            "",
            "## Compilation and peak memory",
            "",
            "| Model | Compile at 32 atoms, infer (s) | Compile at 32 atoms, train (s) | "
            "Peak memory at 2048, infer (GiB) | Peak memory at 2048, train (GiB) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in SERIES:
        lookup = {
            (row["task"], int(row["natoms"])): row
            for row in rows_by_series[label]
        }

        def field(task: str, natoms: int, name: str) -> str:
            row = lookup.get((task, natoms))
            if row is None:
                return "—"
            if row["status"] != "ok":
                return row["status"].upper()
            return f"{float(row[name]):.3f}"

        lines.append(
            f"| {DISPLAY_LABELS[label]} | {field('inference', 32, 'compile_s')} | "
            f"{field('train', 32, 'compile_s')} | "
            f"{field('inference', 2048, 'peak_memory_gib')} | "
            f"{field('train', 2048, 'peak_memory_gib')} |"
        )

    for task, heading in (("inference", "Inference atoms/s"), ("train", "Training atoms/s")):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Model | " + " | ".join(map(str, sizes)) + " |",
                "|---|" + "|".join("---:" for _ in sizes) + "|",
            ]
        )
        for label in FIGURE_SERIES:
            lookup = {
                int(row["natoms"]): row
                for row in rows_by_series[label]
                if row["task"] == task
            }
            lines.append(
                f"| {DISPLAY_LABELS[label]} | "
                + " | ".join(fmt_rate(lookup.get(size)) for size in sizes)
                + " |"
            )
    (args.output / "throughput_table.md").write_text("\n".join(lines) + "\n")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.6,
            "legend.fontsize": 6.25,
            "axes.edgecolor": "#545454",
            "text.color": "#252525",
            "axes.labelcolor": "#343434",
            "xtick.color": "#4B4B4B",
            "ytick.color": "#4B4B4B",
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), sharex=True, sharey=True)
    for ax, task, panel, title in zip(
        axes,
        ("inference", "train"),
        ("a", "b"),
        ("Inference", "Training"),
        strict=True,
    ):
        for label in SERIES:
            valid = sorted(
                (
                    (int(row["natoms"]), float(row["atoms_per_second"]))
                    for row in rows_by_series[label]
                    if row["task"] == task and row["status"] == "ok"
                )
            )
            if not valid:
                continue
            x, y = zip(*valid)
            ax.plot(
                x,
                y,
                label=label,
                linewidth=1.25,
                markersize=3.15,
                markeredgewidth=0.65,
                **STYLES[label],
            )
        ax.text(
            0.0,
            1.035,
            f"({panel})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )
        ax.text(
            0.07,
            1.035,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="normal",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(size) for size in sizes], rotation=0)
        ax.yaxis.grid(True, which="major", color="#E8E8E6", linewidth=0.5)
        ax.xaxis.grid(False)
        ax.grid(False, which="minor")
        ax.tick_params(which="major", direction="out")
        ax.tick_params(which="minor", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Throughput (atoms s$^{-1}$)")
    fig.text(
        0.535,
        0.045,
        "Number of atoms",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#343434",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles, strict=True))
    legend_order = (
        "Native MACE (CuEq)",
        "DPA-4 C48 (FP32 compiled)",
        "MACE-ICTC",
        "TECE C36 (OpenEq)",
        "CHORUS R16",
        "TECE C48 (OpenEq)",
        "DPA-4 C32 (FP32 compiled)",
    )
    legend_labels = {
        "Native MACE (CuEq)": "Native MACE · CuEq",
        "MACE-ICTC": "MACE-ICTC",
        "CHORUS R8": "CHORUS R8",
        "CHORUS R16": "CHORUS R16",
        "CHORUS R32": "CHORUS R32",
        "DPA-4 C32 (FP32 compiled)": "DPA-4 C32 · FP32",
        "DPA-4 C48 (FP32 compiled)": "DPA-4 C48 · FP32",
        "TECE C36 (OpenEq)": "TECE C36 · OpenEq",
        "TECE C48 (OpenEq)": "TECE C48 · OpenEq",
    }
    fig.legend(
        [handle_by_label[key] for key in legend_order],
        [legend_labels[key] for key in legend_order],
        loc="upper center",
        bbox_to_anchor=(0.535, 0.995),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.42,
        columnspacing=1.05,
        labelspacing=0.3,
        borderaxespad=0,
        fontsize=5.9,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.80, wspace=0.13)
    fig.savefig(args.output / "throughput_scaling.svg", bbox_inches="tight")
    fig.savefig(args.output / "throughput_scaling.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output / "throughput_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
