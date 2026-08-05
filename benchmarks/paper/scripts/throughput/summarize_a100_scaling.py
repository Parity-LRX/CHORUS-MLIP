#!/usr/bin/env python3
"""Summarize and plot the strict-FP32 A100 atom-count scaling benchmark.

Chart contract
--------------
Question
    How do the backbone, CHORUS scope, and external equivariant models scale
    from 128 to 4096 atoms under the same fixed-degree workload?
Takeaway
    Final-layer CHORUS retains substantially more throughput than the
    persistent stream, while both remain faster than the tested TECE widths;
    the NequIP implementation shows the same ordered cost pattern.
Form
    Two-panel multi-series line chart: inference and training. A shared legend
    keeps every model on the same visual scale without repeating architecture
    facets. Logarithmic throughput and atom-count axes retain the full scaling
    shape.
Encoding
    Neutral backbones, blue CHORUS, orange DPA-4, olive TECE. Markers and line
    styles duplicate color distinctions for grayscale reproduction.
Outputs
    Machine-readable JSON/CSV, Markdown lookup tables, and PDF/SVG/PNG figures.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SERIES: OrderedDict[str, str] = OrderedDict(
    [
        ("MACE-SH", "mace_sh_c128.json"),
        ("MACE-ICTC", "mace_ictc_c128.json"),
        ("CHORUS-Final R8", "chorus_final_c128_r8.json"),
        ("CHORUS-Final R16", "chorus_final_c128_r16.json"),
        ("CHORUS-Final R32", "chorus_final_c128_r32.json"),
        ("CHORUS-Persistent R16", "chorus_persistent_c128_r16.json"),
        ("NequIP-SH", "nequip_sh_c84.json"),
        ("NequIP-CHORUS-Final", "nequip_chorus_final_c84_r16.json"),
        ("NequIP-CHORUS-Persistent", "nequip_chorus_persistent_c84_r16.json"),
        ("DPA-4 C32", "dpa4_c32_fp32_compiled.json"),
        ("DPA-4 C48", "dpa4_c48_fp32_compiled.json"),
        ("TECE C36", "tece_c36_openeq.json"),
        ("TECE C48", "tece_c48_openeq.json"),
    ]
)

PLOT_SERIES = (
    "MACE-SH",
    "MACE-ICTC",
    "CHORUS-Final R16",
    "CHORUS-Persistent R16",
    "NequIP-SH",
    "NequIP-CHORUS-Final",
    "NequIP-CHORUS-Persistent",
    "DPA-4 C32",
    "DPA-4 C48",
    "TECE C36",
    "TECE C48",
)

PLOT_LABEL = {
    "CHORUS-Final R16": "MACE-CHORUS-Final",
    "CHORUS-Persistent R16": "MACE-CHORUS-Persistent",
}

STYLE = {
    "MACE-SH": dict(color="#9A9A96", marker="o", linestyle=":"),
    "MACE-ICTC": dict(color="#454545", marker="s", linestyle="--"),
    "CHORUS-Final R8": dict(color="#8FB3D9", marker="o", linestyle="-"),
    "CHORUS-Final R16": dict(color="#205381", marker="s", linestyle="-"),
    "CHORUS-Final R32": dict(color="#173F68", marker="^", linestyle="-"),
    "CHORUS-Persistent R16": dict(color="#173F68", marker="D", linestyle="-.", markerfacecolor="white"),
    "NequIP-SH": dict(color="#8D8D89", marker="^", linestyle=":"),
    "NequIP-CHORUS-Final": dict(color="#205381", marker="^", linestyle="-"),
    "NequIP-CHORUS-Persistent": dict(color="#173F68", marker="v", linestyle="-.", markerfacecolor="white"),
    "DPA-4 C32": dict(color="#DC7520", marker="o", linestyle="--", markerfacecolor="white"),
    "DPA-4 C48": dict(color="#9B5A16", marker="s", linestyle="--"),
    "TECE C36": dict(color="#78A48E", marker="o", linestyle=":", markerfacecolor="white"),
    "TECE C48": dict(color="#2D845E", marker="s", linestyle=":"),
}


def load_payloads(source: Path) -> OrderedDict[str, dict[str, Any]]:
    payloads: OrderedDict[str, dict[str, Any]] = OrderedDict()
    missing: list[str] = []
    for label, filename in SERIES.items():
        path = source / filename
        if path.exists():
            payloads[label] = json.loads(path.read_text())
        else:
            missing.append(filename)
    if missing:
        print("Missing inputs: " + ", ".join(missing))
    return payloads


def row_lookup(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["task"], int(row["natoms"])): row for row in payload["rows"]}


def rate_text(row: dict[str, Any] | None) -> str:
    if row is None:
        return "—"
    if row.get("status") != "ok":
        return str(row.get("status", "error")).upper()
    return f"{float(row['atoms_per_second']):,.0f}"


def memory_text(row: dict[str, Any] | None) -> str:
    if row is None:
        return "—"
    if row.get("status") != "ok":
        return str(row.get("status", "error")).upper()
    return f"{float(row['peak_memory_gib']):.2f}"


def short_rate(value: float, _position: float) -> str:
    if value >= 100_000:
        return f"{value / 1000:.0f}k"
    if value >= 10_000:
        return f"{value / 1000:.0f}k"
    if value >= 1_000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def backend_text(payload: dict[str, Any]) -> str:
    if payload.get("backend"):
        return str(payload["backend"])
    oeq = payload.get("openequivariance", {})
    if oeq.get("precompiled_extension"):
        return f"OpenEquivariance {oeq.get('version', '')} precompiled AOTI".strip()
    return "—"


def write_tables(
    output: Path,
    payloads: OrderedDict[str, dict[str, Any]],
    sizes: list[int],
) -> None:
    flat: list[dict[str, Any]] = []
    for label, payload in payloads.items():
        flat.extend({"model": label, **row} for row in payload["rows"])
    fields = sorted({key for row in flat for key in row})
    with (output / "throughput_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    (output / "throughput_merged.json").write_text(
        json.dumps({"series": payloads, "rows": flat}, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# A100 atom-count scaling",
        "",
        "Strict FP32 with TF32 disabled; fixed 32 directed neighbors per atom. "
        "Inference computes energy and conservative forces. Training includes the "
        "energy-force loss, backward pass, and optimizer update. Compilation and "
        "graph preparation are excluded from steady-state throughput.",
        "DPA-4 constructs its neighbor representation inside the timed model call; "
        "the MACE, NequIP, and TECE inputs use prebuilt fixed-degree graphs.",
        "",
        "## Configurations",
        "",
        "| Model | Parameters | Backend |",
        "|:--|--:|:--|",
    ]
    for label, payload in payloads.items():
        lines.append(
            f"| {label} | {int(payload['parameters']):,} | {backend_text(payload)} |"
        )
    for task, title in (("inference", "Inference throughput (atoms/s)"), ("train", "Training throughput (atoms/s)")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Model | " + " | ".join(map(str, sizes)) + " |",
                "|:--|" + "|".join("--:" for _ in sizes) + "|",
            ]
        )
        for label, payload in payloads.items():
            lookup = row_lookup(payload)
            lines.append(
                f"| {label} | "
                + " | ".join(rate_text(lookup.get((task, size))) for size in sizes)
                + " |"
            )
    for task, title in (("inference", "Inference peak memory (GiB)"), ("train", "Training peak memory (GiB)")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Model | " + " | ".join(map(str, sizes)) + " |",
                "|:--|" + "|".join("--:" for _ in sizes) + "|",
            ]
        )
        for label, payload in payloads.items():
            lookup = row_lookup(payload)
            lines.append(
                f"| {label} | "
                + " | ".join(memory_text(lookup.get((task, size))) for size in sizes)
                + " |"
            )
    (output / "throughput_table.md").write_text("\n".join(lines) + "\n")


def plot(
    output: Path,
    payloads: OrderedDict[str, dict[str, Any]],
    sizes: list[int],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#555555",
            "text.color": "#272727",
            "xtick.color": "#4B4B4B",
            "ytick.color": "#4B4B4B",
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), sharex=False, sharey=False)
    tasks = (("inference", "Inference"), ("train", "Training"))
    for panel_index, (task, panel_title) in enumerate(tasks):
        ax = axes[panel_index]
        panel_sizes = sizes if task == "inference" else [size for size in sizes if size <= 2048]
        for label in PLOT_SERIES:
            if label not in payloads:
                continue
            valid = sorted(
                (int(row["natoms"]), float(row["atoms_per_second"]))
                for row in payloads[label]["rows"]
                if row["task"] == task
                and row["status"] == "ok"
                and int(row["natoms"]) in panel_sizes
            )
            if not valid:
                continue
            x, y = zip(*valid)
            ax.plot(
                x,
                y,
                label=PLOT_LABEL.get(label, label),
                linewidth=1.15,
                markersize=3.1,
                markeredgewidth=0.6,
                **STYLE[label],
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(panel_sizes)
        ax.set_xticklabels([str(size) for size in panel_sizes], rotation=30, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(short_rate))
        ax.grid(axis="y", which="major", color="#E6E6E3", linewidth=0.5)
        ax.grid(False, which="minor")
        ax.tick_params(which="minor", length=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({chr(ord('a') + panel_index)})  {panel_title}", loc="left", pad=7)
        ax.set_xlabel("Number of atoms")
    axes[0].set_ylabel("Throughput (atoms s$^{-1}$)")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        ncol=4,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.97, bottom=0.27, wspace=0.10)
    for suffix in ("pdf", "svg"):
        fig.savefig(output / f"throughput_scaling.{suffix}", bbox_inches="tight")
    fig.savefig(output / "throughput_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads(args.source)
    sizes = sorted(
        {
            int(row["natoms"])
            for payload in payloads.values()
            for row in payload["rows"]
        }
    )
    write_tables(args.output, payloads, sizes)
    plot(args.output, payloads, sizes)


if __name__ == "__main__":
    main()
