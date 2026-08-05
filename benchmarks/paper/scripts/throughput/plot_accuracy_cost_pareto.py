#!/usr/bin/env python3
"""Plot the Table-5 accuracy and Figure-3 cost synthesis.

Chart contract
--------------
Question
    Which tested configurations lie on the force-accuracy, throughput and
    parameter-count frontier at the largest common trainable graph size?
Takeaway
    CHORUS supplies the lowest aggregate force-error operating points, while
    its Final and Persistent scopes occupy distinct accuracy--cost positions.
Form
    Four aligned dot plots at N=2048: force-error gap, inference latency,
    training latency and trainable parameters. Models are sorted by force
    error, so every quantitative axis points left toward the preferred region.
Encoding
    Position carries every metric directly. Neutral backbones, blue CHORUS,
    orange DPA-4 and green TECE; marker shape duplicates identity. Full-opacity
    points and bold row labels mark the four-objective non-dominated set.
Outputs
    Machine-readable JSON/CSV plus PDF/SVG/PNG figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODEL_ORDER = (
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

DISPLAY = {
    "MACE-SH": "MACE-SH",
    "MACE-ICTC": "MACE-ICTC",
    "CHORUS-Final R16": "M-CHORUS-F",
    "CHORUS-Persistent R16": "M-CHORUS-P",
    "NequIP-SH": "NequIP-SH",
    "NequIP-CHORUS-Final": "N-CHORUS-F",
    "NequIP-CHORUS-Persistent": "N-CHORUS-P",
    "DPA-4 C32": "DPA-4 C32",
    "DPA-4 C48": "DPA-4 C48",
    "TECE C36": "TECE C36",
    "TECE C48": "TECE C48",
}

STYLE = {
    "MACE-SH": dict(color="#9A9A96", marker="o"),
    "MACE-ICTC": dict(color="#555552", marker="s"),
    "CHORUS-Final R16": dict(color="#2D6C9F", marker="s"),
    "CHORUS-Persistent R16": dict(color="#173F68", marker="D"),
    "NequIP-SH": dict(color="#B0B0AC", marker="^"),
    "NequIP-CHORUS-Final": dict(color="#4D8DBD", marker="^"),
    "NequIP-CHORUS-Persistent": dict(color="#205381", marker="v"),
    "DPA-4 C32": dict(color="#D98531", marker="o"),
    "DPA-4 C48": dict(color="#9B5A16", marker="s"),
    "TECE C36": dict(color="#78A48E", marker="o"),
    "TECE C48": dict(color="#2D845E", marker="s"),
}

def load_throughput(path: Path, natoms: int) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    result: dict[str, dict[str, float]] = {}
    for row in payload["rows"]:
        model = str(row["model"])
        if (
            model in MODEL_ORDER
            and int(row["natoms"]) == natoms
            and row["status"] == "ok"
        ):
            result.setdefault(model, {})[str(row["task"])] = float(
                row["atoms_per_second"]
            )
            result[model]["parameters"] = float(row["parameters"])
    missing = [
        model
        for model in MODEL_ORDER
        if model not in result or not {"inference", "train", "parameters"} <= result[model].keys()
    ]
    if missing:
        raise ValueError("Missing throughput rows: " + ", ".join(missing))
    return result


def normalized_force_scores(path: Path) -> tuple[dict[str, float], list[float]]:
    payload = json.loads(path.read_text())
    values = payload["models"]
    missing = [model for model in MODEL_ORDER if model not in values]
    if missing:
        raise ValueError("Missing accuracy rows: " + ", ".join(missing))
    endpoint_count = len(payload["endpoints"])
    best = [
        min(float(values[model][index]) for model in MODEL_ORDER)
        for index in range(endpoint_count)
    ]
    scores = {}
    for model in MODEL_ORDER:
        ratios = [
            float(value) / best[index]
            for index, value in enumerate(values[model])
        ]
        scores[model] = math.exp(sum(math.log(ratio) for ratio in ratios) / endpoint_count)
    return scores, best


def dominates(
    lhs: dict[str, float],
    rhs: dict[str, float],
    *,
    include_parameters: bool,
) -> bool:
    weak = lhs["score"] <= rhs["score"] and lhs["throughput"] >= rhs["throughput"]
    strict = lhs["score"] < rhs["score"] or lhs["throughput"] > rhs["throughput"]
    if include_parameters:
        weak = weak and lhs["parameters"] <= rhs["parameters"]
        strict = strict or lhs["parameters"] < rhs["parameters"]
    return weak and strict


def frontier(
    rows: dict[str, dict[str, float]], *, include_parameters: bool
) -> list[str]:
    return [
        model
        for model, row in rows.items()
        if not any(
            dominates(other_row, row, include_parameters=include_parameters)
            for other_model, other_row in rows.items()
            if other_model != model
        )
    ]


def write_outputs(
    output: Path,
    rows_by_task: dict[str, dict[str, dict[str, float]]],
    endpoint_best: list[float],
    natoms: int,
) -> None:
    flat: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "natoms": natoms,
        "accuracy_metric": "geometric mean of endpoint-wise Force MAE / best endpoint Force MAE",
        "endpoint_best_force_mae_mev_per_angstrom": endpoint_best,
        "tasks": {},
    }
    for task, rows in rows_by_task.items():
        pareto_2d = frontier(rows, include_parameters=False)
        pareto_3d = frontier(rows, include_parameters=True)
        summary["tasks"][task] = {
            "pareto_accuracy_throughput": pareto_2d,
            "pareto_accuracy_throughput_parameters": pareto_3d,
            "models": rows,
        }
        for model, row in rows.items():
            flat.append(
                {
                    "task": task,
                    "model": model,
                    **row,
                    "pareto_2d": model in pareto_2d,
                    "pareto_3d": model in pareto_3d,
                }
            )
    combined = {
        model: {
            "score": rows_by_task["inference"][model]["score"],
            "inference_latency_ms": natoms
            / rows_by_task["inference"][model]["throughput"]
            * 1000,
            "training_latency_ms": natoms
            / rows_by_task["train"][model]["throughput"]
            * 1000,
            "parameters": rows_by_task["inference"][model]["parameters"],
        }
        for model in MODEL_ORDER
    }

    def dominates_four(lhs: dict[str, float], rhs: dict[str, float]) -> bool:
        keys = ("score", "inference_latency_ms", "training_latency_ms", "parameters")
        return all(lhs[key] <= rhs[key] for key in keys) and any(
            lhs[key] < rhs[key] for key in keys
        )

    pareto_four = [
        model
        for model, row in combined.items()
        if not any(
            dominates_four(other_row, row)
            for other_model, other_row in combined.items()
            if other_model != model
        )
    ]
    summary["pareto_force_inference_training_parameters"] = pareto_four
    summary["combined_models"] = combined
    (output / "accuracy_cost_pareto_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output / "accuracy_cost_pareto_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def plot(
    output: Path,
    rows_by_task: dict[str, dict[str, dict[str, float]]],
    natoms: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#555555",
            "text.color": "#272727",
            "xtick.color": "#4B4B4B",
            "ytick.color": "#4B4B4B",
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    combined = {
        model: {
            "force_gap": (rows_by_task["inference"][model]["score"] - 1.0) * 100,
            "inference_latency": natoms
            / rows_by_task["inference"][model]["throughput"]
            * 1000,
            "training_latency": natoms
            / rows_by_task["train"][model]["throughput"]
            * 1000,
            "parameters": rows_by_task["inference"][model]["parameters"] / 1_000_000,
        }
        for model in MODEL_ORDER
    }

    def dominates_four(lhs: dict[str, float], rhs: dict[str, float]) -> bool:
        return all(lhs[key] <= rhs[key] for key in lhs) and any(
            lhs[key] < rhs[key] for key in lhs
        )

    pareto = {
        model
        for model, row in combined.items()
        if not any(
            dominates_four(other_row, row)
            for other_model, other_row in combined.items()
            if other_model != model
        )
    }
    ordered_models = sorted(MODEL_ORDER, key=lambda model: combined[model]["force_gap"])
    y_position = {model: len(ordered_models) - 1 - index for index, model in enumerate(ordered_models)}
    metrics = (
        ("force_gap", "Force-error gap", "%", (0, 140), "{:.1f}"),
        ("inference_latency", "Inference latency", "ms", (0, 180), "{:.1f}"),
        ("training_latency", "Training latency", "ms", (0, 440), "{:.0f}"),
        ("parameters", "Parameters", "million", (0, 1.55), "{:.2f}"),
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.2, 3.08),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 0.98, 0.98, 0.84]},
    )
    for panel_index, (ax, (key, title, unit, limits, formatter)) in enumerate(zip(axes, metrics)):
        for row_index in range(len(ordered_models)):
            if row_index % 2 == 0:
                ax.axhspan(row_index - 0.5, row_index + 0.5, color="#F7F7F5", zorder=0)
        for model in ordered_models:
            value = combined[model][key]
            style = STYLE[model]
            is_pareto = model in pareto
            ax.scatter(
                value,
                y_position[model],
                s=34,
                color=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidths=0.55,
                alpha=0.98 if is_pareto else 0.28,
                zorder=3,
            )
            ax.annotate(
                formatter.format(value),
                xy=(value, y_position[model]),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=5.35,
                color="#333333" if is_pareto else "#92928E",
            )
        ax.set_xlim(*limits)
        ax.set_title(f"({chr(ord('a') + panel_index)})  {title}", loc="left", pad=7)
        ax.set_xlabel(f"{unit}  ← better")
        ax.grid(axis="x", color="#E6E6E3", linewidth=0.5)
        ax.grid(False, axis="y")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(which="minor", length=0)
    axes[0].set_yticks(
        [y_position[model] for model in ordered_models],
        [DISPLAY[model] for model in ordered_models],
    )
    for tick, model in zip(axes[0].get_yticklabels(), ordered_models):
        tick.set_fontweight("bold" if model in pareto else "normal")
        tick.set_color("#2B2B2B" if model in pareto else "#8A8A86")
        tick.set_fontsize(6.3)
    axes[0].set_ylim(-0.65, len(ordered_models) - 0.35)
    fig.text(
        0.995,
        0.02,
        "Bold/opaque: four-objective Pareto set; faded: dominated",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555552",
    )
    fig.subplots_adjust(left=0.195, right=0.995, top=0.92, bottom=0.18, wspace=0.12)
    for suffix in ("pdf", "svg"):
        fig.savefig(output / f"accuracy_cost_pareto.{suffix}", bbox_inches="tight")
    fig.savefig(output / "accuracy_cost_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--throughput", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--natoms", type=int, default=2048)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    throughput = load_throughput(args.throughput, args.natoms)
    scores, endpoint_best = normalized_force_scores(args.accuracy)
    rows_by_task: dict[str, dict[str, dict[str, float]]] = {}
    for task in ("inference", "train"):
        rows_by_task[task] = OrderedDict(
            (
                model,
                {
                    "score": scores[model],
                    "throughput": throughput[model][task],
                    "parameters": throughput[model]["parameters"],
                },
            )
            for model in MODEL_ORDER
        )

    write_outputs(args.output, rows_by_task, endpoint_best, args.natoms)
    plot(args.output, rows_by_task, args.natoms)


if __name__ == "__main__":
    main()
