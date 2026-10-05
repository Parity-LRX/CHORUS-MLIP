#!/usr/bin/env python3
"""Plot the cross-domain accuracy and cost synthesis.

Chart contract
--------------
Question
    Which tested configurations lie on the in-domain force-accuracy,
    throughput and parameter-count frontier at the largest common trainable
    graph size?
Takeaway
    DPA-4 C48 attains the lowest aggregate force error, while CHORUS Final and
    Persistent occupy lower-parameter and competitive-latency force frontiers.
Form
    Two force-first Pareto scatters at N=2048: six-endpoint force-error gap
    versus trainable parameters and versus inference time per structure.
Encoding
    Position carries both metrics directly. Neutral backbones, blue CHORUS,
    orange DPA-4 and green TECE; marker shape duplicates identity. Opaque
    linked points mark the two-objective non-dominated set in each panel.
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


def normalized_force_scores(
    path: Path,
) -> tuple[dict[str, float], list[float], list[str]]:
    payload = json.loads(path.read_text())
    values = payload["models"]
    missing = [model for model in MODEL_ORDER if model not in values]
    if missing:
        raise ValueError("Missing accuracy rows: " + ", ".join(missing))
    endpoints = [str(endpoint) for endpoint in payload["endpoints"]]
    endpoint_count = len(endpoints)
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
    return scores, best, endpoints


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
    endpoints: list[str],
    natoms: int,
) -> None:
    flat: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "natoms": natoms,
        "accuracy_metric": "geometric mean of endpoint-wise Force MAE / best endpoint Force MAE",
        "accuracy_endpoints": endpoints,
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

    def pair_front(y_key: str) -> list[str]:
        return [
            model
            for model, row in combined.items()
            if not any(
                other_row["score"] <= row["score"]
                and other_row[y_key] <= row[y_key]
                and (
                    other_row["score"] < row["score"]
                    or other_row[y_key] < row[y_key]
                )
                for other_model, other_row in combined.items()
                if other_model != model
            )
        ]

    summary["pareto_force_inference_training_parameters"] = pareto_four
    summary["pareto_force_parameters"] = pair_front("parameters")
    summary["pareto_force_inference"] = pair_front("inference_latency_ms")
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
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.8,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#555555",
            "text.color": "#272727",
            "xtick.color": "#4B4B4B",
            "ytick.color": "#4B4B4B",
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
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

    def pair_front(y_key: str) -> set[str]:
        return {
            model
            for model, row in combined.items()
            if not any(
                other_row["force_gap"] <= row["force_gap"]
                and other_row[y_key] <= row[y_key]
                and (
                    other_row["force_gap"] < row["force_gap"]
                    or other_row[y_key] < row[y_key]
                )
                for other_model, other_row in combined.items()
                if other_model != model
            )
        }

    label_offsets = {
        "parameters": {
            "DPA-4 C48": (6, 6),
            "CHORUS-Persistent R16": (6, -10),
            "CHORUS-Final R16": (6, 6),
            "DPA-4 C32": (6, -10),
            "MACE-ICTC": (6, 6),
            "NequIP-CHORUS-Persistent": (6, 6),
            "TECE C48": (6, 6),
            "TECE C36": (6, 6),
            "NequIP-CHORUS-Final": (6, 6),
            "NequIP-SH": (6, 6),
            "MACE-SH": (6, -10),
        },
        "inference_latency": {
            "DPA-4 C48": (6, -10),
            "CHORUS-Persistent R16": (6, 6),
            "CHORUS-Final R16": (6, -10),
            "DPA-4 C32": (6, -10),
            "MACE-ICTC": (6, 6),
            "NequIP-CHORUS-Persistent": (6, 6),
            "TECE C48": (-6, 6),
            "TECE C36": (-6, 6),
            "NequIP-CHORUS-Final": (6, 6),
            "NequIP-SH": (6, 6),
            "MACE-SH": (6, -10),
        },
    }
    panels = (
        ("parameters", "Trainable parameters", "Parameters (million)  ← lower", (0, 1.55)),
        ("inference_latency", "Inference time at N=2048", "ms per structure  ← lower", (0, 180)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
    for panel_index, (ax, (key, title, unit, y_limits)) in enumerate(zip(axes, panels)):
        pareto = pair_front(key)
        frontier_models = sorted(pareto, key=lambda model: combined[model]["force_gap"])
        ax.plot(
            [combined[model]["force_gap"] for model in frontier_models],
            [combined[model][key] for model in frontier_models],
            color="#5B5B58",
            linewidth=0.8,
            linestyle=(0, (2.2, 2.2)),
            zorder=1,
        )
        for model in MODEL_ORDER:
            x_value = combined[model]["force_gap"]
            y_value = combined[model][key]
            style = STYLE[model]
            is_pareto = model in pareto
            ax.scatter(
                x_value,
                y_value,
                s=44,
                color=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidths=0.65,
                alpha=0.98 if is_pareto else 0.30,
                zorder=3,
            )
            dx, dy = label_offsets[key][model]
            ax.annotate(
                DISPLAY[model],
                xy=(x_value, y_value),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                va="center",
                fontsize=5.7,
                fontweight="normal",
                color="#2B2B2B" if is_pareto else "#8A8A86",
            )
        ax.set_xlim(-3, 125)
        ax.set_ylim(*y_limits)
        ax.set_title(f"({chr(ord('a') + panel_index)})  {title}", loc="left", pad=7)
        ax.set_xlabel("Six-endpoint force-MAE gap (%)  ← lower")
        ax.set_ylabel(unit)
        ax.grid(color="#E6E6E3", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(which="minor", length=0)
    fig.text(
        0.995,
        0.015,
        "Force metric only; opaque linked points form the two-objective Pareto set in each panel",
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="#555552",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.19, wspace=0.25)
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
    scores, endpoint_best, endpoints = normalized_force_scores(args.accuracy)
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

    write_outputs(args.output, rows_by_task, endpoint_best, endpoints, args.natoms)
    plot(args.output, rows_by_task, args.natoms)


if __name__ == "__main__":
    main()
