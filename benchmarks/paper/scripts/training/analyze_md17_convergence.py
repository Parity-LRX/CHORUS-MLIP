#!/usr/bin/env python3
"""Analyze MD17 apple-to-apple training convergence logs.

The raw MACE-ICTC and mace-torch logs report comparable validation RMSE values
but not comparable scalar validation losses. This script therefore extracts only
energy and force RMSE curves, computes per-run convergence metrics, and aggregates
them by dataset and mode.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ICTC_RE = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\s+step\s+(?P<step>\d+)\s+\S+\]\s+"
    r"train loss=(?P<train_loss>[0-9.eE+-]+).*?\|\s+val loss=(?P<val_loss>[0-9.eE+-]+)\s+"
    r"Frmse=(?P<force_rmse>[0-9.eE+-]+)\s+Ermse=(?P<energy_rmse>[0-9.eE+-]+)"
)
MACE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+):.*?loss=(?P<val_loss>[0-9.eE+-]+),\s+"
    r"RMSE_E_per_atom=\s*(?P<energy_mev_atom>[0-9.eE+-]+)\s+meV,\s+"
    r"RMSE_F=\s*(?P<force_mev_a>[0-9.eE+-]+)\s+meV\s*/\s*A"
)
INITIAL_MACE_RE = re.compile(
    r"Initial:.*?loss=(?P<val_loss>[0-9.eE+-]+),\s+"
    r"RMSE_E_per_atom=\s*(?P<energy_mev_atom>[0-9.eE+-]+)\s+meV,\s+"
    r"RMSE_F=\s*(?P<force_mev_a>[0-9.eE+-]+)\s+meV\s*/\s*A"
)

KNOWN_MODES = [
    "ictd_bridge_u_eager",
    "ictd_bridge_u_makefx",
    "ictd_cueq_makefx",
    "mace_e3nn",
    "mace_cueq",
]
MODE_ORDER = {mode: i for i, mode in enumerate(KNOWN_MODES)}


def _split_job_name(stem: str) -> tuple[str, str, str]:
    """Return dataset, mode, seed from a log stem."""
    before_seed, _, rest = stem.partition("_seed")
    seed = rest.split("_", 1)[0] if rest else ""
    for mode in KNOWN_MODES:
        suffix = "_" + mode
        if before_seed.endswith(suffix):
            return before_seed[: -len(suffix)], mode, seed
    return before_seed, "unknown", seed


def iter_logs(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("*.log"))
        elif path.is_file() and path.suffix == ".log":
            yield path


def parse_log(path: Path) -> list[dict[str, object]]:
    stem = path.stem
    if "_run-" in stem or stem.endswith("_debug"):
        return []
    dataset, mode, seed = _split_job_name(stem)
    text = path.read_text(errors="replace")
    rows: list[dict[str, object]] = []
    if mode.startswith("ictd_"):
        for m in ICTC_RE.finditer(text):
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "seed": seed,
                    "epoch": int(m.group("epoch")),
                    "step": int(m.group("step")),
                    "val_loss": float(m.group("val_loss")),
                    "train_loss": float(m.group("train_loss")),
                    "energy_rmse_eV_atom": float(m.group("energy_rmse")),
                    "force_rmse_eV_A": float(m.group("force_rmse")),
                    "source_log": str(path),
                }
            )
    elif mode.startswith("mace_"):
        init = INITIAL_MACE_RE.search(text)
        if init:
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "seed": seed,
                    "epoch": -1,
                    "step": "",
                    "val_loss": float(init.group("val_loss")),
                    "train_loss": "",
                    "energy_rmse_eV_atom": float(init.group("energy_mev_atom")) / 1000.0,
                    "force_rmse_eV_A": float(init.group("force_mev_a")) / 1000.0,
                    "source_log": str(path),
                }
            )
        for m in MACE_RE.finditer(text):
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "seed": seed,
                    "epoch": int(m.group("epoch")),
                    "step": "",
                    "val_loss": float(m.group("val_loss")),
                    "train_loss": "",
                    "energy_rmse_eV_atom": float(m.group("energy_mev_atom")) / 1000.0,
                    "force_rmse_eV_A": float(m.group("force_mev_a")) / 1000.0,
                    "source_log": str(path),
                }
            )
    return dedupe_rows(rows)


def dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_epoch: dict[int, dict[str, object]] = {}
    for row in rows:
        by_epoch[int(row["epoch"])] = row
    return [by_epoch[k] for k in sorted(by_epoch)]


def fmt(x: object) -> str:
    if x is None or x == "":
        return ""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.6g}"
    return str(x)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None, None
    if len(finite) == 1:
        return finite[0], 0.0
    return statistics.mean(finite), statistics.stdev(finite)


def first_epoch_at(rows: list[dict[str, object]], key: str, threshold: float) -> int | None:
    for row in rows:
        epoch = int(row["epoch"])
        if epoch < 0:
            continue
        value = float(row[key])
        if value <= threshold:
            return epoch
    return None


def mean_log_metric(rows: list[dict[str, object]], key: str) -> float | None:
    vals = [max(float(r[key]), 1e-12) for r in rows if int(r["epoch"]) >= 0]
    if not vals:
        return None
    return statistics.mean(math.log10(v) for v in vals)


def summarize_run(
    rows: list[dict[str, object]],
    force_thresholds: list[float],
    energy_thresholds: list[float],
    target_epoch: int | None,
) -> dict[str, object]:
    non_init = [r for r in rows if int(r["epoch"]) >= 0]
    if not non_init:
        first = rows[0]
        return {
            "dataset": first["dataset"],
            "mode": first["mode"],
            "seed": first["seed"],
            "status": "no_epoch_metrics",
            "final_epoch": "",
        }
    first = rows[0]
    final = non_init[-1]
    best_force = min(non_init, key=lambda r: float(r["force_rmse_eV_A"]))
    best_energy = min(non_init, key=lambda r: float(r["energy_rmse_eV_atom"]))
    status = "complete"
    if target_epoch is not None and int(final["epoch"]) < target_epoch - 1:
        status = "partial"
    out: dict[str, object] = {
        "dataset": first["dataset"],
        "mode": first["mode"],
        "seed": first["seed"],
        "status": status,
        "n_eval_points": len(non_init),
        "final_epoch": final["epoch"],
        "final_force_rmse_eV_A": final["force_rmse_eV_A"],
        "final_energy_rmse_eV_atom": final["energy_rmse_eV_atom"],
        "best_force_rmse_eV_A": best_force["force_rmse_eV_A"],
        "best_force_epoch": best_force["epoch"],
        "energy_at_best_force_eV_atom": best_force["energy_rmse_eV_atom"],
        "best_energy_rmse_eV_atom": best_energy["energy_rmse_eV_atom"],
        "best_energy_epoch": best_energy["epoch"],
        "force_at_best_energy_eV_A": best_energy["force_rmse_eV_A"],
        "mean_log10_force_rmse": mean_log_metric(rows, "force_rmse_eV_A"),
        "mean_log10_energy_rmse": mean_log_metric(rows, "energy_rmse_eV_atom"),
        "source_log": final["source_log"],
    }
    init = [r for r in rows if int(r["epoch"]) < 0]
    if init:
        out["initial_force_rmse_eV_A"] = init[-1]["force_rmse_eV_A"]
        out["initial_energy_rmse_eV_atom"] = init[-1]["energy_rmse_eV_atom"]
    for threshold in force_thresholds:
        out[f"epoch_force_le_{threshold:g}"] = first_epoch_at(rows, "force_rmse_eV_A", threshold)
    for threshold in energy_thresholds:
        out[f"epoch_energy_le_{threshold:g}"] = first_epoch_at(rows, "energy_rmse_eV_atom", threshold)
    return out


def aggregate_runs(
    run_rows: list[dict[str, object]],
    force_thresholds: list[float],
    energy_thresholds: list[float],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        groups[(str(row["dataset"]), str(row["mode"]))].append(row)
    aggregate: list[dict[str, object]] = []
    for (dataset, mode), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], MODE_ORDER.get(kv[0][1], 999), kv[0][1])):
        out: dict[str, object] = {
            "dataset": dataset,
            "mode": mode,
            "runs": len(rows),
            "complete_runs": sum(1 for r in rows if r.get("status") == "complete"),
        }
        for key in [
            "final_force_rmse_eV_A",
            "best_force_rmse_eV_A",
            "final_energy_rmse_eV_atom",
            "best_energy_rmse_eV_atom",
            "mean_log10_force_rmse",
            "mean_log10_energy_rmse",
        ]:
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            mean, std = mean_std(vals)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        for threshold in force_thresholds:
            key = f"epoch_force_le_{threshold:g}"
            vals = [int(r[key]) for r in rows if r.get(key) not in (None, "")]
            mean, std = mean_std([float(v) for v in vals])
            out[f"{key}_success"] = len(vals)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        for threshold in energy_thresholds:
            key = f"epoch_energy_le_{threshold:g}"
            vals = [int(r[key]) for r in rows if r.get(key) not in (None, "")]
            mean, std = mean_std([float(v) for v in vals])
            out[f"{key}_success"] = len(vals)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        aggregate.append(out)
    return aggregate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, run_rows: list[dict[str, object]], agg_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MD17 convergence analysis",
        "",
        "All force RMSE values are in eV/A and all energy RMSE values are in eV/atom.",
        "Scalar validation losses are intentionally not compared because MACE-ICTC and mace-torch log different internal loss normalizations.",
        "Partial rows are included so long-running jobs can be monitored before all modes finish.",
        "",
        "## Per-run summary",
        "",
        "| dataset | mode | seed | status | final epoch | best F | best F epoch | E at best F | best E | best E epoch |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(run_rows, key=lambda r: (str(r["dataset"]), MODE_ORDER.get(str(r["mode"]), 999), str(r["seed"]))):
        lines.append(
            "| {dataset} | {mode} | {seed} | {status} | {final_epoch} | {best_force_rmse_eV_A} | "
            "{best_force_epoch} | {energy_at_best_force_eV_atom} | {best_energy_rmse_eV_atom} | {best_energy_epoch} |".format(
                **{k: fmt(v) for k, v in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate by mode",
            "",
            "| dataset | mode | runs | complete | best F mean | best F std | best E mean | best E std | mean log10 F |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in agg_rows:
        lines.append(
            "| {dataset} | {mode} | {runs} | {complete_runs} | {best_force_rmse_eV_A_mean} | "
            "{best_force_rmse_eV_A_std} | {best_energy_rmse_eV_atom_mean} | "
            "{best_energy_rmse_eV_atom_std} | {mean_log10_force_rmse_mean} |".format(
                **{k: fmt(v) for k, v in row.items()}
            )
        )
    force_keys = [k for k in agg_rows[0].keys() if k.startswith("epoch_force_le_") and k.endswith("_mean")] if agg_rows else []
    energy_keys = [k for k in agg_rows[0].keys() if k.startswith("epoch_energy_le_") and k.endswith("_mean")] if agg_rows else []
    if force_keys:
        lines.extend(
            [
                "",
                "## Force convergence thresholds",
                "",
                "Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.",
                "",
                "| dataset | mode | " + " | ".join(k.removeprefix("epoch_force_le_").removesuffix("_mean") for k in force_keys) + " |",
                "|---|---|" + "---:|" * len(force_keys),
            ]
        )
        for row in agg_rows:
            cells = []
            for key in force_keys:
                prefix = key.removesuffix("_mean")
                success = row.get(f"{prefix}_success", "")
                mean = row.get(key, "")
                cells.append(f"{fmt(mean)} ({fmt(success)}/{fmt(row.get('runs', ''))})" if mean not in (None, "") else f"- ({fmt(success)}/{fmt(row.get('runs', ''))})")
            lines.append(f"| {row['dataset']} | {row['mode']} | " + " | ".join(cells) + " |")
    if energy_keys:
        lines.extend(
            [
                "",
                "## Energy convergence thresholds",
                "",
                "Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.",
                "",
                "| dataset | mode | " + " | ".join(k.removeprefix("epoch_energy_le_").removesuffix("_mean") for k in energy_keys) + " |",
                "|---|---|" + "---:|" * len(energy_keys),
            ]
        )
        for row in agg_rows:
            cells = []
            for key in energy_keys:
                prefix = key.removesuffix("_mean")
                success = row.get(f"{prefix}_success", "")
                mean = row.get(key, "")
                cells.append(f"{fmt(mean)} ({fmt(success)}/{fmt(row.get('runs', ''))})" if mean not in (None, "") else f"- ({fmt(success)}/{fmt(row.get('runs', ''))})")
            lines.append(f"| {row['dataset']} | {row['mode']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def plot_curves(curve_rows: list[dict[str, object]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plotting: matplotlib unavailable ({exc})")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted({str(r["dataset"]) for r in curve_rows})
    colors = {
        "ictd_bridge_u_eager": "#4c9a48",
        "ictd_bridge_u_makefx": "#b279a2",
        "ictd_cueq_makefx": "#e45756",
        "mace_e3nn": "#4c78a8",
        "mace_cueq": "#f58518",
    }
    labels = {
        "ictd_bridge_u_eager": "ICTC eager",
        "ictd_bridge_u_makefx": "ICTC compiled",
        "ictd_cueq_makefx": "ICTC+cuEq compiled",
        "mace_e3nn": "MACE e3nn",
        "mace_cueq": "MACE cuEq",
    }
    for dataset in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharex=True)
        for mode in KNOWN_MODES:
            mode_rows = [r for r in curve_rows if r["dataset"] == dataset and r["mode"] == mode and int(r["epoch"]) >= 0]
            if not mode_rows:
                continue
            by_epoch: dict[int, list[dict[str, object]]] = defaultdict(list)
            for row in mode_rows:
                by_epoch[int(row["epoch"])].append(row)
            epochs = sorted(by_epoch)
            force_mean = [statistics.mean(float(r["force_rmse_eV_A"]) for r in by_epoch[ep]) for ep in epochs]
            energy_mean = [statistics.mean(float(r["energy_rmse_eV_atom"]) for r in by_epoch[ep]) for ep in epochs]
            axes[0].plot(epochs, force_mean, label=labels.get(mode, mode), color=colors.get(mode), linewidth=1.8)
            axes[1].plot(epochs, energy_mean, label=labels.get(mode, mode), color=colors.get(mode), linewidth=1.8)
        axes[0].set_title("Force RMSE")
        axes[1].set_title("Energy RMSE")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel("eV/A")
        axes[1].set_ylabel("eV/atom")
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(dataset)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(out_dir / f"{dataset}_convergence.png", dpi=220)
        fig.savefig(out_dir / f"{dataset}_convergence.pdf")
        plt.close(fig)


def parse_thresholds(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path, help="Log files or directories containing *.log files.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-epoch", type=int, default=300)
    parser.add_argument("--force-thresholds", default="0.02,0.015,0.01,0.0075,0.005,0.0035")
    parser.add_argument("--energy-thresholds", default="0.005,0.002,0.001,0.0005,0.0002")
    parser.add_argument("--include-datasets", default="", help="Comma-separated dataset names to keep.")
    parser.add_argument("--exclude-datasets", default="", help="Comma-separated dataset names to drop.")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    force_thresholds = parse_thresholds(args.force_thresholds)
    energy_thresholds = parse_thresholds(args.energy_thresholds)
    include_datasets = {x.strip() for x in args.include_datasets.split(",") if x.strip()}
    exclude_datasets = {x.strip() for x in args.exclude_datasets.split(",") if x.strip()}
    curve_rows: list[dict[str, object]] = []
    by_log: dict[Path, list[dict[str, object]]] = {}
    for log in iter_logs(args.logs):
        rows = parse_log(log)
        if include_datasets:
            rows = [r for r in rows if str(r["dataset"]) in include_datasets]
        if exclude_datasets:
            rows = [r for r in rows if str(r["dataset"]) not in exclude_datasets]
        if rows:
            by_log[log] = rows
            curve_rows.extend(rows)
    run_rows = [summarize_run(rows, force_thresholds, energy_thresholds, args.target_epoch) for rows in by_log.values()]
    agg_rows = aggregate_runs(run_rows, force_thresholds, energy_thresholds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "curves.csv", curve_rows)
    write_csv(args.out_dir / "runs.csv", run_rows)
    write_csv(args.out_dir / "aggregate_by_mode.csv", agg_rows)
    write_markdown(args.out_dir / "summary.md", run_rows, agg_rows)
    if args.plots:
        plot_curves(curve_rows, args.out_dir / "figures")
    print(f"parsed {len(by_log)} logs, {len(curve_rows)} curve rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
