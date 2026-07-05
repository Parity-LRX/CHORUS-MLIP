#!/usr/bin/env python3
"""Aggregate repeated empirical NTK spectrum diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = [
    "lambda_min_pos",
    "lambda_max",
    "kappa_pos",
    "trace",
    "lambda_min_pos_over_trace",
    "lambda_max_over_trace",
    "stable_gd_lr_bound",
    "train_lr_times_lambda_max",
    "train_lr_times_lambda_min_pos",
]


def mean_std(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt(x):
    if x is None:
        return ""
    return f"{float(x):.6g}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path, help="ntk_spectrum.json files or directories containing them")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--baseline", default="mace_e3nn")
    args = p.parse_args()

    files = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("**/ntk_spectrum.json")))
        else:
            files.append(path)
    rows = []
    for file in files:
        payload = json.loads(file.read_text())
        run_id = file.parent.name
        for mode, spec in payload["modes"].items():
            row = {
                "run_id": run_id,
                "mode": mode,
                "source": str(file),
                "seed": payload.get("seed"),
                "batch_index": payload.get("batch_index"),
                "batch_size": payload.get("batch_size"),
                "max_force_components": payload.get("max_force_components"),
                "dtype": payload.get("dtype"),
            }
            for key in METRICS:
                row[key] = spec.get(key)
            rows.append(row)

    by_mode = defaultdict(list)
    by_run_mode = {}
    for row in rows:
        by_mode[row["mode"]].append(row)
        by_run_mode[(row["run_id"], row["mode"])] = row

    agg = []
    for mode, mode_rows in sorted(by_mode.items()):
        out = {"mode": mode, "runs": len(mode_rows)}
        for metric in METRICS:
            mean, std = mean_std([r.get(metric) for r in mode_rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        ratio_rows = []
        for row in mode_rows:
            base = by_run_mode.get((row["run_id"], args.baseline))
            if not base or mode == args.baseline:
                continue
            ratio = {"mode": mode, "run_id": row["run_id"]}
            for metric in METRICS:
                b = base.get(metric)
                v = row.get(metric)
                if b is None or v is None or float(b) == 0.0:
                    ratio[metric] = None
                else:
                    ratio[metric] = float(v) / float(b)
            ratio_rows.append(ratio)
        for metric in METRICS:
            mean, std = mean_std([r.get(metric) for r in ratio_rows])
            out[f"{metric}_ratio_vs_{args.baseline}_mean"] = mean
            out[f"{metric}_ratio_vs_{args.baseline}_std"] = std
        agg.append(out)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "ntk_runs.csv").open("w", newline="") as f:
        fields = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "ntk_aggregate.csv").open("w", newline="") as f:
        fields = list(agg[0].keys()) if agg else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(agg)

    lines = [
        "# Empirical NTK Sweep Summary",
        "",
        f"Baseline for ratios: `{args.baseline}`.",
        "",
        "| mode | runs | lambda_min mean | lambda_max mean | kappa mean | trace mean | lambda_min ratio | lambda_max ratio | trace ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['mode']} | {row['runs']} | {fmt(row.get('lambda_min_pos_mean'))} | "
            f"{fmt(row.get('lambda_max_mean'))} | {fmt(row.get('kappa_pos_mean'))} | "
            f"{fmt(row.get('trace_mean'))} | "
            f"{fmt(row.get(f'lambda_min_pos_ratio_vs_{args.baseline}_mean'))} | "
            f"{fmt(row.get(f'lambda_max_ratio_vs_{args.baseline}_mean'))} | "
            f"{fmt(row.get(f'trace_ratio_vs_{args.baseline}_mean'))} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows from {len(files)} files -> {args.out_dir}")


if __name__ == "__main__":
    main()
