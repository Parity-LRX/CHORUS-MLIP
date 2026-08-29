#!/usr/bin/env python3
"""Build a deterministic, read-only snapshot for the NequIP causal operator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


def task_identity(task_id: int, datasets: list[str], modes: list[str], seeds: list[int]) -> dict:
    seed_index, within_seed = divmod(task_id, len(datasets) * len(modes))
    dataset_index, mode_index = divmod(within_seed, len(modes))
    return {
        "task_id": task_id,
        "seed": seeds[seed_index],
        "dataset": datasets[dataset_index],
        "mode": modes[mode_index],
    }


def metric_csv(root: Path, seed: int, mode: str, dataset: str) -> Path | None:
    paths = sorted((root / f"seed_{seed}" / mode / dataset).glob("**/metrics_epoch.csv"))
    return paths[0] if len(paths) == 1 else None


def numeric_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                key = key.strip()
                if key == "wall":
                    continue
                try:
                    parsed[key] = float(value.strip())
                except (AttributeError, TypeError, ValueError):
                    pass
            rows.append(parsed)
    return rows


def compare_pair(aligned: Path, full: Path) -> dict:
    left = numeric_rows(aligned)
    right = numeric_rows(full)
    max_delta = 0.0
    compared = 0
    if len(left) == len(right):
        for lhs, rhs in zip(left, right):
            for key in sorted(set(lhs) & set(rhs)):
                if key in {"epoch", "LR"}:
                    continue
                max_delta = max(max_delta, abs(lhs[key] - rhs[key]))
                compared += 1
    else:
        max_delta = float("inf")
    return {
        "aligned_epochs": len(left),
        "full_epochs": len(right),
        "compared_values": compared,
        "maximum_metric_delta": max_delta,
        "collapsed": bool(compared and max_delta <= 1.0e-12),
    }


def slurm_rows() -> list[dict[str, str]]:
    command = [
        "squeue", "-u", os.environ.get("USER", "rxlin"), "-h",
        "-o", "%i|%T|%j|%M|%R",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|", 4)
        if len(fields) == 5 and "nequip-causal-v2" in fields[2]:
            rows.append(dict(zip(("job_id", "state", "name", "elapsed", "reason"), fields)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    datasets = list(protocol["datasets"])
    modes = list(protocol["modes"])
    seeds = list(protocol["seeds"])
    total = len(datasets) * len(modes) * len(seeds)
    tasks = []
    for task_id in range(total):
        identity = task_identity(task_id, datasets, modes, seeds)
        status = args.campaign_root / "status" / f"task_{task_id}"
        if (status / "DONE").is_file():
            state = "DONE"
        elif (status / "FAILED").is_file():
            state = "FAILED"
        elif (status / "identity.txt").is_file():
            state = "STARTED"
        else:
            state = "PENDING"
        identity["state"] = state
        tasks.append(identity)

    pair_checks = []
    for dataset in datasets:
        for seed in seeds:
            aligned = metric_csv(args.campaign_root, seed, "aligned", dataset)
            full = metric_csv(args.campaign_root, seed, "full", dataset)
            if aligned is not None and full is not None:
                check = compare_pair(aligned, full)
                check.update(dataset=dataset, seed=seed)
                pair_checks.append(check)

    oom_logs = []
    for path in sorted((args.campaign_root / "slurm").glob("*.err")):
        text = path.read_text(errors="replace")[-200_000:].lower()
        if "out of memory" in text or "cuda oom" in text or "cuda error: out of memory" in text:
            oom_logs.append(str(path))

    state_counts = {state: sum(t["state"] == state for t in tasks) for state in ("DONE", "FAILED", "STARTED", "PENDING")}
    dataset_done = {
        dataset: sum(t["state"] == "DONE" and t["dataset"] == dataset for t in tasks)
        for dataset in datasets
    }
    mode_done = {
        mode: sum(t["state"] == "DONE" and t["mode"] == mode for t in tasks)
        for mode in modes
    }
    snapshot = {
        "campaign_root": str(args.campaign_root),
        "protocol_revision": protocol.get("implementation_revision"),
        "total_tasks": total,
        "state_counts": state_counts,
        "dataset_done_out_of_20": dataset_done,
        "mode_done_out_of_20": mode_done,
        "failed_tasks": [t for t in tasks if t["state"] == "FAILED"],
        "started_tasks": [t for t in tasks if t["state"] == "STARTED"],
        "slurm": slurm_rows(),
        "oom_logs": oom_logs,
        "full_aligned_pair_checks": pair_checks,
        "collapsed_pairs": [p for p in pair_checks if p["collapsed"]],
        "constraints": {
            "maximum_concurrent_a100_jobs": 4,
            "tf32": False,
            "ema": False,
            "test_used_for_selection": False,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    lines = [
        "# NequIP CHORUS causal campaign snapshot",
        "",
        f"- Completed: {state_counts['DONE']}/{total}",
        f"- Failed markers: {state_counts['FAILED']}",
        f"- Started without terminal marker: {state_counts['STARTED']}",
        f"- Slurm rows: {len(snapshot['slurm'])}",
        f"- OOM logs: {len(oom_logs)}",
        f"- Auditable Full/Aligned pairs: {len(pair_checks)}/20",
        f"- Collapsed Full/Aligned pairs: {len(snapshot['collapsed_pairs'])}",
        "",
        "## Completion by dataset",
        "",
    ]
    lines.extend(f"- {dataset}: {count}/20" for dataset, count in dataset_done.items())
    lines += ["", "## Completion by mode", ""]
    lines.extend(f"- {mode}: {count}/20" for mode, count in mode_done.items())
    if snapshot["failed_tasks"]:
        lines += ["", "## Failed tasks", ""]
        lines.extend(f"- `{row}`" for row in snapshot["failed_tasks"])
    if snapshot["collapsed_pairs"]:
        lines += ["", "## Full/Aligned collapse alarms", ""]
        lines.extend(f"- `{row}`" for row in snapshot["collapsed_pairs"])
    lines += [
        "",
        "Kimi may repair and resubmit this campaign under the operator policy. "
        "Any semantic model change requires a new versioned root and invalidation record.",
        "",
    ]
    args.markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
