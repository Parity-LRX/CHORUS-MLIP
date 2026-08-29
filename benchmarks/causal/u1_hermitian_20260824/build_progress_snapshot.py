#!/usr/bin/env python3
"""Build a deterministic, read-only snapshot for the Kimi event supervisor."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def primary_test(dataset: str) -> str:
    return "300K" if dataset == "3bpa" else "test"


def relative_reduction(treatment: float, control: float) -> float:
    return 100.0 * (1.0 - treatment / control)


def last_validation_line(run_dir: Path) -> str:
    path = run_dir / "logs" / "train.log"
    if not path.is_file():
        return "training log not available"
    lines = path.read_text(errors="replace").splitlines()
    for line in reversed(lines):
        if "| val " in line:
            return line[-700:]
    return lines[-1][-700:] if lines else "training log is empty"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    datasets = tuple(protocol["datasets"])
    seeds = tuple(protocol["seeds"])
    modes = tuple(protocol["interventions"])
    contrasts = tuple(protocol["predeclared_contrasts"])

    records: dict[tuple[str, int, str], dict] = {}
    for dataset in datasets:
        for seed in seeds:
            for mode in modes:
                path = args.run_root / dataset / f"seed{seed}" / mode / "metrics.json"
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text())
                selected = payload["checkpoint_selection"]["force_mae_envelope"]
                test_name = primary_test(dataset)
                test = payload["tests"][test_name]
                records[(dataset, int(seed), mode)] = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "mode": mode,
                    "parameter_count": payload["parameter_count"],
                    "selected_step": selected["step"],
                    "validation_energy_mae_mev_per_atom": selected[
                        "energy_mae_mev_per_atom"
                    ],
                    "validation_force_mae_mev_per_angstrom": selected[
                        "force_mae_mev_per_angstrom"
                    ],
                    "test_energy_mae_mev_per_atom": test[
                        "energy_mae_mev_per_atom"
                    ],
                    "test_force_mae_mev_per_angstrom": test[
                        "force_mae_mev_per_angstrom"
                    ],
                }

    blocks: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for (dataset, seed, mode), record in records.items():
        blocks[(dataset, seed)][mode] = record
    complete_blocks = {
        key: value for key, value in blocks.items() if all(mode in value for mode in modes)
    }
    complete_datasets = [
        dataset
        for dataset in datasets
        if all((dataset, int(seed)) in complete_blocks for seed in seeds)
    ]

    running_dirs = sorted(
        marker.parent for marker in args.run_root.glob("**/RUNNING") if marker.is_file()
    )
    failed_dirs = sorted(
        marker.parent for marker in args.run_root.glob("**/FAILED") if marker.is_file()
    )
    expected = len(datasets) * len(seeds) * len(modes)

    snapshot = {
        "expected_runs": expected,
        "completed_runs": len(records),
        "complete_blocks": len(complete_blocks),
        "expected_blocks": len(datasets) * len(seeds),
        "complete_datasets": complete_datasets,
        "failed_runs": [str(path.relative_to(args.run_root)) for path in failed_dirs],
        "running_runs": [str(path.relative_to(args.run_root)) for path in running_dirs],
        "records": list(records.values()),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Deterministic CHORUS causal-campaign progress snapshot",
        "",
        f"- Completed runs: {len(records)}/{expected}",
        f"- Complete dataset-by-seed four-mode blocks: "
        f"{len(complete_blocks)}/{len(datasets) * len(seeds)}",
        f"- Complete datasets: {', '.join(complete_datasets) or 'none'}",
        f"- Active runs: {len(running_dirs)}",
        f"- Failed runs: {len(failed_dirs)}",
        "- Resource-only amendment: array concurrency was changed from two to four "
        "A100 GPUs by explicit user instruction; the scientific protocol is unchanged.",
        "",
    ]

    if running_dirs:
        lines.extend(["## Active runs", ""])
        for run_dir in running_dirs:
            lines.append(f"- `{run_dir.relative_to(args.run_root)}`")
            lines.append(f"  - `{last_validation_line(run_dir)}`")
        lines.append("")

    if complete_blocks:
        lines.extend(
            [
                "## Completed four-mode blocks",
                "",
                "All numbers below are from the single checkpoint selected by validation "
                "Force MAE. Test labels never select checkpoints.",
                "",
                "| Dataset | Seed | Contrast | Test Force-MAE reduction | Treatment/control FMAE |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for (dataset, seed), by_mode in sorted(complete_blocks.items()):
            for contrast in contrasts:
                treatment = contrast["treatment"]
                control = contrast["control"]
                t_value = by_mode[treatment]["test_force_mae_mev_per_angstrom"]
                c_value = by_mode[control]["test_force_mae_mev_per_angstrom"]
                reduction = relative_reduction(t_value, c_value)
                lines.append(
                    f"| {dataset} | {seed} | {treatment} vs {control} "
                    f"({contrast['role']}) | {reduction:+.2f}% | "
                    f"{t_value:.3f}/{c_value:.3f} meV/A |"
                )
        lines.append("")

    if failed_dirs:
        lines.extend(["## Failed runs and immutable log tails", ""])
        for run_dir in failed_dirs:
            lines.append(f"### `{run_dir.relative_to(args.run_root)}`")
            for name in ("logs/train.log", "FAILED"):
                path = run_dir / name
                if not path.is_file():
                    continue
                tail = path.read_text(errors="replace").splitlines()[-40:]
                lines.extend(["```text", *tail, "```", ""])

    if not complete_blocks:
        lines.extend(
            [
                "## Interpretation boundary",
                "",
                "No complete four-mode block exists yet. Do not infer any intervention "
                "effect from the currently completed subset.",
                "",
            ]
        )

    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
