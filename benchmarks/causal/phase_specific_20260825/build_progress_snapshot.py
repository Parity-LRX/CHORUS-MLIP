#!/usr/bin/env python3
"""Build a deterministic read-only snapshot for event-driven Kimi supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    datasets = tuple(protocol["datasets"])
    seeds = tuple(protocol["seeds"])
    probe_root = args.campaign_root / "probes"
    aligned_root = args.campaign_root / "aligned_runs"

    probe_records: list[dict] = []
    aligned_records: list[dict] = []
    for dataset in datasets:
        primary_test = "300K" if dataset == "3bpa" else "test"
        for seed in seeds:
            probe = probe_root / dataset / f"seed{seed}" / primary_test / "metrics.json"
            aligned = (
                aligned_root / dataset / f"seed{seed}" / "aligned" / "metrics.json"
            )
            if probe.is_file():
                payload = json.loads(probe.read_text())
                probe_records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "native_force_mae": payload["metrics"]["native"][
                            "force_mae_mev_per_angstrom"
                        ],
                        "aligned_force_mae": payload["metrics"]["aligned"][
                            "force_mae_mev_per_angstrom"
                        ],
                        "permuted_force_mae": payload["metrics"]["permute_mean"][
                            "force_mae_mev_per_angstrom"
                        ],
                    }
                )
            if aligned.is_file():
                payload = json.loads(aligned.read_text())
                test = payload["tests"][primary_test]
                aligned_records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "force_mae": test["force_mae_mev_per_angstrom"],
                        "parameter_count": payload["parameter_count"],
                    }
                )

    probe_datasets = [
        dataset
        for dataset in datasets
        if sum(row["dataset"] == dataset for row in probe_records) == len(seeds)
    ]
    aligned_datasets = [
        dataset
        for dataset in datasets
        if sum(row["dataset"] == dataset for row in aligned_records) == len(seeds)
    ]
    failures = sorted(
        str(path.parent.relative_to(args.campaign_root))
        for path in args.campaign_root.glob("**/FAILED")
        if path.is_file()
    )
    running = sorted(
        str(path.parent.relative_to(args.campaign_root))
        for path in args.campaign_root.glob("**/RUNNING")
        if path.is_file()
    )
    stage_path = args.campaign_root / "controller" / "stage"
    stage = stage_path.read_text().strip() if stage_path.is_file() else "not_started"
    snapshot = {
        "stage": stage,
        "probe_completed": len(probe_records),
        "aligned_completed": len(aligned_records),
        "probe_complete_datasets": probe_datasets,
        "aligned_complete_datasets": aligned_datasets,
        "failures": failures,
        "running": running,
        "probe_records": probe_records,
        "aligned_records": aligned_records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    lines = [
        "# CHORUS phase-causal campaign snapshot",
        "",
        f"- Controller stage: {stage}",
        f"- Same-checkpoint probes: {len(probe_records)}/20",
        f"- Aligned-Full trained controls: {len(aligned_records)}/20",
        f"- Complete probe datasets: {', '.join(probe_datasets) or 'none'}",
        f"- Complete aligned datasets: {', '.join(aligned_datasets) or 'none'}",
        f"- Active runs: {len(running)}",
        f"- Failure markers: {len(failures)}",
        "- Scientific protocol is frozen; Kimi audits but does not change or resubmit runs.",
        "",
    ]
    if failures:
        lines += ["## Failure markers", "", *[f"- `{x}`" for x in failures], ""]
    if probe_records:
        lines += [
            "## Completed primary-test phase probes",
            "",
            "| Dataset | Seed | Native | Aligned intervention | Permuted |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in probe_records:
            lines.append(
                f"| {row['dataset']} | {row['seed']} | {row['native_force_mae']:.3f} | "
                f"{row['aligned_force_mae']:.3f} | {row['permuted_force_mae']:.3f} |"
            )
        lines.append("")
    if aligned_records:
        lines += [
            "## Completed Aligned-Full controls",
            "",
            "| Dataset | Seed | Test Force MAE | Parameters |",
            "|---|---:|---:|---:|",
        ]
        for row in aligned_records:
            lines.append(
                f"| {row['dataset']} | {row['seed']} | {row['force_mae']:.3f} | "
                f"{row['parameter_count']} |"
            )
        lines.append("")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
