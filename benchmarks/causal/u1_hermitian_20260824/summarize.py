#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DATASETS = ("mal", "bucky", "3bpa", "zro")
MODES = ("off", "diagonal", "charge2", "full")
SEEDS = (20260616, 20260617, 20260618, 20260619, 20260620)
CONTRASTS = (
    ("u1_hermitian_closure", "full", "charge2"),
    ("cross_neighbour_interference", "full", "diagonal"),
    ("stable_self_density", "diagonal", "off"),
)


def primary_test(dataset: str) -> str:
    return "300K" if dataset == "3bpa" else "test"


def hierarchical_ci(values: dict[str, list[float]], *, samples: int = 20000) -> list[float]:
    rng = np.random.default_rng(20260824)
    datasets = tuple(values)
    draws = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in selected:
            vals = np.asarray(values[dataset], dtype=np.float64)
            dataset_means.append(float(rng.choice(vals, size=len(vals), replace=True).mean()))
        draws[i] = float(np.mean(dataset_means))
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing = []
    by_key = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            for mode in MODES:
                path = args.run_root / dataset / f"seed{seed}" / mode / "metrics.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text())
                test_name = primary_test(dataset)
                test = payload["tests"][test_name]
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "mode": mode,
                    "parameter_count": payload["parameter_count"],
                    "selected_step": payload["checkpoint_selection"]["force_mae_envelope"]["step"],
                    "validation_force_mae_mev_per_angstrom": payload["checkpoint_selection"]["force_mae_envelope"]["force_mae_mev_per_angstrom"],
                    "test_energy_mae_mev_per_atom": test["energy_mae_mev_per_atom"],
                    "test_energy_rmse_mev_per_atom": test["energy_rmse_mev_per_atom"],
                    "test_force_mae_mev_per_angstrom": test["force_mae_mev_per_angstrom"],
                    "test_force_rmse_mev_per_angstrom": test["force_rmse_mev_per_angstrom"],
                }
                records.append(row)
                by_key[(dataset, seed, mode)] = row

    audit = {
        "expected_runs": len(DATASETS) * len(SEEDS) * len(MODES),
        "completed_runs": len(records),
        "missing": missing,
    }
    (args.output_dir / "completion_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    if missing:
        raise SystemExit(f"campaign incomplete: {len(missing)} missing metrics files")

    for dataset in DATASETS:
        for seed in SEEDS:
            full = by_key[(dataset, seed, "full")]["parameter_count"]
            charge2 = by_key[(dataset, seed, "charge2")]["parameter_count"]
            if full != charge2:
                raise RuntimeError(
                    f"Full/Charge2 parameter mismatch for {dataset} seed {seed}: "
                    f"{full} != {charge2}"
                )

    with (args.output_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    effects = []
    for name, treatment, control in CONTRASTS:
        by_dataset: dict[str, list[float]] = {dataset: [] for dataset in DATASETS}
        wins = 0
        for dataset in DATASETS:
            for seed in SEEDS:
                t = by_key[(dataset, seed, treatment)]["test_force_mae_mev_per_angstrom"]
                c = by_key[(dataset, seed, control)]["test_force_mae_mev_per_angstrom"]
                log_ratio = math.log(t / c)
                by_dataset[dataset].append(log_ratio)
                wins += int(t < c)
        dataset_effects = {
            dataset: 100.0 * (1.0 - math.exp(float(np.mean(values))))
            for dataset, values in by_dataset.items()
        }
        macro_log = float(np.mean([np.mean(v) for v in by_dataset.values()]))
        ci_log = hierarchical_ci(by_dataset)
        effects.append(
            {
                "contrast": name,
                "treatment": treatment,
                "control": control,
                "paired_runs": len(DATASETS) * len(SEEDS),
                "wins": wins,
                "dataset_relative_error_reduction_percent": dataset_effects,
                "macro_relative_error_reduction_percent": 100.0 * (1.0 - math.exp(macro_log)),
                "macro_95pct_hierarchical_bootstrap_ci_percent": [
                    100.0 * (1.0 - math.exp(ci_log[1])),
                    100.0 * (1.0 - math.exp(ci_log[0])),
                ],
            }
        )

    summary = {
        "selection": "validation Force MAE; test never used for selection",
        "primary_metric": "test Force MAE",
        "causal_dag": "causal_dag.md",
        "primary_estimand": (
            "Paired end-to-end algorithmic effect of Full versus Charge2 within "
            "the frozen dataset-by-seed blocks, including optimisation and "
            "validation checkpoint selection as downstream mediators."
        ),
        "effects": effects,
        "interpretation_scope": (
            "Random-seed-paired intervention effects within the frozen CHORUS/MACE-ICTC "
            "protocol and predeclared mechanism-probing dataset panel. Full versus "
            "Charge2 identifies joint Hermitian-plus-U(1)-neutral closure, not the two "
            "properties separately and not a universal effect over all architectures."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# U(1)-Hermitian causal benchmark",
        "",
        "All checkpoints were selected by full-validation Force MAE; test labels were not used for selection.",
        "The causal DAG and identification assumptions are frozen in `causal_dag.md`.",
        "",
        "| Contrast | Wins | Macro force-MAE reduction | 95% paired hierarchical bootstrap CI |",
        "|---|---:|---:|---:|",
    ]
    for effect in effects:
        lo, hi = effect["macro_95pct_hierarchical_bootstrap_ci_percent"]
        lines.append(
            f"| {effect['treatment']} vs {effect['control']} | "
            f"{effect['wins']}/{effect['paired_runs']} | "
            f"{effect['macro_relative_error_reduction_percent']:.2f}% | "
            f"[{lo:.2f}%, {hi:.2f}%] |"
        )
    lines.extend(
        [
            "",
            "Positive reduction means the treatment has lower Force MAE. Full and Charge2 are audited to have exactly equal parameter counts within every dataset/seed pair.",
        ]
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
