#!/usr/bin/env python3
"""Validate generated frozen-phase heterogeneity tables against probe totals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = tuple(range(20260616, 20260621))
METRICS = {
    "native": "native_force_mae_mev_per_angstrom",
    "aligned": "aligned_force_mae_mev_per_angstrom",
    "permuted": "permute_mean_force_mae_mev_per_angstrom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.results / "seed_level_strata.csv").open(newline="") as handle:
        strata = list(csv.DictReader(handle))
    with args.frozen_probes.open(newline="") as handle:
        probes = list(csv.DictReader(handle))
    probe_index = {(row["dataset"], int(row["seed"])): row for row in probes}
    maximum_differences = {metric: 0.0 for metric in METRICS.values()}
    complete_quartiles = True
    for dataset in DATASETS:
        for seed in SEEDS:
            rows = [
                row
                for row in strata
                if row["dataset"] == dataset
                and int(row["seed"]) == seed
                and row["dimension"] == "reference-force quartile"
            ]
            complete_quartiles &= {row["stratum"] for row in rows} == {
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            }
            atom_count = sum(float(row["atoms"]) for row in rows)
            for source_column, aggregate_column in METRICS.items():
                reconstructed = sum(
                    float(row["atoms"]) * float(row[source_column]) for row in rows
                ) / atom_count
                expected = float(probe_index[(dataset, seed)][aggregate_column])
                maximum_differences[aggregate_column] = max(
                    maximum_differences[aggregate_column],
                    abs(reconstructed - expected),
                )
    audit = json.loads((args.results / "AUDIT.json").read_text())
    readiness = json.loads((args.results / "SCOPE_BACKBONE_READINESS.json").read_text())
    checks = {
        "raw_outcomes_complete": audit["complete"],
        "all_outcomes_finite": audit["finite"],
        "twenty_trained_model_pairs": audit["model_pairs"] == 20,
        "each_force_stratification_has_four_quartiles": complete_quartiles,
        "reconstructed_metrics_max_abs_diff": maximum_differences,
        "reconstructed_metrics_match_within_1e-6": max(maximum_differences.values())
        < 1.0e-6,
        "scope_backbone_completion_gate_active": not readiness["all_complete"],
        "formal_partial_interactions_reported": False,
    }
    if not all(
        value
        for key, value in checks.items()
        if key
        not in {
            "reconstructed_metrics_max_abs_diff",
            "formal_partial_interactions_reported",
        }
    ):
        raise RuntimeError(json.dumps(checks, indent=2))
    (args.results / "VALIDATION_REPORT.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n"
    )
    (args.results / "VALIDATION_REPORT.md").write_text(
        "# Validation report\n\n"
        "- Raw paired outcomes: complete and finite.\n"
        "- Causal units: 20 independently trained dataset-by-seed predictors.\n"
        "- Reference-force quartiles: four within each dataset and seed.\n"
        "- Weighted stratum reconstruction matches the audited aggregate frozen-probe Force MAE to below 1e-6 meV Å⁻¹.\n"
        "- Scope/backbone synthesis remains completion-gated; no partial interaction estimate was emitted.\n"
    )
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
