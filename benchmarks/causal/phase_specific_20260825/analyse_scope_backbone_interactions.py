#!/usr/bin/env python3
"""Completion-gated synthesis of CHORUS scope and backbone interactions.

The script deliberately refuses to report an interaction from a partial cell
matrix.  It accepts the audited MACE-Final tables, discovers MACE-Persistent and
NequIP metrics, validates the declared selection rule, and then estimates paired
log-error effects with hierarchical bootstrap intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = tuple(range(20260616, 20260621))
MACE_MODES = ("aligned", "full")
NEQUIP_FINAL_MODES = ("off", "diagonal", "aligned", "full")
NEQUIP_PERSISTENT_MODES = ("aligned", "full")
NEQUIP_DATASETS = {
    "mal": "xxmd_mal",
    "bucky": "md22_buckyball",
    "3bpa": "3bpa",
    "zro": "zr-o",
}
PRIMARY_TEST_LOG = {
    "mal": "test_test.log",
    "bucky": "test_test.log",
    "3bpa": "test_300K.log",
    "zro": "test_test.log",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mace-final-primary", type=Path, required=True)
    parser.add_argument("--mace-final-aligned", type=Path, required=True)
    parser.add_argument("--mace-persistent-root", type=Path, required=True)
    parser.add_argument("--nequip-final-root", type=Path, required=True)
    parser.add_argument("--nequip-persistent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--rng-seed", type=int, default=20260828)
    parser.add_argument(
        "--readiness-only",
        action="store_true",
        help="write READINESS.json and exit successfully without estimating effects",
    )
    return parser.parse_args()


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def normalize_row(
    *,
    backbone: str,
    scope: str,
    dataset: str,
    seed: int,
    mode: str,
    parameter_count: int,
    selected_step: int | None,
    validation_force_mae: float,
    validation_force_rmse: float,
    validation_energy_mae: float,
    validation_energy_rmse: float,
    test_force_mae: float,
    test_force_rmse: float,
    test_energy_mae: float,
    test_energy_rmse: float,
    source: Path,
) -> dict[str, Any]:
    row = {
        "backbone": backbone,
        "scope": scope,
        "dataset": dataset,
        "seed": int(seed),
        "mode": mode,
        "parameter_count": int(parameter_count),
        "selected_step": selected_step,
        "validation_force_mae": float(validation_force_mae),
        "validation_force_rmse": float(validation_force_rmse),
        "validation_energy_mae": float(validation_energy_mae),
        "validation_energy_rmse": float(validation_energy_rmse),
        "test_force_mae": float(test_force_mae),
        "test_force_rmse": float(test_force_rmse),
        "test_energy_mae": float(test_energy_mae),
        "test_energy_rmse": float(test_energy_rmse),
        "selection_rule": "minimum complete-validation Force MAE",
        "test_used_for_selection": False,
        "source": str(source),
    }
    numeric = (
        row["validation_force_mae"],
        row["validation_force_rmse"],
        row["validation_energy_mae"],
        row["validation_energy_rmse"],
        row["test_force_mae"],
        row["test_force_rmse"],
        row["test_energy_mae"],
        row["test_energy_rmse"],
    )
    if not all(finite(value) and float(value) > 0 for value in numeric):
        raise ValueError(f"non-positive or non-finite metric in {source}")
    return row


def read_mace_final(primary: Path, aligned: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (primary, aligned):
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw["mode"] not in MACE_MODES:
                    continue
                if raw["test_used_for_selection"].strip().lower() not in ("false", "0"):
                    raise ValueError(f"test selection violation in {path}")
                rows.append(
                    normalize_row(
                        backbone="MACE-ICTC",
                        scope="Final",
                        dataset=raw["dataset"],
                        seed=int(raw["seed"]),
                        mode=raw["mode"],
                        parameter_count=int(raw["parameter_count"]),
                        selected_step=int(raw["selected_step"]),
                        validation_force_mae=float(raw["validation_force_mae"]),
                        validation_force_rmse=float(raw["validation_force_rmse"]),
                        validation_energy_mae=float(raw["validation_energy_mae"]),
                        validation_energy_rmse=float(raw["validation_energy_rmse"]),
                        test_force_mae=float(raw["test_force_mae"]),
                        test_force_rmse=float(raw["test_force_rmse"]),
                        test_energy_mae=float(raw["test_energy_mae"]),
                        test_energy_rmse=float(raw["test_energy_rmse"]),
                        source=path,
                    )
                )
    return rows


def read_mace_persistent(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for mode in MACE_MODES:
                path = root / "mace" / "runs" / dataset / f"seed{seed}" / mode / "metrics.json"
                if not path.is_file():
                    continue
                raw = json.loads(path.read_text())
                if raw.get("phase_scope") != "persistent":
                    raise ValueError(f"wrong phase scope in {path}")
                selection = raw["checkpoint_selection"]
                if selection.get("test_used_for_selection") is not False:
                    raise ValueError(f"test selection violation in {path}")
                validation = selection["force_mae_envelope"]
                tests = raw["tests"]
                test_name = "300K" if dataset == "3bpa" else "test"
                test = tests[test_name]
                rows.append(
                    normalize_row(
                        backbone="MACE-ICTC",
                        scope="Persistent",
                        dataset=dataset,
                        seed=seed,
                        mode=mode,
                        parameter_count=int(raw["parameter_count"]),
                        selected_step=int(validation["step"]),
                        validation_force_mae=float(validation["force_mae_mev_per_angstrom"]),
                        validation_force_rmse=float(validation["force_rmse_mev_per_angstrom"]),
                        validation_energy_mae=float(validation["energy_mae_mev_per_atom"]),
                        validation_energy_rmse=float(validation["energy_rmse_mev_per_atom"]),
                        test_force_mae=float(test["force_mae_mev_per_angstrom"]),
                        test_force_rmse=float(test["force_rmse_mev_per_angstrom"]),
                        test_energy_mae=float(test["energy_mae_mev_per_atom"]),
                        test_energy_rmse=float(test["energy_rmse_mev_per_atom"]),
                        source=path,
                    )
                )
    return rows


def parse_test_log(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    matches = re.findall(
        r"^\s*(f_mae|f_rmse|e/N_mae|e/N_rmse)\s*=\s*([0-9.eE+\-]+)",
        text,
        flags=re.MULTILINE,
    )
    values = {name: float(value) * 1000.0 for name, value in matches}
    required = {"f_mae", "f_rmse", "e/N_mae", "e/N_rmse"}
    if not required.issubset(values):
        raise ValueError(f"incomplete test metrics in {path}: {sorted(values)}")
    return values


def parse_protocol_steps(path: Path, epoch: int) -> int | None:
    if not path.is_file():
        return None
    match = re.search(r"^batches_per_epoch:\s*(\d+)\s*$", path.read_text(), re.MULTILINE)
    return int(match.group(1)) * epoch if match else None


def parse_parameter_count(path: Path) -> int:
    text = path.read_text(errors="replace")
    match = re.search(r"Number of trainable weights:\s*(\d+)", text)
    if not match:
        raise ValueError(f"parameter count missing from {path}")
    return int(match.group(1))


def parse_nequip_molecular(
    work: Path,
    backbone: str,
    scope: str,
    dataset: str,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    metrics_path = work / "run" / "metrics_epoch.csv"
    if not metrics_path.is_file() or not (work / "DONE").is_file():
        raise FileNotFoundError(work)
    with metrics_path.open(newline="") as handle:
        epochs = list(csv.DictReader(handle))
    if not epochs:
        raise ValueError(f"empty validation history: {metrics_path}")
    selected = min(
        epochs,
        key=lambda row: (float(row["validation_f_mae"]), int(row["epoch"])),
    )
    epoch = int(selected["epoch"])
    test = parse_test_log(work / PRIMARY_TEST_LOG[dataset])
    return normalize_row(
        backbone=backbone,
        scope=scope,
        dataset=dataset,
        seed=seed,
        mode=mode,
        parameter_count=parse_parameter_count(work / "train.log"),
        selected_step=parse_protocol_steps(work / "protocol.yaml", epoch),
        validation_force_mae=float(selected["validation_f_mae"]) * 1000.0,
        validation_force_rmse=float(selected["validation_f_rmse"]) * 1000.0,
        validation_energy_mae=float(selected["validation_e/N_mae"]) * 1000.0,
        validation_energy_rmse=float(selected["validation_e/N_rmse"]) * 1000.0,
        test_force_mae=test["f_mae"],
        test_force_rmse=test["f_rmse"],
        test_energy_mae=test["e/N_mae"],
        test_energy_rmse=test["e/N_rmse"],
        source=metrics_path,
    )


def parse_nequip_solid(
    path: Path,
    backbone: str,
    scope: str,
    dataset: str,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if raw.get("test_used_for_selection") is not False:
        raise ValueError(f"test selection violation in {path}")
    validation = raw["independent_validation_envelopes"]["force_mae"]
    test = raw["test"]
    return normalize_row(
        backbone=backbone,
        scope=scope,
        dataset=dataset,
        seed=seed,
        mode=mode,
        parameter_count=int(raw["trainable_parameters"]),
        selected_step=int(raw["selected_step"]),
        validation_force_mae=float(validation["force_mae_mev_per_angstrom"]),
        validation_force_rmse=float(validation["force_rmse_mev_per_angstrom"]),
        validation_energy_mae=float(validation["energy_mae_mev_per_atom"]),
        validation_energy_rmse=float(validation["energy_rmse_mev_per_atom"]),
        test_force_mae=float(test["force_mae_mev_per_angstrom"]),
        test_force_rmse=float(test["force_rmse_mev_per_angstrom"]),
        test_energy_mae=float(test["energy_mae_mev_per_atom"]),
        test_energy_rmse=float(test["energy_rmse_mev_per_atom"]),
        source=path,
    )


def discover_work(base: Path, mode: str) -> Path | None:
    candidates = sorted(
        path.parent.parent
        for path in base.rglob("run/metrics_epoch.csv")
        if path.parent.parent.name.startswith("e3nn_chorus_")
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        expected = [path for path in candidates if mode in path.name]
        if len(expected) == 1:
            return expected[0]
        raise ValueError(f"ambiguous NequIP work directories under {base}: {candidates}")
    return candidates[0]


def read_nequip(
    root: Path,
    scope: str,
    modes: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        raw_dataset = NEQUIP_DATASETS[dataset]
        for seed_index, seed in enumerate(SEEDS):
            for mode_index, mode in enumerate(NEQUIP_FINAL_MODES):
                if mode not in modes:
                    continue
                task_id = seed_index * 16 + dataset_index * 4 + mode_index
                if not (root / "status" / f"task_{task_id}" / "DONE").is_file():
                    continue
                base = root / f"seed_{seed}" / mode / raw_dataset
                if dataset == "zro":
                    candidates = [
                        path
                        for path in base.rglob("metrics.json")
                        if "train_only_energy_calibration" not in path.parts
                    ]
                    if len(candidates) != 1:
                        raise ValueError(f"expected one ZrO metrics file under {base}, found {candidates}")
                    rows.append(
                        parse_nequip_solid(
                            candidates[0], "NequIP", scope, dataset, seed, mode
                        )
                    )
                else:
                    work = discover_work(base, mode)
                    if work is None:
                        continue
                    rows.append(
                        parse_nequip_molecular(
                            work, "NequIP", scope, dataset, seed, mode
                        )
                    )
    return rows


def expected_keys(backbone: str, scope: str, modes: Iterable[str]) -> set[tuple[Any, ...]]:
    return {
        (backbone, scope, dataset, seed, mode)
        for dataset in DATASETS
        for seed in SEEDS
        for mode in modes
    }


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["backbone"], row["scope"], row["dataset"], row["seed"], row["mode"])
        if key in result:
            raise ValueError(f"duplicate causal cell: {key}")
        result[key] = row
    return result


def readiness_section(
    name: str,
    rows: list[dict[str, Any]],
    expected: set[tuple[Any, ...]],
) -> dict[str, Any]:
    found = set(index_rows(rows))
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    return {
        "name": name,
        "complete": not missing and not extra,
        "found": len(found),
        "expected": len(expected),
        "missing": [list(key) for key in missing],
        "extra": [list(key) for key in extra],
    }


def hierarchical_bootstrap(
    values: list[dict[str, Any]],
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    datasets = list(DATASETS)
    by_dataset = {
        dataset: np.asarray(
            [row["value"] for row in values if row["dataset"] == dataset],
            dtype=float,
        )
        for dataset in datasets
    }
    result = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for dataset in sampled:
            array = by_dataset[str(dataset)]
            means.append(float(rng.choice(array, size=len(array), replace=True).mean()))
        result[index] = float(np.mean(means))
    return result


def effect_summary(
    rows: list[dict[str, Any]],
    *,
    backbone: str,
    scope: str,
    treatment: str,
    control: str,
    metric: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indexed = index_rows(rows)
    pairs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            treatment_row = indexed[(backbone, scope, dataset, seed, treatment)]
            control_row = indexed[(backbone, scope, dataset, seed, control)]
            treatment_value = float(treatment_row[metric])
            control_value = float(control_row[metric])
            if treatment_row["parameter_count"] != control_row["parameter_count"] and {
                treatment,
                control,
            } == {"full", "aligned"}:
                raise ValueError(
                    f"Full/Aligned parameter mismatch for {backbone}/{scope}/{dataset}/{seed}"
                )
            pairs.append(
                {
                    "backbone": backbone,
                    "scope": scope,
                    "dataset": dataset,
                    "seed": seed,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "treatment_value": treatment_value,
                    "control_value": control_value,
                    "value": math.log(control_value / treatment_value),
                }
            )
    bootstrap = hierarchical_bootstrap(pairs, replicates, rng)
    dataset_effects = {
        dataset: float(
            np.mean([row["value"] for row in pairs if row["dataset"] == dataset])
        )
        for dataset in DATASETS
    }
    mean_log = float(np.mean(list(dataset_effects.values())))
    return (
        {
            "name": f"{backbone} {scope} {treatment} vs {control}",
            "backbone": backbone,
            "scope": scope,
            "treatment": treatment,
            "control": control,
            "metric": metric,
            "paired_units": len(pairs),
            "wins": int(sum(row["value"] > 0 for row in pairs)),
            "macro_mean_log_benefit": mean_log,
            "macro_relative_error_reduction_percent": 100.0 * (1.0 - math.exp(-mean_log)),
            "macro_95pct_hierarchical_bootstrap_ci_percent": [
                float(100.0 * (1.0 - math.exp(-value)))
                for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "dataset_relative_error_reduction_percent": {
                dataset: 100.0 * (1.0 - math.exp(-value))
                for dataset, value in dataset_effects.items()
            },
        },
        pairs,
    )


def interaction_summary(
    first_pairs: list[dict[str, Any]],
    second_pairs: list[dict[str, Any]],
    name: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first = {(row["dataset"], row["seed"]): row for row in first_pairs}
    second = {(row["dataset"], row["seed"]): row for row in second_pairs}
    if set(first) != set(second):
        raise ValueError(f"interaction pair mismatch for {name}")
    rows = [
        {
            "dataset": key[0],
            "seed": key[1],
            "value": second[key]["value"] - first[key]["value"],
        }
        for key in sorted(first)
    ]
    bootstrap = hierarchical_bootstrap(rows, replicates, rng)
    dataset_effects = {
        dataset: float(np.mean([row["value"] for row in rows if row["dataset"] == dataset]))
        for dataset in DATASETS
    }
    mean_log = float(np.mean(list(dataset_effects.values())))
    return (
        {
            "name": name,
            "definition": "second paired log benefit minus first paired log benefit",
            "paired_units": len(rows),
            "positive_pairs": int(sum(row["value"] > 0 for row in rows)),
            "macro_log_interaction": mean_log,
            "benefit_ratio_change_percent": 100.0 * math.expm1(mean_log),
            "benefit_ratio_change_95pct_hierarchical_bootstrap_ci_percent": [
                float(100.0 * math.expm1(value))
                for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "dataset_benefit_ratio_change_percent": {
                dataset: 100.0 * math.expm1(value)
                for dataset, value in dataset_effects.items()
            },
        },
        rows,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path, effects: list[dict[str, Any]], interactions: list[dict[str, Any]]) -> None:
    lines = [
        "# CHORUS scope and backbone causal synthesis",
        "",
        "All entries use one validation-Force-MAE-selected checkpoint per trained model. Positive reductions favour Full over the named control. Frames and atoms are not treated as independent causal units.",
        "",
        "## Within-configuration effects",
        "",
        "| Contrast | Error reduction | 95% hierarchical-bootstrap interval | Wins |",
        "|---|---:|---:|---:|",
    ]
    for effect in effects:
        low, high = effect["macro_95pct_hierarchical_bootstrap_ci_percent"]
        lines.append(
            f"| {effect['name']} | {effect['macro_relative_error_reduction_percent']:.2f}% | "
            f"[{low:.2f}, {high:.2f}]% | {effect['wins']}/{effect['paired_units']} |"
        )
    lines.extend(
        [
            "",
            "## Effect modification",
            "",
            "| Interaction | Benefit-ratio change | 95% hierarchical-bootstrap interval | Positive pairs |",
            "|---|---:|---:|---:|",
        ]
    )
    for effect in interactions:
        low, high = effect["benefit_ratio_change_95pct_hierarchical_bootstrap_ci_percent"]
        lines.append(
            f"| {effect['name']} | {effect['benefit_ratio_change_percent']:.2f}% | "
            f"[{low:.2f}, {high:.2f}]% | {effect['positive_pairs']}/{effect['paired_units']} |"
        )
    lines.extend(
        [
            "",
            "The propagation-scope interaction compares Persistent and Final inside one backbone. The backbone interaction compares two complete bundled configurations; it is evidence of transportability or effect modification, not a pure intervention on backbone identity alone.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    mace_final = read_mace_final(args.mace_final_primary, args.mace_final_aligned)
    mace_persistent = read_mace_persistent(args.mace_persistent_root)
    nequip_final = read_nequip(args.nequip_final_root, "Final", NEQUIP_FINAL_MODES)
    nequip_persistent = read_nequip(
        args.nequip_persistent_root, "Persistent", NEQUIP_PERSISTENT_MODES
    )
    sections = [
        readiness_section(
            "MACE-Final",
            mace_final,
            expected_keys("MACE-ICTC", "Final", MACE_MODES),
        ),
        readiness_section(
            "MACE-Persistent",
            mace_persistent,
            expected_keys("MACE-ICTC", "Persistent", MACE_MODES),
        ),
        readiness_section(
            "NequIP-Final",
            nequip_final,
            expected_keys("NequIP", "Final", NEQUIP_FINAL_MODES),
        ),
        readiness_section(
            "NequIP-Persistent",
            nequip_persistent,
            expected_keys("NequIP", "Persistent", NEQUIP_PERSISTENT_MODES),
        ),
    ]
    readiness = {
        "schema_version": 1,
        "all_complete": all(section["complete"] for section in sections),
        "sections": sections,
    }
    (args.output / "READINESS.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(readiness, indent=2, sort_keys=True))
    if args.readiness_only:
        return
    if not readiness["all_complete"]:
        raise SystemExit(3)

    effects: list[dict[str, Any]] = []
    pair_tables: dict[str, list[dict[str, Any]]] = {}
    for rows, backbone, scope, treatment, control in (
        (mace_final, "MACE-ICTC", "Final", "full", "aligned"),
        (mace_persistent, "MACE-ICTC", "Persistent", "full", "aligned"),
        (nequip_final, "NequIP", "Final", "diagonal", "off"),
        (nequip_final, "NequIP", "Final", "aligned", "diagonal"),
        (nequip_final, "NequIP", "Final", "full", "aligned"),
        (nequip_persistent, "NequIP", "Persistent", "full", "aligned"),
    ):
        effect, pairs = effect_summary(
            rows,
            backbone=backbone,
            scope=scope,
            treatment=treatment,
            control=control,
            metric="test_force_mae",
            replicates=args.bootstrap,
            rng=rng,
        )
        effects.append(effect)
        pair_tables[effect["name"]] = pairs

    interactions: list[dict[str, Any]] = []
    interaction_pairs: dict[str, list[dict[str, Any]]] = {}
    for first_name, second_name, name in (
        (
            "MACE-ICTC Final full vs aligned",
            "MACE-ICTC Persistent full vs aligned",
            "MACE-ICTC Persistent-minus-Final phase interaction",
        ),
        (
            "NequIP Final full vs aligned",
            "NequIP Persistent full vs aligned",
            "NequIP Persistent-minus-Final phase interaction",
        ),
        (
            "MACE-ICTC Final full vs aligned",
            "NequIP Final full vs aligned",
            "NequIP-minus-MACE Final phase-effect replication contrast",
        ),
        (
            "MACE-ICTC Persistent full vs aligned",
            "NequIP Persistent full vs aligned",
            "NequIP-minus-MACE Persistent phase-effect replication contrast",
        ),
    ):
        effect, pairs = interaction_summary(
            pair_tables[first_name],
            pair_tables[second_name],
            name,
            args.bootstrap,
            rng,
        )
        interactions.append(effect)
        interaction_pairs[name] = pairs

    payload = {
        "schema_version": 1,
        "primary_outcome": "test Force MAE at the validation-Force-MAE-selected checkpoint",
        "effects": effects,
        "interactions": interactions,
        "readiness": readiness,
        "analysis": {
            "bootstrap_replicates": args.bootstrap,
            "rng_seed": args.rng_seed,
            "causal_unit": "independently trained dataset-by-seed model",
        },
    }
    (args.output / "scope_backbone_interactions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    all_pairs = [row for rows in pair_tables.values() for row in rows]
    all_interactions = [row for rows in interaction_pairs.values() for row in rows]
    write_csv(
        args.output / "trained_runs.csv",
        mace_final + mace_persistent + nequip_final + nequip_persistent,
    )
    write_csv(args.output / "paired_effects.csv", all_pairs)
    write_csv(args.output / "paired_interactions.csv", all_interactions)
    write_report(args.output, effects, interactions)


if __name__ == "__main__":
    main()
