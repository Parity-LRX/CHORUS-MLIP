#!/usr/bin/env python3
"""Independent validation of the publication-grade CHORUS DAG artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = tuple(range(20260616, 20260621))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute_trained_effect(
    runs: pd.DataFrame, effect: dict
) -> tuple[float, int]:
    subset = runs[
        (runs.backbone == effect["backbone"])
        & (runs.scope == effect["scope"])
    ].set_index(["dataset", "seed", "mode"], verify_integrity=True)
    dataset_means = []
    wins = 0
    for dataset in DATASETS:
        values = []
        for seed in SEEDS:
            treatment = float(
                subset.loc[(dataset, seed, effect["treatment"]), effect["metric"]]
            )
            control = float(
                subset.loc[(dataset, seed, effect["control"]), effect["metric"]]
            )
            value = math.log(control / treatment)
            values.append(value)
            wins += int(value > 0)
        dataset_means.append(float(np.mean(values)))
    return float(np.mean(dataset_means)), wins


def recompute_frozen_effect(
    frozen: pd.DataFrame, effect: dict
) -> tuple[float, int]:
    subset = frozen[
        (frozen.backbone == effect["backbone"])
        & (frozen.scope == effect["scope"])
    ].set_index(["dataset", "seed"], verify_integrity=True)
    dataset_means = []
    wins = 0
    for dataset in DATASETS:
        values = []
        for seed in SEEDS:
            raw = subset.loc[(dataset, seed)]
            factual = float(raw.native_force_mae)
            if effect["control"] == "permute_mean":
                intervention = float(
                    np.mean(
                        [raw[f"permute_{repeat}_force_mae"] for repeat in range(4)]
                    )
                )
            else:
                intervention = float(raw[f"{effect['control']}_force_mae"])
            value = math.log(intervention / factual)
            values.append(value)
            wins += int(value > 0)
        dataset_means.append(float(np.mean(values)))
    return float(np.mean(dataset_means)), wins


def main() -> None:
    args = parse_args()
    artifact = args.artifact
    required = (
        "DATA_AUDIT.json",
        "causal_analysis.json",
        "trained_runs.csv",
        "frozen_probe_runs.csv",
        "trained_paired_effects.csv",
        "frozen_paired_effects.csv",
        "paired_interactions.csv",
        "phase_dose_response.csv",
        "trained_effect_summary.csv",
        "frozen_effect_summary.csv",
        "interaction_summary.csv",
        "REPORT.md",
        "CAUSAL_IDENTIFICATION.md",
        "CLAIM_EVIDENCE_MATRIX.md",
        "PEARL_CAUSAL_LADDER.md",
        "SOURCE_MANIFEST.json",
        "CODE_MANIFEST.json",
        "causal_dag.svg",
        "trained_component_forest.svg",
        "frozen_intervention_forest.svg",
        "phase_dose_response.svg",
        "DONE",
    )
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    missing = [name for name in required if not (artifact / name).is_file()]
    check("required_files", not missing, f"missing={missing}")
    if missing:
        raise SystemExit(2)

    payload = json.loads((artifact / "causal_analysis.json").read_text())
    audit = json.loads((artifact / "DATA_AUDIT.json").read_text())
    runs = pd.read_csv(artifact / "trained_runs.csv")
    frozen = pd.read_csv(artifact / "frozen_probe_runs.csv")
    check("declared_audit", bool(audit.get("passed")), json.dumps(audit))
    check("trained_cell_count", len(runs) == 240, f"found={len(runs)}")
    check("frozen_cell_count", len(frozen) == 80, f"found={len(frozen)}")
    check(
        "trained_unique_keys",
        not runs.duplicated(["backbone", "scope", "dataset", "seed", "mode"]).any(),
        "backbone/scope/dataset/seed/mode",
    )
    check(
        "frozen_unique_keys",
        not frozen.duplicated(["backbone", "scope", "dataset", "seed"]).any(),
        "backbone/scope/dataset/seed",
    )
    check(
        "test_not_selected",
        not runs.test_used_for_selection.astype(str).str.lower().isin(["true", "1"]).any(),
        "all 240 cells",
    )

    trained_differences = []
    trained_win_mismatches = []
    for effect in payload["trained_effects"]:
        value, wins = recompute_trained_effect(runs, effect)
        trained_differences.append(abs(value - effect["macro_mean_log_benefit"]))
        trained_win_mismatches.append(wins != effect["wins"])
    check(
        "trained_point_estimates_recomputed",
        max(trained_differences, default=0.0) <= 1.0e-12,
        f"max_log_difference={max(trained_differences, default=0.0):.3e}",
    )
    check(
        "trained_wins_recomputed",
        not any(trained_win_mismatches),
        f"mismatches={sum(trained_win_mismatches)}",
    )

    frozen_differences = []
    frozen_win_mismatches = []
    for effect in payload["frozen_effects"]:
        value, wins = recompute_frozen_effect(frozen, effect)
        frozen_differences.append(abs(value - effect["macro_mean_log_benefit"]))
        frozen_win_mismatches.append(wins != effect["wins"])
    check(
        "frozen_point_estimates_recomputed",
        max(frozen_differences, default=0.0) <= 1.0e-12,
        f"max_log_difference={max(frozen_differences, default=0.0):.3e}",
    )
    check(
        "frozen_wins_recomputed",
        not any(frozen_win_mismatches),
        f"mismatches={sum(frozen_win_mismatches)}",
    )

    common_shift = float(frozen.common_shift_max_force_change.max())
    check(
        "common_shift_negative_control",
        common_shift <= 0.02,
        f"maximum_pointwise_force_change={common_shift:.6g} meV/angstrom",
    )
    dose = pd.read_csv(artifact / "phase_dose_response.csv")
    monotonic_failures = []
    for (backbone, scope), group in dose.groupby(["backbone", "scope"]):
        values = group.sort_values("lambda").force_mae_ratio_to_factual.to_numpy()
        if not np.all(values[:-1] >= values[1:] - 1.0e-12):
            monotonic_failures.append(f"{backbone}/{scope}")
    check("macro_dose_monotonic", not monotonic_failures, str(monotonic_failures))
    dose_count_mismatches = []
    mode_order = ("aligned", "dose_0.25", "dose_0.50", "dose_0.75", "native")
    for (backbone, scope), group in frozen.groupby(["backbone", "scope"]):
        ratios = np.column_stack(
            [
                group[f"{mode}_force_mae"].to_numpy(dtype=float)
                / group["native_force_mae"].to_numpy(dtype=float)
                for mode in mode_order
            ]
        )
        observed = int(
            np.all(ratios[:, :-1] >= ratios[:, 1:] - 1.0e-12, axis=1).sum()
        )
        key = f"{backbone}/{scope}"
        declared = int(
            payload["dose_monotonicity"][key]["paired_cells_monotonic"]
        )
        if observed != declared:
            dose_count_mismatches.append(f"{key}:{observed}!={declared}")
    check(
        "cell_dose_monotonic_counts_recomputed",
        not dose_count_mismatches,
        str(dose_count_mismatches),
    )

    source_manifest = json.loads((artifact / "SOURCE_MANIFEST.json").read_text())
    source_failures = []
    for record in source_manifest.get("files", []):
        source = Path(record["path"])
        if not source.is_file():
            source_failures.append(f"missing:{source}")
            continue
        observed = sha256(source)
        if observed != record["sha256"]:
            source_failures.append(f"hash:{source}")
    check(
        "source_manifest_integrity",
        bool(source_manifest.get("files")) and not source_failures,
        f"files={len(source_manifest.get('files', []))}; failures={source_failures}",
    )
    code_manifest = json.loads((artifact / "CODE_MANIFEST.json").read_text())
    code_failures = []
    for record in code_manifest.get("files", []):
        source = Path(record["path"])
        if not source.is_file():
            code_failures.append(f"missing:{source}")
            continue
        if sha256(source) != record["sha256"]:
            code_failures.append(f"hash:{source}")
    check(
        "code_manifest_integrity",
        bool(code_manifest.get("files")) and not code_failures,
        f"files={len(code_manifest.get('files', []))}; failures={code_failures}",
    )

    hashes = {
        name: sha256(artifact / name)
        for name in required
        if name != "DONE"
    }
    passed = all(bool(item["passed"]) for item in checks)
    report = {
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "sha256": hashes,
        "validation_boundary": (
            "point estimates, cell completeness, selection hygiene, negative "
            "control, monotonic dose response, and artifact integrity"
        ),
    }
    (artifact / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Independent validation report",
        "",
        f"Overall status: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for item in checks:
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(
            f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |"
        )
    lines.extend(
        [
            "",
            "Validation independently recomputes every trained and frozen point estimate from the exported run tables. Bootstrap intervals are not treated as independent evidence because they are generated by the declared deterministic analysis pipeline.",
            "",
        ]
    )
    (artifact / "VALIDATION_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
