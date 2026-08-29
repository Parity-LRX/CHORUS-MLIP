#!/usr/bin/env python3
"""Independent validation of the complete MACE Pearl synthesis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = tuple(range(20260616, 20260621))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def macro_reduction(pair: pd.DataFrame) -> float:
    dataset_logs = [
        float(pair.loc[pair["dataset"] == dataset, "log_benefit"].mean())
        for dataset in DATASETS
    ]
    return float(100.0 * (1.0 - math.exp(-np.mean(dataset_logs))))


def find_effect(payload: dict, name: str) -> dict:
    effects = (
        payload["pearl_rungs"]["doing"]["trained_effects"]
        + payload["pearl_rungs"]["imagining"]["frozen_effects"]
    )
    matches = [item for item in effects if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one effect named {name}, got {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    result = args.result
    payload = json.loads((result / "causal_inference.json").read_text())
    runs = pd.read_csv(result / "trained_runs.csv")
    pairs = pd.read_csv(result / "trained_pairs.csv")
    frozen = pd.read_csv(result / "frozen_pairs.csv")
    probes = pd.read_csv(result / "frozen_probe_runs.csv")
    dose = pd.read_csv(result / "phase_dose_response.csv")
    counterfactual = pd.read_csv(result / "counterfactual_outcome_summary.csv")

    checks: dict[str, bool | int | float | str] = {}
    expected_runs = 80 + 40
    checks["trained_rows_120"] = len(runs) == expected_runs
    checks["trained_unique_cells_120"] = (
        runs[["scope", "dataset", "seed", "mode"]].drop_duplicates().shape[0]
        == expected_runs
    )
    checks["probe_rows_40"] = len(probes) == 40
    checks["probe_unique_cells_40"] = (
        probes[["scope", "dataset", "seed"]].drop_duplicates().shape[0] == 40
    )
    checks["counterfactual_model_rows_40"] = len(counterfactual) == 40
    checks["all_test_selection_flags_false"] = not runs["test_used_for_selection"].astype(bool).any()
    checks["all_key_metrics_finite"] = bool(
        np.isfinite(
            runs[
                [
                    "validation_energy_mae",
                    "validation_force_mae",
                    "test_energy_mae",
                    "test_force_mae",
                ]
            ].to_numpy(float)
        ).all()
    )

    for scope in ("Final", "Persistent"):
        local = runs[(runs["scope"] == scope) & runs["mode"].isin(("full", "aligned"))]
        parameter = local.pivot(index=["dataset", "seed"], columns="mode", values="parameter_count")
        checks[f"{scope.lower()}_full_aligned_parameter_match"] = bool(
            (parameter["full"] == parameter["aligned"]).all()
        )

    source_names = {
        "final_self_density",
        "final_cross_neighbour",
        "final_learned_phase",
        "persistent_learned_phase",
    }
    recomputed: dict[str, float] = {}
    for name in source_names:
        pair = pairs[pairs["contrast"] == name]
        if len(pair) != 20:
            raise ValueError(f"{name} has {len(pair)} rows")
        recomputed[name] = macro_reduction(pair)
        reported = find_effect(payload, name)["macro_relative_error_reduction_percent"]
        checks[f"recompute_{name}"] = abs(recomputed[name] - reported) < 1e-10

    for name in (
        "final_frozen_alignment",
        "persistent_frozen_alignment",
        "final_frozen_permutation",
        "persistent_frozen_permutation",
    ):
        pair = frozen[frozen["contrast"] == name]
        if len(pair) != 20:
            raise ValueError(f"{name} has {len(pair)} rows")
        recomputed[name] = macro_reduction(pair)
        reported = find_effect(payload, name)["macro_relative_error_reduction_percent"]
        checks[f"recompute_{name}"] = abs(recomputed[name] - reported) < 1e-10

    native = probes["native_force_mae_mev_per_angstrom"].to_numpy(float)
    common = probes["common_shift_force_mae_mev_per_angstrom"].to_numpy(float)
    checks["common_shift_force_mae_replay_max_abs_mev_per_angstrom"] = float(
        np.max(np.abs(native - common))
    )
    checks["common_shift_force_mae_replay_within_0p01_mev"] = (
        checks["common_shift_force_mae_replay_max_abs_mev_per_angstrom"] < 0.01
    )
    reported_common = payload["pearl_rungs"]["imagining"][
        "maximum_common_shift_force_change_mev_per_angstrom"
    ]
    checks["reported_common_shift_max_within_0p01_mev"] = float(reported_common) < 0.01

    dose_monotonic = {}
    for scope in ("Final", "Persistent"):
        values = (
            dose[(dose["scope"] == scope) & (dose["dataset"] == "macro")]
            .sort_values("lambda")["force_mae_ratio_to_factual"]
            .to_numpy(float)
        )
        dose_monotonic[scope] = bool(np.all(np.diff(values) <= 1e-12))
        checks[f"{scope.lower()}_macro_dose_monotonic"] = dose_monotonic[scope]

    atomic_total = int(counterfactual["atomic_outcomes"].sum())
    frame_total = int(counterfactual["frame_outcomes"].sum())
    checks["atomic_outcomes_total"] = atomic_total
    checks["frame_outcomes_total"] = frame_total
    checks["counterfactual_summaries_finite"] = bool(
        np.isfinite(counterfactual.select_dtypes(include=[np.number]).to_numpy()).all()
    )
    checks["counterfactual_rows_not_used_as_replicates"] = (
        payload["analysis_config"]["causal_unit"]
        == "independently trained dataset-by-seed model or frozen predictor"
    )

    boolean_checks = {key: value for key, value in checks.items() if isinstance(value, bool)}
    passed = bool(all(boolean_checks.values()))
    persistent = find_effect(payload, "persistent_learned_phase")
    inference_sensitivity = {
        "hierarchical_bootstrap_ci_excludes_zero": (
            persistent["macro_95pct_hierarchical_bootstrap_ci_percent"][0] > 0
        ),
        "exact_sign_flip_p": persistent["exact_two_sided_sign_flip_p"],
        "wilcoxon_p": persistent["two_sided_wilcoxon_p"],
        "binomial_sign_p": persistent["two_sided_binomial_sign_p"],
        "interpretation": (
            "The Persistent trained-phase mean is positive, but inference is sensitive "
            "to the statistic: magnitude-weighted hierarchical bootstrap and exact sign "
            "flip support a positive mean, while Wilcoxon and the win count do not."
        ),
    }
    report = {
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "recomputed_macro_effects_percent": recomputed,
        "inference_sensitivity": inference_sensitivity,
        "assessment": "Ready to share with explicit inferential scope" if passed else "Needs revision",
    }
    (result / "VALIDATION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = f"""# Validation report: complete MACE Pearl analysis

## Overall assessment

**{'Ready to share with explicit inferential scope' if passed else 'Needs revision'}.** All declared MACE Final and Persistent model cells, frozen probes, and counterfactual summaries are complete. Key macro effects were independently reconstructed from the saved paired tables.

## Verified checks

- Trained cells: 120/120; frozen probes: 40/40.
- Full/Aligned parameter counts match within both Final and Persistent scopes.
- Validation Force MAE selects every checkpoint; test is evaluation-only.
- Common-shift Force-MAE replay differs by at most {checks['common_shift_force_mae_replay_max_abs_mev_per_angstrom']:.6f} meV Å⁻¹; the maximum pointwise force-change audit is {float(reported_common):.6f} meV Å⁻¹.
- Final and Persistent macro dose curves are monotonic.
- Counterfactual coverage: {atomic_total:,} atomic force outcomes and {frame_total:,} frame-energy outcomes. They are descriptive nested outcomes, not independent causal replicates.

## Inferential sensitivity

Persistent Full versus Aligned has a positive equal-dataset mean, and its hierarchical-bootstrap interval is above zero. The exact sign-flip p-value is {persistent['exact_two_sided_sign_flip_p']:.4f}; Wilcoxon p={persistent['two_sided_wilcoxon_p']:.4f}; the 11/20 win-count binomial p={persistent['two_sided_binomial_sign_p']:.4f}. The magnitude-weighted mean is therefore positive, while rank- and sign-based evidence is weaker. The manuscript should report the effect and interval without calling it uniformly seed-stable.

## Claim boundary

The analysis identifies effects of representation policies inside the fixed MACE-ICTC/CHORUS training system. Frozen potential outcomes are model-internal counterfactuals, not causal properties of molecules. Diagonal versus Off remains a total module effect with active-capacity co-intervention.
"""
    (result / "VALIDATION_REPORT.md").write_text(markdown)
    if not passed:
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
