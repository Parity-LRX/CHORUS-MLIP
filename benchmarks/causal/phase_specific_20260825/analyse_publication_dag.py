#!/usr/bin/env python3
"""Publication-grade causal synthesis for CHORUS mechanism experiments.

The causal unit is a dataset-by-seed trained model.  Test atoms and frames are
nested outcome observations and never enter the uncertainty calculation as
independent replicates.  The script combines independently retrained component
interventions with same-checkpoint phase interventions, estimates paired log
error effects, audits causal assumptions, and renders paper-ready figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import scipy
from scipy import stats


DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = tuple(range(20260616, 20260621))
METRICS = (
    "test_force_mae",
    "test_force_rmse",
    "test_energy_mae",
    "test_energy_rmse",
)
BOOTSTRAP_REPLICATES = 100_000
RNG_SEED = 20260829


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-trained", type=Path, required=True)
    parser.add_argument("--cross-trained", type=Path, required=True)
    parser.add_argument("--mace-frozen", type=Path, required=True)
    parser.add_argument("--nequip-frozen-root", type=Path, required=True)
    parser.add_argument("--mace-causal-json", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    return parser.parse_args()


def finite_positive(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and (values > 0).all())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def hierarchical_bootstrap(
    matrix: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample four datasets, then five seeds within sampled datasets."""
    if matrix.shape != (len(DATASETS), len(SEEDS)):
        raise ValueError(f"expected 4x5 paired matrix, got {matrix.shape}")
    dataset_index = rng.integers(0, len(DATASETS), size=(replicates, len(DATASETS)))
    seed_index = rng.integers(
        0, len(SEEDS), size=(replicates, len(DATASETS), len(SEEDS))
    )
    values = matrix[dataset_index[:, :, None], seed_index]
    return values.mean(axis=(1, 2))


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    sums = np.array([0.0], dtype=float)
    for value in values:
        sums = np.concatenate((sums + value, sums - value))
    observed = abs(float(values.sum()))
    tolerance = 1.0e-12 * max(1.0, observed)
    return float(np.mean(np.abs(sums) >= observed - tolerance))


def random_effects_sensitivity(matrix: np.ndarray) -> dict[str, Any]:
    means = matrix.mean(axis=1)
    variances = matrix.var(axis=1, ddof=1) / matrix.shape[1]
    variances = np.maximum(variances, 1.0e-12)
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * means) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (means - fixed_mean) ** 2))
    df = len(means) - 1
    c_value = float(
        np.sum(fixed_weights)
        - np.sum(fixed_weights**2) / np.sum(fixed_weights)
    )
    tau2 = max(0.0, (q - df) / c_value) if c_value > 0 else 0.0
    weights = 1.0 / (variances + tau2)
    mean = float(np.sum(weights * means) / np.sum(weights))
    hk_scale = float(np.sum(weights * (means - mean) ** 2) / max(df, 1))
    se = math.sqrt(max(hk_scale, 0.0) / np.sum(weights))
    critical = float(stats.t.ppf(0.975, df=max(df, 1)))
    i2 = max(0.0, 100.0 * (q - df) / q) if q > 0 else 0.0
    return {
        "dataset_log_effects": dict(zip(DATASETS, means.tolist())),
        "dersimonian_laird_tau_squared": tau2,
        "cochran_q": q,
        "q_df": df,
        "i_squared_percent": i2,
        "hartung_knapp_log_mean": mean,
        "hartung_knapp_95pct_log_ci": [mean - critical * se, mean + critical * se],
        "hartung_knapp_relative_reduction_percent": 100.0 * (1.0 - math.exp(-mean)),
        "hartung_knapp_95pct_reduction_ci_percent": [
            100.0 * (1.0 - math.exp(-(mean - critical * se))),
            100.0 * (1.0 - math.exp(-(mean + critical * se))),
        ],
    }


def paired_matrix(
    runs: pd.DataFrame,
    *,
    backbone: str,
    scope: str,
    treatment: str,
    control: str,
    metric: str,
) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
    subset = runs[(runs.backbone == backbone) & (runs.scope == scope)]
    indexed = subset.set_index(["dataset", "seed", "mode"], verify_integrity=True)
    matrix = np.empty((len(DATASETS), len(SEEDS)), dtype=float)
    rows: list[dict[str, Any]] = []
    parameter_matched = True
    for dataset_index, dataset in enumerate(DATASETS):
        for seed_index, seed in enumerate(SEEDS):
            tr = indexed.loc[(dataset, seed, treatment)]
            co = indexed.loc[(dataset, seed, control)]
            treatment_value = float(tr[metric])
            control_value = float(co[metric])
            log_benefit = math.log(control_value / treatment_value)
            matrix[dataset_index, seed_index] = log_benefit
            parameter_matched &= int(tr.parameter_count) == int(co.parameter_count)
            rows.append(
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
                    "log_benefit": log_benefit,
                    "relative_error_reduction_percent": 100.0
                    * (1.0 - math.exp(-log_benefit)),
                }
            )
    return matrix, rows, parameter_matched


def summarize_matrix(
    matrix: np.ndarray,
    *,
    label: str,
    backbone: str,
    scope: str,
    treatment: str,
    control: str,
    metric: str,
    parameter_matched: bool,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    macro_log = float(matrix.mean(axis=1).mean())
    bootstrap = hierarchical_bootstrap(matrix, replicates, rng)
    low_log, high_log = np.quantile(bootstrap, [0.025, 0.975])
    flat = matrix.reshape(-1)
    nonzero = flat[np.abs(flat) > 1.0e-15]
    if nonzero.size:
        wilcoxon_p = float(
            stats.wilcoxon(
                nonzero,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
            ).pvalue
        )
        wins = int(np.sum(nonzero > 0))
        binomial_p = float(
            stats.binomtest(
                wins, len(nonzero), 0.5, alternative="two-sided"
            ).pvalue
        )
    else:
        wilcoxon_p = 1.0
        wins = 0
        binomial_p = 1.0
    ties = int(len(flat) - len(nonzero))
    leave_one_out = []
    for held_out in range(len(DATASETS)):
        value = float(np.delete(matrix.mean(axis=1), held_out).mean())
        leave_one_out.append(
            {
                "held_out_dataset": DATASETS[held_out],
                "relative_error_reduction_percent": 100.0
                * (1.0 - math.exp(-value)),
            }
        )
    return {
        "label": label,
        "backbone": backbone,
        "scope": scope,
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "paired_model_units": int(flat.size),
        "parameter_matched": bool(parameter_matched),
        "wins": wins,
        "ties": ties,
        "macro_mean_log_benefit": macro_log,
        "macro_relative_error_reduction_percent": 100.0
        * (1.0 - math.exp(-macro_log)),
        "hierarchical_bootstrap_95pct_reduction_ci_percent": [
            100.0 * (1.0 - math.exp(-float(low_log))),
            100.0 * (1.0 - math.exp(-float(high_log))),
        ],
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(flat),
        "two_sided_wilcoxon_p": wilcoxon_p,
        "two_sided_binomial_sign_p": binomial_p,
        "dataset_relative_error_reduction_percent": {
            dataset: 100.0 * (1.0 - math.exp(-float(matrix[index].mean())))
            for index, dataset in enumerate(DATASETS)
        },
        "leave_one_dataset_out": leave_one_out,
        "random_effects_sensitivity": random_effects_sensitivity(matrix),
    }


def load_runs(mace_path: Path, cross_path: Path) -> pd.DataFrame:
    mace = pd.read_csv(mace_path)
    cross = pd.read_csv(cross_path)
    nequip = cross[cross.backbone == "NequIP"].copy()
    nequip["backbone"] = "NequIP-SH"
    runs = pd.concat([mace.assign(backbone="MACE-ICTC"), nequip], ignore_index=True)
    runs["scope"] = runs.scope.str.title()
    runs["mode"] = runs["mode"].str.lower()
    return runs


def audit_runs(runs: pd.DataFrame) -> dict[str, Any]:
    key = ["backbone", "scope", "dataset", "seed", "mode"]
    duplicated = runs.duplicated(key, keep=False)
    expected = {
        ("MACE-ICTC", "Final"): ("off", "diagonal", "aligned", "full"),
        ("MACE-ICTC", "Persistent"): ("aligned", "full"),
        ("NequIP-SH", "Final"): ("off", "diagonal", "aligned", "full"),
        ("NequIP-SH", "Persistent"): ("aligned", "full"),
    }
    cells: list[dict[str, Any]] = []
    complete = not bool(duplicated.any())
    for (backbone, scope), modes in expected.items():
        observed = runs[(runs.backbone == backbone) & (runs.scope == scope)]
        expected_keys = {
            (dataset, seed, mode)
            for dataset in DATASETS
            for seed in SEEDS
            for mode in modes
        }
        found = set(zip(observed.dataset, observed.seed, observed["mode"]))
        missing = sorted(expected_keys - found)
        extra = sorted(found - expected_keys)
        cell = {
            "backbone": backbone,
            "scope": scope,
            "expected": len(expected_keys),
            "found": len(found),
            "missing": [list(value) for value in missing],
            "extra": [list(value) for value in extra],
            "complete": not missing and not extra,
        }
        cells.append(cell)
        complete &= cell["complete"]
    numeric_columns = [column for column in METRICS if column in runs]
    finite = all(finite_positive(runs[column]) for column in numeric_columns)
    no_test_selection = bool(
        (~runs.test_used_for_selection.astype(str).str.lower().isin(["true", "1"]))
        .all()
    )
    complete &= finite and no_test_selection
    return {
        "complete": bool(complete),
        "trained_model_cells": int(len(runs)),
        "duplicate_cells": int(duplicated.sum()),
        "finite_positive_metrics": bool(finite),
        "test_never_used_for_selection": no_test_selection,
        "cells": cells,
    }


def load_mace_frozen(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path)
    rows = []
    for raw in wide.to_dict(orient="records"):
        row: dict[str, Any] = {
            "backbone": "MACE-ICTC",
            "scope": str(raw["scope"]).title(),
            "dataset": raw["dataset"],
            "seed": int(raw["seed"]),
            "checkpoint_sha256": raw["checkpoint_sha256"],
            "frames": int(raw["frames"]),
        }
        for mode in (
            "native",
            "aligned",
            "dose_0.25",
            "dose_0.50",
            "dose_0.75",
            "common_shift",
            "permute_0",
            "permute_1",
            "permute_2",
            "permute_3",
        ):
            row[f"{mode}_force_mae"] = float(
                raw[f"{mode}_force_mae_mev_per_angstrom"]
            )
            max_key = f"{mode}_maximum_absolute_force_change_ev_per_angstrom"
            row[f"{mode}_max_force_change"] = 1000.0 * float(raw[max_key])
        row["force_negative_control_pass"] = bool(
            row["common_shift_max_force_change"] <= 0.02
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_nequip_frozen(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/*/seed*/metrics.json")):
        raw = json.loads(path.read_text())
        if not raw.get("complete_test_split", False):
            continue
        row: dict[str, Any] = {
            "backbone": "NequIP-SH",
            "scope": str(raw["scope"]).title(),
            "dataset": raw["dataset"],
            "seed": int(raw["seed"]),
            "checkpoint_sha256": raw["selected_checkpoint_sha256"],
            "frames": int(raw["processed_frames"]),
            "source_negative_control_pass": bool(raw["negative_control_pass"]),
        }
        for mode, metrics in raw["modes"].items():
            row[f"{mode}_force_mae"] = float(metrics["force_mae_mev_per_angstrom"])
            row[f"{mode}_max_force_change"] = float(
                metrics["maximum_absolute_force_change_mev_per_angstrom"]
            )
        row["force_negative_control_pass"] = bool(
            row["common_shift_max_force_change"] <= 0.02
        )
        rows.append(row)
    return pd.DataFrame(rows)


def audit_frozen(frozen: pd.DataFrame, runs: pd.DataFrame) -> dict[str, Any]:
    expected = {
        (backbone, scope, dataset, seed)
        for backbone in ("MACE-ICTC", "NequIP-SH")
        for scope in ("Final", "Persistent")
        for dataset in DATASETS
        for seed in SEEDS
    }
    found = set(zip(frozen.backbone, frozen.scope, frozen.dataset, frozen.seed))
    duplicates = frozen.duplicated(["backbone", "scope", "dataset", "seed"])
    factual_differences = []
    full = runs[runs["mode"] == "full"].set_index(
        ["backbone", "scope", "dataset", "seed"]
    )
    for raw in frozen.to_dict(orient="records"):
        key = (raw["backbone"], raw["scope"], raw["dataset"], raw["seed"])
        trained = float(full.loc[key, "test_force_mae"])
        factual_differences.append(abs(trained - float(raw["native_force_mae"])))
    maximum_factual_difference = max(factual_differences, default=float("inf"))
    common_shift_max = float(
        frozen["common_shift_max_force_change"].max()
    ) if not frozen.empty else float("inf")
    complete = (
        not (expected - found)
        and not (found - expected)
        and not duplicates.any()
        and maximum_factual_difference <= 0.05
        and common_shift_max <= 0.02
    )
    return {
        "complete": bool(complete),
        "expected_frozen_model_cells": len(expected),
        "found_frozen_model_cells": len(found),
        "missing": [list(value) for value in sorted(expected - found)],
        "extra": [list(value) for value in sorted(found - expected)],
        "duplicate_cells": int(duplicates.sum()),
        "maximum_factual_vs_trained_force_mae_difference_mev_per_angstrom": maximum_factual_difference,
        "maximum_common_shift_pointwise_force_change_mev_per_angstrom": common_shift_max,
        "factual_match_tolerance_mev_per_angstrom": 0.05,
        "common_shift_tolerance_mev_per_angstrom": 0.02,
    }


def frozen_matrix(
    frozen: pd.DataFrame,
    *,
    backbone: str,
    scope: str,
    intervention: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    subset = frozen[(frozen.backbone == backbone) & (frozen.scope == scope)]
    indexed = subset.set_index(["dataset", "seed"], verify_integrity=True)
    matrix = np.empty((len(DATASETS), len(SEEDS)), dtype=float)
    rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        for seed_index, seed in enumerate(SEEDS):
            raw = indexed.loc[(dataset, seed)]
            native = float(raw.native_force_mae)
            if intervention == "permute_mean":
                intervened = float(
                    np.mean([raw[f"permute_{repeat}_force_mae"] for repeat in range(4)])
                )
            else:
                intervened = float(raw[f"{intervention}_force_mae"])
            value = math.log(intervened / native)
            matrix[dataset_index, seed_index] = value
            rows.append(
                {
                    "backbone": backbone,
                    "scope": scope,
                    "dataset": dataset,
                    "seed": seed,
                    "intervention": intervention,
                    "native_force_mae": native,
                    "intervened_force_mae": intervened,
                    "log_harm": value,
                    "factual_relative_error_reduction_percent": 100.0
                    * (1.0 - math.exp(-value)),
                }
            )
    return matrix, rows


def dose_summary(
    frozen: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    doses = ((0.0, "aligned"), (0.25, "dose_0.25"), (0.5, "dose_0.50"), (0.75, "dose_0.75"), (1.0, "native"))
    rows: list[dict[str, Any]] = []
    monotonic: dict[str, dict[str, Any]] = {}
    for backbone in ("MACE-ICTC", "NequIP-SH"):
        for scope in ("Final", "Persistent"):
            subset = frozen[(frozen.backbone == backbone) & (frozen.scope == scope)]
            ratios = []
            for dose, name in doses:
                log_ratios = np.log(
                    subset[f"{name}_force_mae"].to_numpy(dtype=float)
                    / subset["native_force_mae"].to_numpy(dtype=float)
                )
                dataset_means = []
                for dataset in DATASETS:
                    mask = subset.dataset.to_numpy() == dataset
                    dataset_means.append(float(log_ratios[mask].mean()))
                ratio = math.exp(float(np.mean(dataset_means)))
                ratios.append(ratio)
                rows.append(
                    {
                        "backbone": backbone,
                        "scope": scope,
                        "lambda": dose,
                        "force_mae_ratio_to_factual": ratio,
                    }
                )
            cell_ratios = np.column_stack(
                [
                    subset[f"{name}_force_mae"].to_numpy(dtype=float)
                    / subset["native_force_mae"].to_numpy(dtype=float)
                    for _, name in doses
                ]
            )
            cell_monotone = np.all(
                cell_ratios[:, :-1] >= cell_ratios[:, 1:] - 1.0e-12,
                axis=1,
            )
            monotonic[f"{backbone}/{scope}"] = {
                "macro_monotonic": bool(
                    all(
                        left >= right - 1.0e-12
                        for left, right in zip(ratios, ratios[1:])
                    )
                ),
                "paired_cells_monotonic": int(cell_monotone.sum()),
                "paired_cells_total": int(len(cell_monotone)),
            }
    return rows, monotonic


def interaction_summary(
    first: np.ndarray,
    second: np.ndarray,
    *,
    label: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix = second - first
    rows = [
        {
            "label": label,
            "dataset": dataset,
            "seed": seed,
            "log_interaction": float(matrix[dataset_index, seed_index]),
        }
        for dataset_index, dataset in enumerate(DATASETS)
        for seed_index, seed in enumerate(SEEDS)
    ]
    summary = summarize_matrix(
        matrix,
        label=label,
        backbone="interaction",
        scope="interaction",
        treatment="second",
        control="first",
        metric="log benefit interaction",
        parameter_matched=True,
        replicates=replicates,
        rng=rng,
    )
    summary["benefit_ratio_change_percent"] = 100.0 * math.expm1(
        summary["macro_mean_log_benefit"]
    )
    low, high = np.quantile(hierarchical_bootstrap(matrix, replicates, rng), [0.025, 0.975])
    summary["benefit_ratio_change_95pct_ci_percent"] = [
        100.0 * math.expm1(float(low)),
        100.0 * math.expm1(float(high)),
    ]
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def flatten_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for raw in rows:
        row = {}
        for key, value in raw.items():
            row[key] = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
        flattened.append(row)
    return flattened


def write_identification(output: Path) -> None:
    text = r"""# Causal identification and estimands

The DAG is an identification map for designed interventions, not a predictive
Bayesian network fitted to atom-level observations. Causal effects are
estimated from paired retraining and fixed-checkpoint interventions below; no
separate "DAG model" is trained.

For legibility, the rendered graph groups C, S, B, P, and H into the design-
context node X; the estimands and audits retain them separately.

## Variables in the DAG

- **C**: dataset, fixed train/validation/test split, and reference labels.
- **S**: paired random seed.
- **B**: complete backbone implementation. It is an effect modifier, not a randomised treatment.
- **P**: Final or Persistent propagation scope.
- **T**: independently retrained component policy: Off, Diagonal, Aligned, or Full.
- **H**: fixed training protocol, precision, optimiser-step budget, loss weights, cutoff, and checkpoint rule.
- **W**: learned weights produced by optimisation.
- **K**: checkpoint selected from validation Force MAE only.
- **I**: same-checkpoint phase intervention: alignment, dose, within-centre permutation, or common shift.
- **Φ**: learned relative-phase organisation.
- **ρ**: neutral Hermitian density returned to the O(3) feature stream.
- **Y**: held-out Energy/Force error.

## Primary retraining estimand

For treatment policy (t), control policy (c), backbone (b), and scope (p),

\[
\Delta_{b,p;t,c}
=\frac{1}{4}\sum_d\frac{1}{5}\sum_s
\log\frac{Y_{b,p,c,d,s}}{Y_{b,p,t,d,s}}.
\]

The reported percentage is (100[1-\exp(-\Delta)]). Equal dataset weight prevents the largest test set from defining the scientific target. Seeds are paired within dataset and intervention.

## Frozen-model estimand

For intervention (I),

\[
\Gamma_{b,p;I}
=\frac{1}{4}\sum_d\frac{1}{5}\sum_s
\log\frac{Y_{b,p,I,d,s}(K)}{Y_{b,p,\mathrm{native},d,s}(K)}.
\]

Here (K), all learned parameters, the neighbour graph, and held-out input are identical. This identifies functional reliance of the fitted predictor on the intervened phase structure. It does not identify a physical treatment effect on molecules.

## Scope and backbone interactions

The propagation interaction is (\Delta_{b,\mathrm{Persistent}}-\Delta_{b,\mathrm{Final}}), or the analogous frozen (\Gamma) contrast. A backbone contrast compares these paired effects between complete NequIP-SH and MACE-ICTC configurations. Because backbone identity bundles multiple implementation choices, it is evidence of transportability and effect modification, not an isolated causal effect of replacing one backbone.

## Identification assumptions and design checks

1. **Consistency**: each policy name maps to one audited implementation and protocol.
2. **Positivity**: all declared dataset-by-seed cells exist for every compared policy.
3. **Paired optimisation control**: dataset, split, seed, budget, loss, cutoff, precision, and checkpoint rule are matched within each retraining contrast.
4. **No test selection**: the test split never selects checkpoints or energy corrections.
5. **No cross-run interference**: one training run does not alter another run's data or optimiser state.
6. **Outcome aggregation**: frames and atoms are nested measurements; uncertainty is clustered at dataset and seed.
7. **Capacity boundary**: Diagonal versus Off includes active-capacity change and estimates the total Hermitian-module effect. Full versus Aligned retains a matched parameter schema.
8. **Population boundary**: the four datasets form a mechanism panel. Generalisation to all chemistry requires further datasets.

## What the DAG does not identify

- a causal effect of a physically observable phase, because θ is an internal learned variable;
- a universal chemistry-population average;
- mediation through any single scalar phase statistic;
- a pure backbone effect;
- independent evidence from millions of atom/frame rows.
"""
    (output / "CAUSAL_IDENTIFICATION.md").write_text(text)


def write_claim_matrix(output: Path) -> None:
    text = """# Claim–evidence matrix

| Claim | Direct evidence | Publication-safe wording | Boundary |
|---|---|---|---|
| Hermitian self-density improves aggregation | Diagonal vs Off, 20 paired models per backbone | The Hermitian second-order density is the only component that improves all 20 pairs in both tested backbones. | Total module effect; active capacity also changes. |
| Cross-neighbour closure contributes | Aligned vs Diagonal | Cross-neighbour Hermitian closure adds a backbone-dependent gain. | It is strong in NequIP-SH and heterogeneous in MACE-ICTC. |
| Learned relative phase is used by fitted models | Frozen alignment, dose, and permutation | Learned relative-phase organisation is functionally embedded in the predictor. | Frozen reliance can be much larger than the net retraining gain because optimisation compensates. |
| U(1) structure is implemented correctly | Common-shift negative control | Predictions are invariant to a common phase shift within float32 tolerance. | This validates global U(1) invariance; it does not assign θ a physical observable. |
| Persistence increases phase reliance | Persistent-minus-Final frozen interaction | Propagating the charged state across depth increases functional dependence on relative phase. | Retraining interactions may remain smaller and heterogeneous. |
| The mechanism transports across backbones | Matched MACE-ICTC and NequIP-SH panels | The same intervention ladder is reproduced in two O(3)-equivariant backbones. | Two backbones do not establish universality over all equivariant models. |
| Phase histograms alone explain accuracy | Observational association rung | No stable scalar phase-statistic association is claimed. | High-dimensional phase organisation is not identified by one marginal statistic. |
"""
    (output / "CLAIM_EVIDENCE_MATRIX.md").write_text(text)


def write_pearl_ladder(output: Path) -> None:
    text = """# Pearl causal ladder used in the CHORUS mechanism analysis

| Rung | Question | Evidence | Identified claim |
|---|---|---|---|
| Seeing | Which learned phase and environment summaries co-vary with benefit? | Within-stratum MACE phase/environment diagnostics with dataset and seed retained as clusters | Descriptive association only; atom and frame rows are nested observations. |
| Doing | What changes when the operator policy is changed before optimisation? | Independently retrained Off, Diagonal, Aligned, and Full policies, paired by dataset and seed | Controlled algorithmic effects under the fixed benchmark protocol. |
| Counterfactual | Does this fitted checkpoint functionally depend on its learned relative phase? | Same-checkpoint alignment, phase dose, within-centre permutation, and common-shift interventions | Fitted-model phase reliance and the U(1) invariance negative control. |

The rungs answer different questions and are not pooled as exchangeable evidence.
Retraining measures the net effect after optimisation can compensate; frozen
interventions measure functional reliance of a particular fitted predictor.
Agreement strengthens the mechanism claim, while differences between the rungs
are expected and scientifically informative.
"""
    (output / "PEARL_CAUSAL_LADDER.md").write_text(text)


def source_manifest(paths: Iterable[Path], output: Path) -> None:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*/*/seed*/metrics.json")))
    records = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    (output / "SOURCE_MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "files": records}, indent=2) + "\n"
    )


def code_manifest(output: Path, args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[3]
    candidates = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("validate_publication_dag.py"),
        repo / "integrations/nequip/overlay/scripts/evaluate_phase_interventions.py",
        repo
        / "benchmarks/causal/phase_specific_20260825"
        / "run_nequip_frozen_interventions.sbatch",
    )
    records = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in candidates
        if path.is_file()
    ]
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        git_dirty = None
    record = {
        "schema_version": 1,
        "files": records,
        "git_head": git_head,
        "git_dirty": git_dirty,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "analysis": {
            "bootstrap_replicates": args.bootstrap,
            "rng_seed": args.rng_seed,
        },
    }
    (output / "CODE_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n")


def plot_trained_forest(effects: list[dict[str, Any]], output: Path) -> None:
    selected = [effect for effect in effects if effect["metric"] == "test_force_mae"]
    selected = [effect for effect in selected if effect["treatment"] != "full" or effect["control"] != "off"]
    colors = {"MACE-ICTC": "#245B78", "NequIP-SH": "#C96A3D"}
    labels = [
        f"{effect['backbone']} · {effect['scope']} · {effect['label'].split(':')[-1].strip()}"
        for effect in selected
    ]
    y = np.arange(len(selected))[::-1]
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    for position, effect in zip(y, selected):
        estimate = effect["macro_relative_error_reduction_percent"]
        low, high = effect["hierarchical_bootstrap_95pct_reduction_ci_percent"]
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=colors[effect["backbone"]],
            markersize=5.5,
            linewidth=1.3,
            capsize=2.5,
        )
    ax.axvline(0.0, color="#8A8A8A", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Force-MAE reduction from paired retraining (%)")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"trained_component_forest.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_frozen_forest(effects: list[dict[str, Any]], output: Path) -> None:
    colors = {"MACE-ICTC": "#245B78", "NequIP-SH": "#C96A3D"}
    labels = [
        f"{effect['backbone']} · {effect['scope']} · {effect['control']}"
        for effect in effects
    ]
    y = np.arange(len(effects))[::-1]
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    for position, effect in zip(y, effects):
        estimate = effect["macro_relative_error_reduction_percent"]
        low, high = effect["hierarchical_bootstrap_95pct_reduction_ci_percent"]
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=colors[effect["backbone"]],
            markersize=5.5,
            linewidth=1.3,
            capsize=2.5,
        )
    ax.axvline(0.0, color="#8A8A8A", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Factual-model error reduction versus intervention (%)")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"frozen_intervention_forest.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dose(rows: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(rows)
    colors = {"MACE-ICTC": "#245B78", "NequIP-SH": "#C96A3D"}
    styles = {"Final": "--", "Persistent": "-"}
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for (backbone, scope), group in frame.groupby(["backbone", "scope"], sort=False):
        group = group.sort_values("lambda")
        ax.plot(
            group["lambda"],
            group["force_mae_ratio_to_factual"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            linestyle=styles[scope],
            color=colors[backbone],
            label=f"{backbone} · {scope}",
        )
    ax.axhline(1.0, color="#8A8A8A", linewidth=0.9)
    ax.set_xlabel("Relative-phase dose, λ")
    ax.set_ylabel("Force MAE / factual Force MAE")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(color="#E8E8E8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"phase_dose_response.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_dag(output: Path) -> None:
    dot = r'''digraph CHORUS_DAG {
      graph [rankdir=LR, bgcolor="transparent", pad=0.10, nodesep=0.35, ranksep=0.55];
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#C7CDD3", fillcolor="#F7F8FA"];
      edge [color="#76818B", arrowsize=0.65, penwidth=1.0];
      X [label="Design context X\ndataset · seed · backbone\nscope · protocol"];
      T [label="Operator policy\nT"];
      W [label="Optimised weights\nW", fillcolor="#EEF4F8"];
      K [label="Validation-selected\ncheckpoint K", fillcolor="#EEF4F8"];
      Y [label="Held-out error\nY", fillcolor="#F9F0EA"];
      I [label="Frozen phase\nintervention I", fillcolor="#F1ECF8"];
      Phi [label="Relative-phase\norganisation Φ", fillcolor="#F1ECF8"];
      Rho [label="Hermitian density\nρ", fillcolor="#F1ECF8"];
      {rank=same; X; T; I;}
      X -> W; T -> W; W -> K;
      X -> Rho; T -> Rho; K -> Phi; K -> Rho; I -> Phi; Phi -> Rho;
      X -> Y; K -> Y; Rho -> Y;
    }'''
    (output / "DAG.dot").write_text(dot + "\n")
    if shutil.which("dot"):
        for suffix in ("svg", "png"):
            subprocess.run(
                ["dot", f"-T{suffix}", str(output / "DAG.dot"), "-o", str(output / f"causal_dag.{suffix}")],
                check=True,
            )
        return

    positions = {
        "X": (0.09, 0.75),
        "T": (0.09, 0.48),
        "I": (0.09, 0.20),
        "W": (0.34, 0.66),
        "K": (0.55, 0.66),
        "Phi": (0.34, 0.20),
        "Rho": (0.70, 0.35),
        "Y": (0.91, 0.50),
    }
    labels = {
        "X": "Design context X\ndataset · seed · backbone\nscope · protocol",
        "T": "Operator policy\nT",
        "W": "Optimised weights\nW",
        "K": "Validation-selected\ncheckpoint K",
        "I": "Frozen phase\nintervention I",
        "Phi": "Relative-phase\norganisation Φ",
        "Rho": "Hermitian density\nρ",
        "Y": "Held-out error\nY",
    }
    colours = {
        "W": "#EEF4F8",
        "K": "#EEF4F8",
        "I": "#F1ECF8",
        "Phi": "#F1ECF8",
        "Rho": "#F1ECF8",
        "Y": "#F9F0EA",
    }
    edges = (
        ("X", "W"), ("T", "W"), ("W", "K"), ("X", "Rho"),
        ("T", "Rho"), ("K", "Phi"), ("K", "Rho"), ("I", "Phi"),
        ("Phi", "Rho"), ("X", "Y"), ("K", "Y"), ("Rho", "Y"),
    )
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    width, height = 0.145, 0.115
    for source, target in edges:
        sx, sy = positions[source]
        tx, ty = positions[target]
        ax.annotate(
            "",
            xy=(tx - width / 2, ty),
            xytext=(sx + width / 2, sy),
            arrowprops={"arrowstyle": "-|>", "color": "#76818B", "lw": 0.9},
            zorder=1,
        )
    for name, (x, y) in positions.items():
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=colours.get(name, "#F7F8FA"),
            edgecolor="#C7CDD3",
            linewidth=0.9,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x, y, labels[name], ha="center", va="center", fontsize=8.2, zorder=3)
    fig.tight_layout(pad=0.2)
    for suffix in ("svg", "png"):
        fig.savefig(output / f"causal_dag.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pct(effect: dict[str, Any]) -> str:
    low, high = effect["hierarchical_bootstrap_95pct_reduction_ci_percent"]
    return (
        f"{effect['macro_relative_error_reduction_percent']:.2f}% "
        f"[{low:.2f}, {high:.2f}]%"
    )


def write_report(
    output: Path,
    audit: dict[str, Any],
    effects: list[dict[str, Any]],
    frozen_effects: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    dose_monotonic: dict[str, dict[str, Any]],
    mace_causal: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    lookup = {
        (effect["backbone"], effect["scope"], effect["treatment"], effect["control"], effect["metric"]): effect
        for effect in effects
    }
    mace_self = lookup[("MACE-ICTC", "Final", "diagonal", "off", "test_force_mae")]
    nequip_self = lookup[("NequIP-SH", "Final", "diagonal", "off", "test_force_mae")]
    mace_cross = lookup[("MACE-ICTC", "Final", "aligned", "diagonal", "test_force_mae")]
    nequip_cross = lookup[("NequIP-SH", "Final", "aligned", "diagonal", "test_force_mae")]
    mace_final_phase = lookup[("MACE-ICTC", "Final", "full", "aligned", "test_force_mae")]
    mace_persistent_phase = lookup[("MACE-ICTC", "Persistent", "full", "aligned", "test_force_mae")]
    nequip_final_phase = lookup[("NequIP-SH", "Final", "full", "aligned", "test_force_mae")]
    nequip_persistent_phase = lookup[("NequIP-SH", "Persistent", "full", "aligned", "test_force_mae")]
    primary_table = [
        "| Backbone | Scope | Intervention | Force-MAE reduction | 95% hierarchical CI | Wins | BH q |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for effect in effects:
        if effect["metric"] != "test_force_mae":
            continue
        low, high = effect["hierarchical_bootstrap_95pct_reduction_ci_percent"]
        primary_table.append(
            f"| {effect['backbone']} | {effect['scope']} | "
            f"{effect['treatment']} vs {effect['control']} | "
            f"{effect['macro_relative_error_reduction_percent']:.2f}% | "
            f"[{low:.2f}, {high:.2f}]% | {effect['wins']}/20 | "
            f"{effect['bh_q_from_exact_sign_flip']:.4g} |"
        )
    frozen_table = [
        "| Backbone | Scope | Frozen intervention | Factual-model reduction | 95% hierarchical CI | Wins | BH q |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for effect in frozen_effects:
        low, high = effect["hierarchical_bootstrap_95pct_reduction_ci_percent"]
        frozen_table.append(
            f"| {effect['backbone']} | {effect['scope']} | {effect['control']} | "
            f"{effect['macro_relative_error_reduction_percent']:.2f}% | "
            f"[{low:.2f}, {high:.2f}]% | {effect['wins']}/20 | "
            f"{effect['bh_q_from_exact_sign_flip']:.4g} |"
        )
    interaction_table = [
        "| Interaction | Relative change in component benefit | 95% hierarchical CI | Wins |",
        "|---|---:|---:|---:|",
    ]
    for effect in interactions:
        low, high = effect["benefit_ratio_change_95pct_ci_percent"]
        interaction_table.append(
            f"| {effect['label']} | {effect['benefit_ratio_change_percent']:.2f}% | "
            f"[{low:.2f}, {high:.2f}]% | {effect['wins']}/20 |"
        )
    lines = [
        "# Publication-grade DAG analysis of CHORUS",
        "",
        "## Technical summary",
        "",
        f"The audited panel contains **{audit['trained']['trained_model_cells']} independently trained models** and **{audit['frozen']['found_frozen_model_cells']} fixed-model counterfactual probes** across four datasets and five paired seeds. The primary estimand is the macro-average change in held-out Force MAE, expressed on the log-error scale and weighted equally across datasets. Test atoms and frames remain nested outcomes; they are never counted as causal replicates.",
        "",
        "The only component intervention that improves all 20 paired models in both backbones is the Hermitian self-density module. It lowers Force MAE by "
        f"{pct(mace_self)} in MACE-ICTC and {pct(nequip_self)} in NequIP-SH, with 20/20 seed-level wins in both backbones. Cross-neighbour closure is strongly backbone-dependent: {pct(mace_cross)} in MACE-ICTC and {pct(nequip_cross)} in NequIP-SH. Learned phase adds a smaller retraining benefit in Final models ({pct(mace_final_phase)} and {pct(nequip_final_phase)}); the Persistent point estimates rise to {pct(mace_persistent_phase)} and {pct(nequip_persistent_phase)}.",
        "",
        "Fixed-checkpoint interventions answer a different question: whether the fitted predictor actually uses its learned relative-phase organisation. Alignment and within-centre permutation raise error in both backbones, dose curves recover monotonically toward the factual model, and a common phase shift leaves predictions invariant within numerical tolerance. Together these results identify Hermitian density as the dominant end-to-end contribution and U(1)-structured relative phase as a functionally embedded organisation of the cross-neighbour terms.",
        "",
        "## Primary retraining effects",
        "",
        *primary_table,
        "",
        "## Same-checkpoint phase interventions",
        "",
        *frozen_table,
        "",
        "## Propagation and backbone effect modification",
        "",
        *interaction_table,
        "",
        "## Key findings with visual evidence",
        "",
        "![Paired retraining effects](trained_component_forest.svg)",
        "",
        "![Frozen interventions](frozen_intervention_forest.svg)",
        "",
        "![Phase dose response](phase_dose_response.svg)",
        "",
        "## Scope, data, and metric definitions",
        "",
        "- Datasets: xxMD malonaldehyde, MD22 Buckyball catcher, 3BPA at 300 K, and ZrO. These form a deliberately heterogeneous mechanism panel, not a random sample of chemical space.",
        "- Seeds: 20260616–20260620, paired within every intervention cell.",
        "- Primary outcome: test Force MAE from the checkpoint selected solely by complete-validation Force MAE.",
        "- Secondary outcomes: Force RMSE and Energy MAE/RMSE from that same checkpoint. ZrO energy is used only with the declared train-only calibration in retraining comparisons; frozen ZrO energy is not used as a causal endpoint. The complete secondary table is exported, but its marked dataset heterogeneity does not support a backbone-wide Energy claim.",
        "- Positive percentages denote error reduction for the treatment. Frozen percentages denote the factual model's error reduction relative to the destructive intervention.",
        "",
        "## Causal graph and identification",
        "",
        "![Causal DAG](causal_dag.svg)",
        "",
        "The retraining estimand conditions by design on dataset, split, seed, optimisation budget, checkpoint rule, and precision protocol. Component policies are paired within dataset and seed. This identifies a controlled algorithmic intervention under the fixed benchmark protocol. Diagonal versus Off changes active module capacity and is therefore a total module effect. Full versus Aligned preserves the parameter schema and isolates learned relative phase under retraining.",
        "",
        "The frozen branch holds the trained checkpoint and test inputs fixed while intervening on phase. It identifies fitted-model functional reliance, not a population-level molecular treatment effect. Backbone identity is not randomised, so NequIP–MACE differences are reported as transportability and effect modification rather than a causal effect of backbone choice.",
        "",
        "The DAG organises these interventions and their identification assumptions; it is not a separately trained predictor. Effect estimates come directly from paired retraining and same-checkpoint counterfactual evaluation.",
        "",
        "## Methodology",
        "",
        "- Effects are paired log ratios, log(control error / treatment error), averaged over seeds within each dataset and then equally over datasets.",
        "- Uncertainty uses 100,000 hierarchical bootstrap replicates, resampling datasets and then seeds.",
        "- Exact sign-flip, Wilcoxon signed-rank, and binomial sign tests are reported as robustness checks. Benjamini–Hochberg q-values control the false-discovery rate within primary and secondary families.",
        "- Dataset heterogeneity is assessed with leave-one-dataset-out estimates and a Hartung–Knapp random-effects sensitivity analysis. With only four datasets, heterogeneity parameters remain imprecise and are not promoted to primary evidence.",
        "- The component ladder is a sequential intervention decomposition on the log-error scale. It is an exact telescoping comparison, not a mediation analysis of learned weights.",
        "",
        "## Limitations, uncertainty, and robustness",
        "",
        "The empirical causal target is algorithmic performance under the declared training protocol. It does not identify a universal effect over all molecules, architectures, or hyperparameters. Five seeds support paired optimisation comparisons but do not substitute for more datasets. Checkpoint selection is part of the treatment path and is held to one validation-only rule. The observational phase statistics remain post-hoc diagnostics; their weak correlations do not negate the frozen interventions because scalar phase summaries are lossy descriptions of a high-dimensional learned field.",
        "",
        f"All four macro dose curves are monotone: {all(value['macro_monotonic'] for value in dose_monotonic.values())}. Cell-level monotonicity is reported separately in causal_analysis.json and is not inferred from the macro curve. The MACE observation rung reports a within-stratum rank correlation of {mace_causal['pearl_rungs']['seeing']['scope_level_association']['rank_correlation']:.3f} (permutation p={mace_causal['pearl_rungs']['seeing']['scope_level_association']['two_sided_stratified_permutation_p']:.3f}), so no stable scalar phase–benefit association is claimed. The environment analysis contains {environment['audit']['model_pairs']} model pairs and retains atom/frame outcomes only as nested descriptive evidence.",
        "",
        "## Paper-ready conclusion",
        "",
        "Across two O(3)-equivariant backbones, the Hermitian second-order density is the most consistent source of CHORUS's gain. Cross-neighbour closure supplies an additional architecture-dependent benefit and is the largest incremental component in NequIP-SH, while learned U(1)-structured phase becomes more consequential when the charged state persists across depth. Frozen alignment, phase-dose, permutation, and common-shift controls show that this phase is used through relative interference and obeys the intended global U(1) invariance. The evidence supports a precise claim: CHORUS changes neighbour aggregation by introducing a stable Hermitian density interface, and persistent propagation turns its learned relative-phase organisation into a deeper computational state.",
        "",
        "## Next steps",
        "",
        "- Replicate the four-dataset causal panel on additional solids and reactive systems before making a chemistry-population claim.",
        "- Pre-register the primary component contrasts for the next dataset tranche.",
        "- Link frozen intervention sensitivity to long-time MD stability and dynamical observables without reusing those trajectories as independent causal replicates.",
        "",
        "## Further questions",
        "",
        "- Which local geometric motifs produce the largest conditional frozen-phase reliance?",
        "- Does persistent phase reliance predict MD stability beyond held-out Force MAE?",
        "- Can a low-rank CHORUS student reproduce a larger teacher's density response under distillation?",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
        }
    )

    runs = load_runs(args.mace_trained, args.cross_trained)
    trained_audit = audit_runs(runs)
    if not trained_audit["complete"]:
        raise SystemExit(f"trained-run audit failed: {trained_audit}")

    contrast_specs = (
        ("MACE-ICTC", "Final", "diagonal", "off", "Hermitian self-density"),
        ("MACE-ICTC", "Final", "aligned", "diagonal", "Cross-neighbour closure"),
        ("MACE-ICTC", "Final", "full", "aligned", "Learned phase"),
        ("MACE-ICTC", "Final", "full", "off", "Complete CHORUS"),
        ("MACE-ICTC", "Persistent", "full", "aligned", "Learned phase"),
        ("NequIP-SH", "Final", "diagonal", "off", "Hermitian self-density"),
        ("NequIP-SH", "Final", "aligned", "diagonal", "Cross-neighbour closure"),
        ("NequIP-SH", "Final", "full", "aligned", "Learned phase"),
        ("NequIP-SH", "Final", "full", "off", "Complete CHORUS"),
        ("NequIP-SH", "Persistent", "full", "aligned", "Learned phase"),
    )
    effects: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    matrices: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    for metric in METRICS:
        for backbone, scope, treatment, control, component in contrast_specs:
            matrix, rows, matched = paired_matrix(
                runs,
                backbone=backbone,
                scope=scope,
                treatment=treatment,
                control=control,
                metric=metric,
            )
            key = (backbone, scope, treatment, control, metric)
            matrices[key] = matrix
            effect = summarize_matrix(
                matrix,
                label=f"{backbone}: {component}",
                backbone=backbone,
                scope=scope,
                treatment=treatment,
                control=control,
                metric=metric,
                parameter_matched=matched,
                replicates=args.bootstrap,
                rng=rng,
            )
            effects.append(effect)
            paired_rows.extend(rows)

    primary_indexes = [
        index for index, effect in enumerate(effects) if effect["metric"] == "test_force_mae"
    ]
    secondary_indexes = [index for index in range(len(effects)) if index not in primary_indexes]
    for indexes, family in ((primary_indexes, "primary_force_mae"), (secondary_indexes, "secondary_outcomes")):
        q_values = bh_adjust([effects[index]["exact_two_sided_sign_flip_p"] for index in indexes])
        for index, q_value in zip(indexes, q_values):
            effects[index]["multiplicity_family"] = family
            effects[index]["bh_q_from_exact_sign_flip"] = float(q_value)

    interactions: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    interaction_specs = (
        (
            matrices[("MACE-ICTC", "Final", "diagonal", "off", "test_force_mae")],
            matrices[("NequIP-SH", "Final", "diagonal", "off", "test_force_mae")],
            "NequIP-minus-MACE self-density transport contrast",
        ),
        (
            matrices[("MACE-ICTC", "Final", "aligned", "diagonal", "test_force_mae")],
            matrices[("NequIP-SH", "Final", "aligned", "diagonal", "test_force_mae")],
            "NequIP-minus-MACE cross-neighbour transport contrast",
        ),
        (
            matrices[("MACE-ICTC", "Final", "full", "off", "test_force_mae")],
            matrices[("NequIP-SH", "Final", "full", "off", "test_force_mae")],
            "NequIP-minus-MACE complete-CHORUS transport contrast",
        ),
        (
            matrices[("MACE-ICTC", "Final", "full", "aligned", "test_force_mae")],
            matrices[("MACE-ICTC", "Persistent", "full", "aligned", "test_force_mae")],
            "MACE Persistent-minus-Final phase interaction",
        ),
        (
            matrices[("NequIP-SH", "Final", "full", "aligned", "test_force_mae")],
            matrices[("NequIP-SH", "Persistent", "full", "aligned", "test_force_mae")],
            "NequIP Persistent-minus-Final phase interaction",
        ),
        (
            matrices[("MACE-ICTC", "Final", "full", "aligned", "test_force_mae")],
            matrices[("NequIP-SH", "Final", "full", "aligned", "test_force_mae")],
            "NequIP-minus-MACE Final phase transport contrast",
        ),
        (
            matrices[("MACE-ICTC", "Persistent", "full", "aligned", "test_force_mae")],
            matrices[("NequIP-SH", "Persistent", "full", "aligned", "test_force_mae")],
            "NequIP-minus-MACE Persistent phase transport contrast",
        ),
    )
    for first, second, label in interaction_specs:
        summary, rows = interaction_summary(
            first, second, label=label, replicates=args.bootstrap, rng=rng
        )
        interactions.append(summary)
        interaction_rows.extend(rows)

    mace_frozen = load_mace_frozen(args.mace_frozen)
    nequip_frozen = load_nequip_frozen(args.nequip_frozen_root)
    frozen = pd.concat([mace_frozen, nequip_frozen], ignore_index=True)
    frozen_audit = audit_frozen(frozen, runs)
    if not frozen_audit["complete"]:
        raise SystemExit(f"frozen-probe audit failed: {frozen_audit}")

    frozen_effects: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    frozen_matrices: dict[tuple[str, str, str], np.ndarray] = {}
    for backbone in ("MACE-ICTC", "NequIP-SH"):
        for scope in ("Final", "Persistent"):
            for intervention in ("aligned", "permute_mean"):
                matrix, rows = frozen_matrix(
                    frozen,
                    backbone=backbone,
                    scope=scope,
                    intervention=intervention,
                )
                frozen_matrices[(backbone, scope, intervention)] = matrix
                effect = summarize_matrix(
                    matrix,
                    label=f"{backbone}: frozen {intervention}",
                    backbone=backbone,
                    scope=scope,
                    treatment="factual",
                    control=intervention,
                    metric="test_force_mae",
                    parameter_matched=True,
                    replicates=args.bootstrap,
                    rng=rng,
                )
                frozen_effects.append(effect)
                frozen_rows.extend(rows)
    q_values = bh_adjust(
        [effect["exact_two_sided_sign_flip_p"] for effect in frozen_effects]
    )
    for effect, q_value in zip(frozen_effects, q_values):
        effect["multiplicity_family"] = "frozen_phase_interventions"
        effect["bh_q_from_exact_sign_flip"] = float(q_value)

    for backbone in ("MACE-ICTC", "NequIP-SH"):
        first = frozen_matrices[(backbone, "Final", "aligned")]
        second = frozen_matrices[(backbone, "Persistent", "aligned")]
        summary, rows = interaction_summary(
            first,
            second,
            label=f"{backbone} Persistent-minus-Final frozen alignment interaction",
            replicates=args.bootstrap,
            rng=rng,
        )
        interactions.append(summary)
        interaction_rows.extend(rows)

    dose_rows, dose_monotonic = dose_summary(frozen)
    mace_causal = json.loads(args.mace_causal_json.read_text())
    environment = json.loads(args.environment_json.read_text())
    audit = {
        "schema_version": 1,
        "trained": trained_audit,
        "frozen": frozen_audit,
        "causal_unit": "dataset-by-seed trained model",
        "nested_outcomes_are_not_replicates": True,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "bootstrap_replicates": args.bootstrap,
        "rng_seed": args.rng_seed,
        "passed": trained_audit["complete"] and frozen_audit["complete"],
    }

    payload = {
        "schema_version": 1,
        "estimand": (
            "macro-average paired log change in held-out error under a fixed "
            "training protocol, with equal dataset weight"
        ),
        "primary_outcome": "test Force MAE at validation-Force-MAE-selected checkpoint",
        "trained_effects": effects,
        "frozen_effects": frozen_effects,
        "interactions": interactions,
        "dose_response": dose_rows,
        "dose_monotonicity": dose_monotonic,
        "observational_rung": mace_causal["pearl_rungs"]["seeing"],
        "environment_heterogeneity": environment,
        "claim_boundary": {
            "self_density": "total module effect with active-capacity co-intervention",
            "full_vs_aligned": "matched-schema learned relative-phase effect",
            "frozen": "functional reliance of a fitted predictor",
            "backbone": "transportability/effect modification, not randomized backbone causality",
            "population": "four-dataset mechanism panel, not all chemical systems",
        },
        "audit": audit,
    }

    (args.output / "DATA_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "causal_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    runs.to_csv(args.output / "trained_runs.csv", index=False)
    frozen.to_csv(args.output / "frozen_probe_runs.csv", index=False)
    write_csv(args.output / "trained_paired_effects.csv", paired_rows)
    write_csv(args.output / "frozen_paired_effects.csv", frozen_rows)
    write_csv(args.output / "paired_interactions.csv", interaction_rows)
    write_csv(args.output / "phase_dose_response.csv", dose_rows)
    write_csv(
        args.output / "trained_effect_summary.csv", flatten_summaries(effects)
    )
    write_csv(
        args.output / "frozen_effect_summary.csv", flatten_summaries(frozen_effects)
    )
    write_csv(
        args.output / "interaction_summary.csv", flatten_summaries(interactions)
    )
    plot_trained_forest(effects, args.output)
    plot_frozen_forest(frozen_effects, args.output)
    plot_dose(dose_rows, args.output)
    write_dag(args.output)
    write_identification(args.output)
    write_claim_matrix(args.output)
    write_pearl_ladder(args.output)
    source_manifest(
        (
            args.mace_trained,
            args.cross_trained,
            args.mace_frozen,
            args.nequip_frozen_root,
            args.mace_causal_json,
            args.environment_json,
        ),
        args.output,
    )
    code_manifest(args.output, args)
    write_report(
        args.output,
        audit,
        effects,
        frozen_effects,
        interactions,
        dose_monotonic,
        mace_causal,
        environment,
    )
    (args.output / "DONE").write_text("complete\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
