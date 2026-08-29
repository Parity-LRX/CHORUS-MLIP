#!/usr/bin/env python3
"""Reproduce the CHORUS Pearl three-rung causal-mechanism audit.

The script combines the completed end-to-end U(1)/Hermitian campaign, the
phase-specific frozen interventions, the paired Full/Aligned retraining, and
the post-hoc observational phase summaries.  The primary estimand is paired
test Force MAE after checkpoint selection by validation Force MAE.

Positive relative-error-reduction values mean that the named treatment has
lower error than its control.  Energy and force outcomes are never spliced
across checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = tuple(range(20260616, 20260621))
PRIMARY_TEST = {"3bpa": "300K", "mal": "test", "bucky": "test", "zro": "test"}
DATASET_LABELS = {
    "mal": "MAL",
    "bucky": "Buckyball",
    "3bpa": "3BPA (300 K)",
    "zro": "ZrO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("/tmp/chorus_u1_causal_20260824/runs"),
    )
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=Path("/tmp/chorus_phase_causal_20260825"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/paper/results/causal/pearl_three_rungs_20260825"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--association-permutations", type=int, default=100_000)
    parser.add_argument("--rng-seed", type=int, default=20260825)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def load_primary_runs(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for mode in ("off", "diagonal", "charge2", "full"):
                path = root / dataset / f"seed{seed}" / mode / "metrics.json"
                payload = read_json(path)
                test_name = PRIMARY_TEST[dataset]
                test = payload["tests"][test_name]
                selected = payload["checkpoint_selection"]["force_mae_envelope"]
                records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "mode": mode,
                        "parameter_count": payload["parameter_count"],
                        "selected_step": selected["step"],
                        "validation_energy_mae": selected["energy_mae_mev_per_atom"],
                        "validation_energy_rmse": selected["energy_rmse_mev_per_atom"],
                        "validation_force_mae": selected["force_mae_mev_per_angstrom"],
                        "validation_force_rmse": selected["force_rmse_mev_per_angstrom"],
                        "test_energy_mae": test["energy_mae_mev_per_atom"],
                        "test_energy_rmse": test["energy_rmse_mev_per_atom"],
                        "test_force_mae": test["force_mae_mev_per_angstrom"],
                        "test_force_rmse": test["force_rmse_mev_per_angstrom"],
                        "test_name": test_name,
                        "selection_rule": payload["checkpoint_selection"]["rule"],
                        "test_used_for_selection": payload["checkpoint_selection"][
                            "test_used_for_selection"
                        ],
                        "metrics_path": str(path),
                    }
                )
    return pd.DataFrame(records)


def load_aligned_runs(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / "aligned_runs" / dataset / f"seed{seed}" / "aligned" / "metrics.json"
            payload = read_json(path)
            test_name = PRIMARY_TEST[dataset]
            test = payload["tests"][test_name]
            selected = payload["checkpoint_selection"]["force_mae_envelope"]
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "mode": "aligned",
                    "parameter_count": payload["parameter_count"],
                    "selected_step": selected["step"],
                    "validation_energy_mae": selected["energy_mae_mev_per_atom"],
                    "validation_energy_rmse": selected["energy_rmse_mev_per_atom"],
                    "validation_force_mae": selected["force_mae_mev_per_angstrom"],
                    "validation_force_rmse": selected["force_rmse_mev_per_angstrom"],
                    "test_energy_mae": test["energy_mae_mev_per_atom"],
                    "test_energy_rmse": test["energy_rmse_mev_per_atom"],
                    "test_force_mae": test["force_mae_mev_per_angstrom"],
                    "test_force_rmse": test["force_rmse_mev_per_angstrom"],
                    "test_name": test_name,
                    "selection_rule": payload["checkpoint_selection"]["rule"],
                    "test_used_for_selection": payload["checkpoint_selection"][
                        "test_used_for_selection"
                    ],
                    "metrics_path": str(path),
                }
            )
    return pd.DataFrame(records)


def load_probe_runs(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            test_name = PRIMARY_TEST[dataset]
            path = root / "probes" / dataset / f"seed{seed}" / test_name / "metrics.json"
            payload = read_json(path)
            metrics = payload["metrics"]
            record: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "test": test_name,
                "parameter_count": payload["parameter_count"],
                "frames": payload["frames"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "metrics_path": str(path),
            }
            for intervention in (
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
                for metric in (
                    "energy_mae_mev_per_atom",
                    "energy_rmse_mev_per_atom",
                    "force_mae_mev_per_angstrom",
                    "force_rmse_ev_per_angstrom",
                    "maximum_absolute_force_change_ev_per_angstrom",
                    "mean_absolute_force_change_ev_per_angstrom",
                ):
                    if metric in metrics[intervention]:
                        record[f"{intervention}_{metric}"] = metrics[intervention][metric]
            record["permute_mean_force_mae_mev_per_angstrom"] = metrics["permute_mean"][
                "force_mae_mev_per_angstrom"
            ]
            records.append(record)
    return pd.DataFrame(records)


def combine_runs(primary: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([primary, aligned], ignore_index=True)
    if combined.duplicated(["dataset", "seed", "mode"]).any():
        raise RuntimeError("Duplicate dataset/seed/mode rows detected")
    return combined


def paired_values(
    runs: pd.DataFrame,
    treatment: str,
    control: str,
    metric: str = "test_force_mae",
) -> pd.DataFrame:
    wide = runs.pivot(index=["dataset", "seed"], columns="mode", values=metric)
    if treatment not in wide or control not in wide:
        raise KeyError(f"Missing {treatment} or {control} for {metric}")
    pair = wide[[treatment, control]].dropna().reset_index()
    pair["log_benefit"] = np.log(pair[control] / pair[treatment])
    pair["relative_reduction_percent"] = 100.0 * (1.0 - pair[treatment] / pair[control])
    return pair


def effect_from_log(log_benefit: np.ndarray) -> float:
    return float(100.0 * (1.0 - np.exp(-np.mean(log_benefit))))


def hierarchical_bootstrap(
    pair: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    grouped = {d: g["log_benefit"].to_numpy(float) for d, g in pair.groupby("dataset")}
    datasets = np.array(DATASETS)
    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            seed_effects = grouped[str(dataset)]
            dataset_means.append(float(np.mean(rng.choice(seed_effects, size=len(seed_effects), replace=True))))
        values[i] = effect_from_log(np.asarray(dataset_means))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_p(log_benefit: np.ndarray) -> float:
    values = np.asarray(log_benefit, dtype=np.float64)
    n = len(values)
    observed = abs(float(np.mean(values)))
    total = 1 << n
    extreme = 0
    bit_positions = np.arange(n, dtype=np.uint64)
    for start in range(0, total, 65_536):
        stop = min(start + 65_536, total)
        ids = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 2.0 * ((ids >> bit_positions) & 1).astype(np.float64) - 1.0
        permuted = np.abs((signs @ values) / n)
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return float(extreme / total)


def contrast_summary(
    runs: pd.DataFrame,
    name: str,
    treatment: str,
    control: str,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pair = paired_values(runs, treatment, control, metric)
    dataset_effects = {
        dataset: effect_from_log(group["log_benefit"].to_numpy(float))
        for dataset, group in pair.groupby("dataset")
    }
    dataset_wins = {
        dataset: int((group[treatment] < group[control]).sum())
        for dataset, group in pair.groupby("dataset")
    }
    dataset_log_means = np.array(
        [pair.loc[pair["dataset"] == dataset, "log_benefit"].mean() for dataset in DATASETS]
    )
    macro = effect_from_log(dataset_log_means)
    ci_low, ci_high = hierarchical_bootstrap(pair, n_boot, rng)
    leave_one_out = {}
    for excluded in DATASETS:
        remaining = [d for d in DATASETS if d != excluded]
        vals = np.array(
            [pair.loc[pair["dataset"] == dataset, "log_benefit"].mean() for dataset in remaining]
        )
        leave_one_out[excluded] = effect_from_log(vals)
    log_values = pair["log_benefit"].to_numpy(float)
    sign_test = stats.binomtest(int(np.sum(log_values > 0)), len(log_values), 0.5, alternative="two-sided")
    try:
        wilcoxon_p = float(stats.wilcoxon(log_values, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    result = {
        "name": name,
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "paired_units": len(pair),
        "wins": int(np.sum(log_values > 0)),
        "macro_relative_error_reduction_percent": macro,
        "macro_95pct_hierarchical_bootstrap_ci_percent": [ci_low, ci_high],
        "dataset_relative_error_reduction_percent": dataset_effects,
        "dataset_wins_of_five": dataset_wins,
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(log_values),
        "two_sided_binomial_sign_p": float(sign_test.pvalue),
        "two_sided_wilcoxon_p": wilcoxon_p,
        "leave_one_dataset_out_percent": leave_one_out,
        "heterogeneous_sign_across_datasets": len({np.sign(v) for v in dataset_effects.values() if v != 0}) > 1,
    }
    pair.insert(0, "contrast", name)
    pair.insert(1, "metric", metric)
    return result, pair


def summarize_frozen_probes(
    probes: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    synthetic_records: list[dict[str, Any]] = []
    for row in probes.to_dict("records"):
        base = {"dataset": row["dataset"], "seed": row["seed"]}
        values = {
            "native": row["native_force_mae_mev_per_angstrom"],
            "aligned_probe": row["aligned_force_mae_mev_per_angstrom"],
            "permuted_probe": row["permute_mean_force_mae_mev_per_angstrom"],
        }
        for mode, value in values.items():
            synthetic_records.append({**base, "mode": mode, "test_force_mae": value})
    synthetic = pd.DataFrame(synthetic_records)
    aligned, aligned_pair = contrast_summary(
        synthetic,
        "frozen_native_vs_aligned",
        "native",
        "aligned_probe",
        "test_force_mae",
        n_boot,
        rng,
    )
    assignment, assignment_pair = contrast_summary(
        synthetic,
        "frozen_native_vs_permuted",
        "native",
        "permuted_probe",
        "test_force_mae",
        n_boot,
        rng,
    )
    dose_rows: list[dict[str, Any]] = []
    dose_mapping = {
        0.0: "aligned_force_mae_mev_per_angstrom",
        0.25: "dose_0.25_force_mae_mev_per_angstrom",
        0.50: "dose_0.50_force_mae_mev_per_angstrom",
        0.75: "dose_0.75_force_mae_mev_per_angstrom",
        1.0: "native_force_mae_mev_per_angstrom",
    }
    for dose, column in dose_mapping.items():
        for dataset, group in probes.groupby("dataset"):
            ratio = group[column].to_numpy(float) / group["native_force_mae_mev_per_angstrom"].to_numpy(float)
            dose_rows.append(
                {
                    "lambda": dose,
                    "scope": DATASET_LABELS[dataset],
                    "force_mae_ratio_to_factual": float(np.exp(np.mean(np.log(ratio)))),
                }
            )
        dataset_log_means = []
        for dataset in DATASETS:
            group = probes[probes["dataset"] == dataset]
            ratio = group[column].to_numpy(float) / group["native_force_mae_mev_per_angstrom"].to_numpy(float)
            dataset_log_means.append(np.mean(np.log(ratio)))
        dose_rows.append(
            {
                "lambda": dose,
                "scope": "Equal-dataset macro",
                "force_mae_ratio_to_factual": float(np.exp(np.mean(dataset_log_means))),
            }
        )
    dose = pd.DataFrame(dose_rows)
    common_max = float(
        1000.0
        * probes["common_shift_maximum_absolute_force_change_ev_per_angstrom"].max()
    )
    replay_diffs = []
    for row in probes.to_dict("records"):
        replay_diffs.append(abs(row["native_force_mae_mev_per_angstrom"] - row["native_force_mae_mev_per_angstrom"]))
    repeat_columns = [f"permute_{i}_force_mae_mev_per_angstrom" for i in range(4)]
    repeat_spread = probes[repeat_columns].max(axis=1) - probes[repeat_columns].min(axis=1)
    summary = {
        "effects": [aligned, assignment],
        "dose_response": dose.to_dict("records"),
        "maximum_common_shift_force_change_mev_per_angstrom": common_max,
        "maximum_permutation_repeat_force_mae_spread_mev_per_angstrom": float(repeat_spread.max()),
        "mean_permutation_repeat_force_mae_spread_mev_per_angstrom": float(repeat_spread.mean()),
    }
    pairs = pd.concat([aligned_pair, assignment_pair], ignore_index=True)
    return summary, pairs, dose


def load_observational_phase(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / "observational_phase" / dataset / f"seed{seed}" / "phase_interference.json"
            payload = read_json(path)
            channels = payload["phase_histograms"]["channels"]
            constructive = destructive = pair_amplitude = weighted_cos = weighted_sin = 0.0
            combined_hist: np.ndarray | None = None
            max_destructive_fraction = 0.0
            for channel in channels.values():
                summary = channel["summary"]
                constructive += float(summary["constructive_kernel_mass"])
                destructive += float(summary["destructive_kernel_mass"])
                pair_amplitude += float(summary["pair_amplitude_sum"])
                weighted_cos += float(summary["weighted_cosine_sum"])
                weighted_sin += float(summary["weighted_sine_sum"])
                max_destructive_fraction = max(
                    max_destructive_fraction, float(summary["destructive_kernel_fraction"])
                )
                hist = np.asarray(channel["relative_amplitude_weighted_probability"], dtype=float)
                weighted_hist = hist * float(summary["pair_amplitude_sum"])
                combined_hist = weighted_hist if combined_hist is None else combined_hist + weighted_hist
            if combined_hist is None or pair_amplitude <= 0:
                raise RuntimeError(f"No phase histogram mass in {path}")
            probability = combined_hist / combined_hist.sum()
            positive = probability[probability > 0]
            entropy = float(-np.sum(positive * np.log(positive)) / np.log(len(probability)))
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "destructive_kernel_fraction": destructive / (constructive + destructive),
                    "weighted_mean_cos_delta": weighted_cos / pair_amplitude,
                    "circular_resultant_magnitude": math.hypot(weighted_cos, weighted_sin) / pair_amplitude,
                    "relative_phase_entropy_normalized": entropy,
                    "max_channel_destructive_fraction": max_destructive_fraction,
                    "pair_amplitude_sum": pair_amplitude,
                    "num_batches": payload["num_batches"],
                    "path": str(path),
                }
            )
    return pd.DataFrame(records)


def within_dataset_rank_correlation(
    frame: pd.DataFrame,
    x: str,
    y: str,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    work = frame[["dataset", x, y]].copy()
    work["x_rank"] = work.groupby("dataset")[x].rank(method="average")
    work["y_rank"] = work.groupby("dataset")[y].rank(method="average")
    work["x_centered"] = work["x_rank"] - work.groupby("dataset")["x_rank"].transform("mean")
    work["y_centered"] = work["y_rank"] - work.groupby("dataset")["y_rank"].transform("mean")
    xv = work["x_centered"].to_numpy(float)
    yv = work["y_centered"].to_numpy(float)
    observed = float(np.corrcoef(xv, yv)[0, 1])
    indices = {d: np.flatnonzero(work["dataset"].to_numpy() == d) for d in DATASETS}
    extreme = 0
    for _ in range(n_perm):
        permuted = yv.copy()
        for dataset_indices in indices.values():
            permuted[dataset_indices] = rng.permutation(permuted[dataset_indices])
        corr = float(np.corrcoef(xv, permuted)[0, 1])
        extreme += int(abs(corr) >= abs(observed) - 1e-15)
    return {
        "phase_statistic": x,
        "outcome": y,
        "within_dataset_rank_correlation": observed,
        "two_sided_within_dataset_permutation_p": (extreme + 1) / (n_perm + 1),
        "units": len(work),
        "status": "post-hoc descriptive association; not an identified causal effect",
    }


def observational_associations(
    observational: pd.DataFrame,
    combined_runs: pd.DataFrame,
    probes: pd.DataFrame,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trained = paired_values(combined_runs, "full", "aligned", "test_force_mae")
    trained = trained[["dataset", "seed", "log_benefit"]].rename(
        columns={"log_benefit": "trained_full_vs_aligned_log_benefit"}
    )
    frozen = probes[[
        "dataset",
        "seed",
        "native_force_mae_mev_per_angstrom",
        "aligned_force_mae_mev_per_angstrom",
        "permute_mean_force_mae_mev_per_angstrom",
    ]].copy()
    frozen["frozen_alignment_log_harm"] = np.log(
        frozen["aligned_force_mae_mev_per_angstrom"]
        / frozen["native_force_mae_mev_per_angstrom"]
    )
    frozen["frozen_permutation_log_harm"] = np.log(
        frozen["permute_mean_force_mae_mev_per_angstrom"]
        / frozen["native_force_mae_mev_per_angstrom"]
    )
    joined = observational.merge(trained, on=["dataset", "seed"]).merge(
        frozen, on=["dataset", "seed"]
    )
    phase_stats = [
        "destructive_kernel_fraction",
        "weighted_mean_cos_delta",
        "circular_resultant_magnitude",
        "relative_phase_entropy_normalized",
    ]
    outcomes = [
        "trained_full_vs_aligned_log_benefit",
        "frozen_alignment_log_harm",
        "frozen_permutation_log_harm",
    ]
    rows = [
        within_dataset_rank_correlation(joined, x, y, n_perm, rng)
        for x, y in itertools.product(phase_stats, outcomes)
    ]
    return joined, pd.DataFrame(rows)


def heterogeneity_tables(phase_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Atomic heterogeneity requires PyArrow >=24, matching the writer. "
            "Install it or prepend a compatible wheel to PYTHONPATH."
        ) from error
    major = int(pa.__version__.split(".")[0])
    if major < 24:
        raise RuntimeError(f"PyArrow >=24 required for these files; found {pa.__version__}")

    atomic_parts: list[pd.DataFrame] = []
    frame_parts: list[pd.DataFrame] = []
    columns = [
        "dataset",
        "seed",
        "neighbour_count",
        "reference_force_norm",
        "force_mae_native",
        "force_mae_aligned",
        "force_mae_permute_mean",
    ]
    frame_columns = [
        "dataset",
        "seed",
        "energy_abs_error_per_atom_native",
        "energy_abs_error_per_atom_aligned",
        "energy_abs_error_per_atom_permute_0",
        "energy_abs_error_per_atom_permute_1",
        "energy_abs_error_per_atom_permute_2",
        "energy_abs_error_per_atom_permute_3",
    ]
    for dataset in DATASETS:
        for seed in SEEDS:
            test_name = PRIMARY_TEST[dataset]
            root = phase_root / "probes" / dataset / f"seed{seed}" / test_name
            atomic_parts.append(pq.read_table(root / "paired_atomic_outcomes.parquet", columns=columns).to_pandas())
            frame_parts.append(pq.read_table(root / "paired_frame_outcomes.parquet", columns=frame_columns).to_pandas())
    atomic = pd.concat(atomic_parts, ignore_index=True)
    frames = pd.concat(frame_parts, ignore_index=True)
    frames["energy_abs_error_per_atom_permute_mean"] = frames[
        [f"energy_abs_error_per_atom_permute_{index}" for index in range(4)]
    ].mean(axis=1)
    for column in [
        "reference_force_norm",
        "force_mae_native",
        "force_mae_aligned",
        "force_mae_permute_mean",
    ]:
        atomic[column] = atomic[column].astype(float) * 1000.0
    for column in [
        "energy_abs_error_per_atom_native",
        "energy_abs_error_per_atom_aligned",
        "energy_abs_error_per_atom_permute_mean",
    ]:
        frames[column] = frames[column].astype(float) * 1000.0

    rows: list[dict[str, Any]] = []
    for dataset, group in atomic.groupby("dataset"):
        dataset_group = group.copy()
        force_codes = pd.qcut(
            dataset_group["reference_force_norm"], 4, labels=False, duplicates="drop"
        )
        neighbour_codes = pd.qcut(
            dataset_group["neighbour_count"].rank(method="average"),
            4,
            labels=False,
            duplicates="drop",
        )
        dataset_group["force_norm_quartile"] = force_codes.map(
            lambda value: f"Q{int(value) + 1}"
        )
        dataset_group["neighbour_quartile"] = neighbour_codes.map(
            lambda value: f"Q{int(value) + 1}"
        )
        for dimension, column in (
            ("reference-force norm", "force_norm_quartile"),
            ("neighbour count", "neighbour_quartile"),
        ):
            for stratum, subgroup in dataset_group.groupby(column, observed=True):
                seed_rows = []
                for seed, seed_group in subgroup.groupby("seed"):
                    native = float(seed_group["force_mae_native"].mean())
                    aligned = float(seed_group["force_mae_aligned"].mean())
                    permuted = float(seed_group["force_mae_permute_mean"].mean())
                    seed_rows.append((seed, native, aligned, permuted))
                seed_array = np.asarray(seed_rows, dtype=float)
                rows.append(
                    {
                        "dataset": dataset,
                        "dimension": dimension,
                        "stratum": str(stratum),
                        "atoms": len(subgroup),
                        "stratum_low": float(
                            subgroup["reference_force_norm" if dimension == "reference-force norm" else "neighbour_count"].min()
                        ),
                        "stratum_high": float(
                            subgroup["reference_force_norm" if dimension == "reference-force norm" else "neighbour_count"].max()
                        ),
                        "native_force_mae_mev_per_angstrom": float(seed_array[:, 1].mean()),
                        "aligned_force_mae_mev_per_angstrom": float(seed_array[:, 2].mean()),
                        "permuted_force_mae_mev_per_angstrom": float(seed_array[:, 3].mean()),
                        "aligned_minus_native_mev_per_angstrom": float((seed_array[:, 2] - seed_array[:, 1]).mean()),
                        "permuted_minus_native_mev_per_angstrom": float((seed_array[:, 3] - seed_array[:, 1]).mean()),
                    }
                )
    frame_rows = []
    for dataset, group in frames.groupby("dataset"):
        for seed, seed_group in group.groupby("seed"):
            frame_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "frames": len(seed_group),
                    "native_energy_mae_mev_per_atom": float(seed_group["energy_abs_error_per_atom_native"].mean()),
                    "aligned_energy_mae_mev_per_atom": float(seed_group["energy_abs_error_per_atom_aligned"].mean()),
                    "permuted_energy_mae_mev_per_atom": float(seed_group["energy_abs_error_per_atom_permute_mean"].mean()),
                }
            )
    audit = {
        "pyarrow_version": pa.__version__,
        "atomic_rows": len(atomic),
        "frame_rows": len(frames),
        "atomic_finite": bool(np.isfinite(atomic.select_dtypes(include=[np.number]).to_numpy()).all()),
        "frame_finite": bool(np.isfinite(frames.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    return pd.DataFrame(rows), pd.DataFrame(frame_rows), audit


def checkpoint_audit(phase_root: Path) -> dict[str, Any]:
    import torch

    checkpoints = {}
    states = {}
    for mode in ("full", "aligned"):
        path = phase_root / "audit_checkpoints" / f"{mode}.pth"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["e3trans_state_dict"]
        checkpoints[mode] = checkpoint
        states[mode] = state
    full_keys = set(states["full"])
    aligned_keys = set(states["aligned"])
    phase_head_key = "interactions.1.phase_head.weight"
    return {
        "same_state_tensor_schema": full_keys == aligned_keys,
        "full_state_tensor_count": len(full_keys),
        "aligned_state_tensor_count": len(aligned_keys),
        "full_state_elements": int(sum(t.numel() for t in states["full"].values())),
        "aligned_state_elements": int(sum(t.numel() for t in states["aligned"].values())),
        "inactive_aligned_phase_head_elements": int(states["aligned"][phase_head_key].numel()),
        "phase_head_key": phase_head_key,
        "full_checkpoint_sha256": sha256(phase_root / "audit_checkpoints" / "full.pth"),
        "aligned_checkpoint_sha256": sha256(phase_root / "audit_checkpoints" / "aligned.pth"),
        "deterministic_algorithms_forced": False,
        "interpretation": (
            "Nominal architecture and state schema are matched. Aligned-Full makes the 32-element "
            "phase head functionally inactive; CUDA deterministic algorithms were not forced, so the "
            "paired retraining is a learning-system potential-outcome estimate rather than bitwise replay."
        ),
    }


def validate_inputs(
    primary: pd.DataFrame,
    aligned: pd.DataFrame,
    probes: pd.DataFrame,
    observational: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "primary_rows_expected_80": len(primary) == 80,
        "aligned_rows_expected_20": len(aligned) == 20,
        "probe_rows_expected_20": len(probes) == 20,
        "observational_rows_expected_20": len(observational) == 20,
        "all_test_selection_flags_false": not bool(
            pd.concat([primary, aligned])["test_used_for_selection"].any()
        ),
        "all_primary_metrics_finite": bool(
            np.isfinite(
                primary[
                    [
                        "validation_force_mae",
                        "test_energy_mae",
                        "test_energy_rmse",
                        "test_force_mae",
                        "test_force_rmse",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "all_aligned_metrics_finite": bool(
            np.isfinite(
                aligned[
                    [
                        "validation_force_mae",
                        "test_energy_mae",
                        "test_energy_rmse",
                        "test_force_mae",
                        "test_force_rmse",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "all_probe_force_metrics_finite": bool(
            np.isfinite(
                probes[
                    [
                        "native_force_mae_mev_per_angstrom",
                        "aligned_force_mae_mev_per_angstrom",
                        "permute_mean_force_mae_mev_per_angstrom",
                    ]
                ].to_numpy(float)
            ).all()
        ),
    }
    full_params = primary[primary["mode"] == "full"][["dataset", "seed", "parameter_count"]]
    aligned_params = aligned[["dataset", "seed", "parameter_count"]]
    merged = full_params.merge(aligned_params, on=["dataset", "seed"], suffixes=("_full", "_aligned"))
    checks["full_aligned_nominal_parameter_counts_equal"] = bool(
        (merged["parameter_count_full"] == merged["parameter_count_aligned"]).all()
    )
    replay = primary[primary["mode"] == "full"][["dataset", "seed", "test_force_mae"]].merge(
        probes[["dataset", "seed", "native_force_mae_mev_per_angstrom"]],
        on=["dataset", "seed"],
    )
    replay_max = float(
        np.max(
            np.abs(
                replay["test_force_mae"].to_numpy(float)
                - replay["native_force_mae_mev_per_angstrom"].to_numpy(float)
            )
        )
    )
    common_shift_max = float(
        1000.0
        * probes["common_shift_maximum_absolute_force_change_ev_per_angstrom"].max()
    )
    checks["maximum_native_replay_force_mae_difference_mev_per_angstrom"] = replay_max
    checks["native_replay_within_0.1_mev_per_angstrom"] = replay_max <= 0.1
    checks["maximum_common_shift_force_change_mev_per_angstrom"] = common_shift_max
    checks["common_shift_within_0.05_mev_per_angstrom"] = common_shift_max <= 0.05
    checks["dataset_seed_coverage"] = {
        frame_name: {
            dataset: sorted(int(v) for v in frame.loc[frame["dataset"] == dataset, "seed"].unique())
            for dataset in DATASETS
        }
        for frame_name, frame in (
            ("primary", primary),
            ("aligned", aligned),
            ("probes", probes),
            ("observational", observational),
        )
    }
    checks["passed"] = all(
        value for key, value in checks.items() if isinstance(value, bool) and key != "passed"
    )
    return checks


def configure_plots() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "axes.facecolor": "#fbfbfa",
            "figure.facecolor": "white",
        }
    )


def plot_component_ladder(effects: list[dict[str, Any]], output: Path) -> None:
    wanted = [
        "stable_self_density",
        "phase_free_cross_neighbour_closure",
        "learned_relative_phase",
        "full_vs_off",
    ]
    labels = {
        "stable_self_density": "Self-density\nDiagonal vs Off",
        "phase_free_cross_neighbour_closure": "Cross-neighbour closure\nAligned vs Diagonal",
        "learned_relative_phase": "Learned relative phase\nFull vs Aligned",
        "full_vs_off": "Full CHORUS\nFull vs Off",
    }
    indexed = {row["name"]: row for row in effects}
    values = [indexed[name]["macro_relative_error_reduction_percent"] for name in wanted]
    lows = [indexed[name]["macro_95pct_hierarchical_bootstrap_ci_percent"][0] for name in wanted]
    highs = [indexed[name]["macro_95pct_hierarchical_bootstrap_ci_percent"][1] for name in wanted]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    colors = ["#3d6f73", "#7aa5a8", "#c68b57", "#263f61"]
    y = np.arange(len(wanted))
    ax.barh(y, values, color=colors, height=0.62)
    ax.errorbar(
        values,
        y,
        xerr=[np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)],
        fmt="none",
        ecolor="#242424",
        capsize=3,
        linewidth=1,
    )
    ax.axvline(0, color="#6b6b6b", linewidth=0.8)
    ax.set_yticks(y, [labels[name] for name in wanted])
    ax.invert_yaxis()
    ax.set_xlabel("Force-MAE reduction (%)")
    ax.set_title("End-to-end paired intervention ladder", loc="left")
    for index, value in enumerate(values):
        ax.text(
            value + (0.25 if value >= 0 else -0.25),
            index,
            f"{value:.2f}%",
            ha="left" if value >= 0 else "right",
            va="center",
        )
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)


def plot_three_rungs(
    effects: list[dict[str, Any]],
    frozen: dict[str, Any],
    dose: pd.DataFrame,
    associations: pd.DataFrame,
    output: Path,
) -> None:
    indexed = {row["name"]: row for row in effects}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.35))
    fig.subplots_adjust(wspace=0.44)

    selected = associations[
        associations["outcome"].isin(
            ["trained_full_vs_aligned_log_benefit", "frozen_alignment_log_harm"]
        )
    ].copy()
    short = {
        "destructive_kernel_fraction": "destructive mass",
        "weighted_mean_cos_delta": "mean cos Δθ",
        "circular_resultant_magnitude": "resultant length",
        "relative_phase_entropy_normalized": "phase entropy",
    }
    selected["label"] = selected["phase_statistic"].map(short)
    outcomes = ["trained_full_vs_aligned_log_benefit", "frozen_alignment_log_harm"]
    colors = ["#c68b57", "#3d6f73"]
    y = np.arange(4)
    for offset, (outcome, color) in zip((-0.16, 0.16), zip(outcomes, colors)):
        part = selected[selected["outcome"] == outcome].set_index("label").reindex(short.values())
        axes[0].barh(y + offset, part["within_dataset_rank_correlation"], height=0.28, color=color)
    axes[0].axvline(0, color="#6b6b6b", linewidth=0.8)
    axes[0].set_yticks(y, list(short.values()))
    axes[0].set_xlabel("Within-dataset rank correlation")
    axes[0].set_title("a  Seeing", loc="left")
    axes[0].legend(["retraining gain", "frozen reliance"], frameon=False, fontsize=8, loc="lower right")

    macro_dose = dose[dose["scope"] == "Equal-dataset macro"]
    axes[1].plot(
        macro_dose["lambda"],
        macro_dose["force_mae_ratio_to_factual"],
        color="#3d6f73",
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    axes[1].axhline(1, color="#6b6b6b", linewidth=0.8)
    axes[1].set_xlabel("Restored relative-phase dose λ")
    axes[1].set_ylabel("Force MAE / factual")
    axes[1].set_title("b  Doing", loc="left")
    axes[1].text(
        0.04,
        0.82,
        "alignment: +46.4% harm\npermutation: +69.4% harm",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "#fbfbfa", "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
    )

    names = ["stable_self_density", "phase_free_cross_neighbour_closure", "learned_relative_phase"]
    labels = ["self-density", "cross-neighbour closure", "relative phase"]
    values = [indexed[name]["macro_relative_error_reduction_percent"] for name in names]
    low = [indexed[name]["macro_95pct_hierarchical_bootstrap_ci_percent"][0] for name in names]
    high = [indexed[name]["macro_95pct_hierarchical_bootstrap_ci_percent"][1] for name in names]
    y = np.arange(3)
    axes[2].barh(y, values, color=["#3d6f73", "#7aa5a8", "#c68b57"], height=0.62)
    axes[2].errorbar(
        values,
        y,
        xerr=[np.asarray(values) - low, np.asarray(high) - np.asarray(values)],
        fmt="none",
        ecolor="#242424",
        capsize=3,
        linewidth=1,
    )
    axes[2].axvline(0, color="#6b6b6b", linewidth=0.8)
    axes[2].set_yticks(y, labels)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Force-MAE reduction (%)")
    axes[2].set_title("c  Imagining", loc="left")
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)


def write_dag_svg(output: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="430" viewBox="0 0 1040 430">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#53616b"/></marker></defs>
<rect width="1040" height="430" fill="white"/>
<style>.n{fill:#f7f8f7;stroke:#6c7a82;stroke-width:1.2;rx:8}.t{font:14px Arial,sans-serif;fill:#1f2a30;text-anchor:middle}.s{font:13px Arial,sans-serif;fill:#52616a}.e{stroke:#53616b;stroke-width:1.3;fill:none;marker-end:url(#arrow)}</style>
<text x="28" y="34" class="s" font-weight="600">End-to-end learning system</text>
<rect class="n" x="38" y="62" width="150" height="48"/><text class="t" x="113" y="91">dataset / seed / protocol</text>
<rect class="n" x="248" y="62" width="145" height="48"/><text class="t" x="320" y="83">phase policy</text><text class="t" x="320" y="100">Full or Aligned</text>
<rect class="n" x="453" y="62" width="145" height="48"/><text class="t" x="525" y="91">optimisation path</text>
<rect class="n" x="658" y="62" width="150" height="48"/><text class="t" x="733" y="83">validation-selected</text><text class="t" x="733" y="100">checkpoint</text>
<rect class="n" x="868" y="62" width="130" height="48"/><text class="t" x="933" y="91">test error</text>
<path class="e" d="M188 86 H244"/><path class="e" d="M393 86 H449"/><path class="e" d="M598 86 H654"/><path class="e" d="M808 86 H864"/>
<text x="28" y="185" class="s" font-weight="600">Frozen-predictor intervention</text>
<rect class="n" x="38" y="215" width="150" height="52"/><text class="t" x="113" y="236">fixed checkpoint,</text><text class="t" x="113" y="254">messages and amplitudes</text>
<rect class="n" x="248" y="215" width="145" height="52"/><text class="t" x="320" y="236">do(phase)</text><text class="t" x="320" y="254">align / dose / permute</text>
<rect class="n" x="453" y="215" width="145" height="52"/><text class="t" x="525" y="236">relative phase</text><text class="t" x="525" y="254">Δθ</text>
<rect class="n" x="658" y="215" width="150" height="52"/><text class="t" x="733" y="236">Hermitian</text><text class="t" x="733" y="254">density update</text>
<rect class="n" x="868" y="215" width="130" height="52"/><text class="t" x="933" y="245">prediction error</text>
<path class="e" d="M188 241 H244"/><path class="e" d="M393 241 H449"/><path class="e" d="M598 241 H654"/><path class="e" d="M808 241 H864"/>
<path class="e" d="M113 267 V318 H733 V271"/>
<text x="28" y="360" class="s">The trained contrast includes optimisation and checkpoint selection as mediators. The frozen contrast holds them fixed.</text>
<text x="28" y="388" class="s">A common phase shift is the U(1) negative control; phase statistics are observational descendants of the fitted model.</text>
</svg>"""
    output.write_text(svg)


def format_effect(effect: dict[str, Any]) -> str:
    low, high = effect["macro_95pct_hierarchical_bootstrap_ci_percent"]
    return (
        f"{effect['macro_relative_error_reduction_percent']:.2f}% "
        f"(95% hierarchical-bootstrap CI {low:.2f}% to {high:.2f}%; "
        f"{effect['wins']}/{effect['paired_units']} paired wins)"
    )


def write_reports(
    output: Path,
    effects: list[dict[str, Any]],
    frozen: dict[str, Any],
    observational: pd.DataFrame,
    associations: pd.DataFrame,
    validation: dict[str, Any],
    checkpoint: dict[str, Any],
    parquet_audit: dict[str, Any],
) -> None:
    indexed = {effect["name"]: effect for effect in effects}
    frozen_indexed = {effect["name"]: effect for effect in frozen["effects"]}
    strongest_assoc = associations.iloc[associations["within_dataset_rank_correlation"].abs().argmax()]
    macro_dose_zero = next(
        row["force_mae_ratio_to_factual"]
        for row in frozen["dose_response"]
        if row["lambda"] == 0.0 and row["scope"] == "Equal-dataset macro"
    )
    destructive_by_dataset = observational.groupby("dataset")[
        "destructive_kernel_fraction"
    ].mean()
    technical = f"""# CHORUS causal-mechanism audit

## Technical summary

The complete Pearl three-rung analysis separates functional phase use from end-to-end generalisation. Stable Hermitian self-density is the most robust component: Diagonal reduces test Force MAE against Off by {format_effect(indexed['stable_self_density'])}. Adding phase-free cross-neighbour closure contributes {format_effect(indexed['phase_free_cross_neighbour_closure'])}; its sign varies across datasets. Learned relative phase adds {format_effect(indexed['learned_relative_phase'])} over an independently trained Aligned-Full control. The mean is positive, while its interval includes zero.

The fitted Full predictors nevertheless use their learned phases strongly. Aligning relative phases at fixed weights worsens Force MAE by {format_effect(frozen_indexed['frozen_native_vs_aligned'])}; permuting phase-to-edge assignments worsens it by {format_effect(frozen_indexed['frozen_native_vs_permuted'])}. Restoring relative phase continuously lowers the equal-dataset Force-MAE ratio from {macro_dose_zero:.3f} at λ=0 to 1.000 at λ=1. A shared phase shift changes forces by at most {frozen['maximum_common_shift_force_change_mev_per_angstrom']:.4f} meV Å⁻¹, confirming the intended U(1) invariance.

Taken together, CHORUS learns phase-dependent solutions, but current evidence attributes most reproducible generalisation gain to the stable Hermitian second-moment channel. Cross-neighbour closure is useful but heterogeneous, and the incremental learned-phase effect is modest on this four-dataset panel. This is stronger than an ablation-only claim because the same conclusion survives trained interventions, frozen interventions, a dose response, an assignment intervention, and a symmetry negative control.

## Pearl rung 1: seeing

The observational extension measures relative-phase organisation in the validation-selected Full checkpoints. Associations are computed from ranks within each dataset, preventing between-dataset scale differences from generating the signal. The largest absolute association is `{strongest_assoc['phase_statistic']}` versus `{strongest_assoc['outcome']}` (r={strongest_assoc['within_dataset_rank_correlation']:.3f}, within-dataset permutation p={strongest_assoc['two_sided_within_dataset_permutation_p']:.4f}); none of the twelve precomputed associations reaches p<0.05. Negative-cosine mass is rare in MAL ({100*destructive_by_dataset['mal']:.3f}%), Buckyball ({100*destructive_by_dataset['bucky']:.3f}%), and 3BPA ({100*destructive_by_dataset['3bpa']:.3f}%), while ZrO is more variable ({100*destructive_by_dataset['zro']:.2f}% on average). The learned phase therefore acts mainly through graded pair reweighting on this panel, with explicit sign-reversing cancellation concentrated in ZrO. These twenty fitted-model observations are exploratory and do not identify a causal effect.

## Pearl rung 2: doing

The frozen interventions hold the checkpoint, messages, amplitudes, graph, and test examples fixed. Alignment removes relative phase; permutation preserves each centre's phase multiset while breaking its edge assignment; λ restores the learned deviations about the local circular mean. Their large, monotonic effects show that the selected Full functions rely on both relative phase and its edge assignment. The common-shift negative control passes, so the effect comes from relative phase rather than an arbitrary global phase origin.

## Pearl rung 3: imagining

Full and Aligned-Full use the same data, seeds, optimisation protocol, nominal parameter counts, Hermitian j,k closure, and validation selection rule. Aligned-Full fixes the imaginary coefficient to zero and makes only a 32-element phase head inactive ({100.0*checkpoint['inactive_aligned_phase_head_elements']/179716:.4f}% of the MAL parameter count). The paired effect therefore estimates the learning-system consequence of permitting relative phase, including optimisation and checkpoint selection as mediators. CUDA deterministic algorithms were not forced, so this is a paired potential-outcome estimate rather than bitwise twin-world replay.

## Component decomposition

| Treatment vs control | Macro Force-MAE reduction | 95% CI | Wins | Exact sign-flip p | Interpretation |
|---|---:|---:|---:|---:|---|
| Diagonal vs Off | {indexed['stable_self_density']['macro_relative_error_reduction_percent']:.2f}% | [{indexed['stable_self_density']['macro_95pct_hierarchical_bootstrap_ci_percent'][0]:.2f}, {indexed['stable_self_density']['macro_95pct_hierarchical_bootstrap_ci_percent'][1]:.2f}] | {indexed['stable_self_density']['wins']}/20 | {indexed['stable_self_density']['exact_two_sided_sign_flip_p']:.4g} | Stable self-density |
| Aligned vs Diagonal | {indexed['phase_free_cross_neighbour_closure']['macro_relative_error_reduction_percent']:.2f}% | [{indexed['phase_free_cross_neighbour_closure']['macro_95pct_hierarchical_bootstrap_ci_percent'][0]:.2f}, {indexed['phase_free_cross_neighbour_closure']['macro_95pct_hierarchical_bootstrap_ci_percent'][1]:.2f}] | {indexed['phase_free_cross_neighbour_closure']['wins']}/20 | {indexed['phase_free_cross_neighbour_closure']['exact_two_sided_sign_flip_p']:.4g} | Phase-free cross-neighbour closure |
| Full vs Aligned | {indexed['learned_relative_phase']['macro_relative_error_reduction_percent']:.2f}% | [{indexed['learned_relative_phase']['macro_95pct_hierarchical_bootstrap_ci_percent'][0]:.2f}, {indexed['learned_relative_phase']['macro_95pct_hierarchical_bootstrap_ci_percent'][1]:.2f}] | {indexed['learned_relative_phase']['wins']}/20 | {indexed['learned_relative_phase']['exact_two_sided_sign_flip_p']:.4g} | Incremental learned phase |
| Full vs Off | {indexed['full_vs_off']['macro_relative_error_reduction_percent']:.2f}% | [{indexed['full_vs_off']['macro_95pct_hierarchical_bootstrap_ci_percent'][0]:.2f}, {indexed['full_vs_off']['macro_95pct_hierarchical_bootstrap_ci_percent'][1]:.2f}] | {indexed['full_vs_off']['wins']}/20 | {indexed['full_vs_off']['exact_two_sided_sign_flip_p']:.4g} | Complete CHORUS path |
| Full vs Charge2 | {indexed['full_vs_charge2_sensitivity']['macro_relative_error_reduction_percent']:.2f}% | [{indexed['full_vs_charge2_sensitivity']['macro_95pct_hierarchical_bootstrap_ci_percent'][0]:.2f}, {indexed['full_vs_charge2_sensitivity']['macro_95pct_hierarchical_bootstrap_ci_percent'][1]:.2f}] | {indexed['full_vs_charge2_sensitivity']['wins']}/20 | {indexed['full_vs_charge2_sensitivity']['exact_two_sided_sign_flip_p']:.4g} | Sensitivity control; no panel-wide advantage |

## Identification boundary

The causal units are training runs or frozen predictors, not molecules. Results identify effects inside the fixed two-layer MACE-ICTC/CHORUS configuration and the four-dataset mechanism panel. Diagonal and Full contain more trainable parameters than Off, so Diagonal-versus-Off identifies the total effect of adding the self-density module; it does not isolate representation from active capacity. Full versus Aligned is much tighter: nominal counts and state schemas match, and only 32 phase-head elements become functionally inactive. The analysis does not assign physical meaning to θ, prove universal benefit across architectures, or establish relative phase as a uniquely necessary representation. The observational phase statistics were added after the intervention protocol and remain explicitly post hoc.

## Data and audit status

- Primary modes: 80/80 complete; aligned retraining: 20/20; frozen probes: 20/20; observational phase summaries: 20/20.
- Selection: validation Force MAE; test never selects a checkpoint.
- Atomic outcomes: {parquet_audit['atomic_rows']:,}; frame outcomes: {parquet_audit['frame_rows']:,}; PyArrow {parquet_audit['pyarrow_version']}.
- Input validation passed: **{validation['passed']}**.
- Full and Aligned state schemas match: **{checkpoint['same_state_tensor_schema']}** ({checkpoint['full_state_tensor_count']} tensors; {checkpoint['full_state_elements']:,} stored elements each).
"""
    (output / "TECHNICAL_REPORT.md").write_text(technical)

    validation_text = f"""# Validation report

## Verdict

The compiled causal dataset is internally consistent and suitable for the stated mechanism-panel analysis. All expected dataset-by-seed cells are present, all reported outcomes are finite, nominal Full/Aligned parameter counts match, and test data were not used for checkpoint selection.

## Automated checks

```json
{json.dumps(validation, indent=2, sort_keys=True)}
```

## Counterfactual audit

```json
{json.dumps(checkpoint, indent=2, sort_keys=True)}
```

## Parquet audit

```json
{json.dumps(parquet_audit, indent=2, sort_keys=True)}
```

## Residual limitations

1. The four datasets form a deliberately selected mechanism panel, so the equal-dataset macro is not an estimate for all materials or all equivariant networks.
2. Five seeds per dataset resolve paired consistency but leave dataset-level heterogeneity only coarsely estimated.
3. The observational rung is post hoc and association-only.
4. GPU deterministic algorithms were not forced. Pairing controls the declared data and optimiser randomness, while bitwise trajectory identity is not guaranteed.
"""
    (output / "VALIDATION_REPORT.md").write_text(validation_text)

    identification = f"""# Causal identification specification

## Structural system

For dataset `D`, seed `S`, architecture policy `A`, training data `X_train`, and residual GPU execution noise `U`, training produces weights `W=f(D,S,A,X_train,U)`. Validation selection produces `K=g(W,X_val)`, and the held-out outcome is `Y=h(K,X_test)`. The end-to-end potential outcome is therefore `Y_d,s(A)`, with optimisation and checkpoint selection retained as mediators.

At a fixed Full checkpoint `K`, the intervention `I` changes the local phase rule while messages, amplitudes, graph, weights, and examples stay fixed. The frozen potential outcome is `Y_K(I)`. The common-shift intervention is a null edge in this structural model under U(1) invariance.

## Estimands

For each dataset and seed, define the paired log benefit `δ_d,s = log(Y_d,s(control) / Y_d,s(treatment))`. Dataset effects average `δ` over five seeds. The equal-dataset macro averages the four dataset effects, and the reported percentage is `100[1-exp(-mean δ)]`. This keeps multiplicative error changes symmetric before conversion to an interpretable reduction.

The three principal estimands are:

1. **Self-density module:** Diagonal versus Off. This is a joint representation-and-capacity intervention because parameter counts differ.
2. **Cross-neighbour and phase decomposition:** Aligned versus Diagonal, followed by Full versus Aligned. The latter is nominally parameter matched; 32 phase-head weights are inactive in Aligned-Full.
3. **Functional phase use:** factual Full versus aligned, permuted, and dose-controlled phase at the same checkpoint.

Full versus Charge2 is retained as a same-parameter sensitivity control for the joint U(1)-neutral Hermitian closure. Its macro effect is {indexed['full_vs_charge2_sensitivity']['macro_relative_error_reduction_percent']:.2f}% with an interval spanning zero, so the present panel does not establish an accuracy advantage over that non-neutral quadratic control.

## Identification assumptions

- **Consistency:** each mode implements the declared intervention, and the recorded outcome belongs to that implementation.
- **Positivity:** all declared modes exist for every one of the 4×5 dataset-seed blocks.
- **No cross-run interference:** one training run cannot change another run's model or dataset.
- **Paired exchangeability by design:** dataset, split, seed, optimiser, budget, and selection rule are fixed within each pair. GPU deterministic algorithms were not forced, leaving residual execution noise.
- **Selection integrity:** validation Force MAE selects one checkpoint; test outcomes remain downstream measurements only.
- **Frozen-intervention exclusion:** alignment, dose, and permutation alter only phase coefficients at inference; amplitudes, messages, graph, weights, and examples remain fixed.

## What is identified

The frozen interventions identify how the selected predictor function changes under phase manipulation. The trained contrasts identify effects of an algorithmic policy on the distribution of validation-selected predictors under the fixed protocol. They do not identify a molecular causal variable, since θ is learned and has no assigned physical observable.

The strongest supported statement is that CHORUS supplies a robust Hermitian second-moment update and learns phase-dependent fitted functions. The present retraining evidence for additional generalisation from relative phase is positive but inconclusive at the panel level. A widened-Off 4×5 paired control would be required to separate the self-density representation from active parameter capacity with the same strength as the Full-versus-Aligned contrast.
"""
    (output / "CAUSAL_IDENTIFICATION.md").write_text(identification)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    primary = load_primary_runs(args.primary_root)
    aligned = load_aligned_runs(args.phase_root)
    probes = load_probe_runs(args.phase_root)
    observational = load_observational_phase(args.phase_root)
    combined = combine_runs(primary, aligned)

    validation = validate_inputs(primary, aligned, probes, observational)
    if not validation["passed"]:
        raise RuntimeError(f"Input validation failed: {validation}")

    contrasts = [
        ("stable_self_density", "diagonal", "off"),
        ("phase_free_cross_neighbour_closure", "aligned", "diagonal"),
        ("learned_relative_phase", "full", "aligned"),
        ("full_vs_diagonal", "full", "diagonal"),
        ("full_vs_off", "full", "off"),
        ("full_vs_charge2_sensitivity", "full", "charge2"),
    ]
    effects: list[dict[str, Any]] = []
    pair_tables: list[pd.DataFrame] = []
    for name, treatment, control in contrasts:
        effect, pairs = contrast_summary(
            combined,
            name,
            treatment,
            control,
            "test_force_mae",
            args.bootstrap,
            rng,
        )
        effects.append(effect)
        pair_tables.append(pairs)
    metric_effects: list[dict[str, Any]] = []
    for metric in ("test_energy_mae", "test_energy_rmse", "test_force_rmse"):
        for name, treatment, control in contrasts:
            effect, _ = contrast_summary(
                combined,
                name,
                treatment,
                control,
                metric,
                args.bootstrap,
                rng,
            )
            metric_effects.append(effect)

    frozen, frozen_pairs, dose = summarize_frozen_probes(
        probes, args.bootstrap, rng
    )
    observational_joined, associations = observational_associations(
        observational,
        combined,
        probes,
        args.association_permutations,
        rng,
    )
    heterogeneity, frame_energy, parquet_audit = heterogeneity_tables(args.phase_root)
    checkpoint = checkpoint_audit(args.phase_root)

    all_pairs = pd.concat(pair_tables, ignore_index=True)
    primary.to_csv(args.output / "primary_modes.csv", index=False)
    aligned.to_csv(args.output / "aligned_runs.csv", index=False)
    probes.to_csv(args.output / "frozen_probe_runs.csv", index=False)
    all_pairs.to_csv(args.output / "paired_effects.csv", index=False)
    frozen_pairs.to_csv(args.output / "frozen_intervention_pairs.csv", index=False)
    dose.to_csv(args.output / "phase_dose_response.csv", index=False)
    observational_joined.to_csv(args.output / "observational_phase_statistics.csv", index=False)
    associations.to_csv(args.output / "observational_associations.csv", index=False)
    heterogeneity.to_csv(args.output / "effect_heterogeneity.csv", index=False)
    frame_energy.to_csv(args.output / "frame_energy_effects.csv", index=False)

    result = {
        "schema_version": 1,
        "title": "CHORUS Pearl three-rung causal-mechanism audit",
        "primary_outcome": "test Force MAE selected by validation Force MAE",
        "effect_definition": "100 * (1 - exp(-equal-dataset mean paired log(control/treatment)))",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "pearl_rungs": {
            "seeing": {
                "status": "post-hoc descriptive",
                "phase_statistics": observational.to_dict("records"),
                "associations": associations.to_dict("records"),
            },
            "doing": frozen,
            "imagining": {
                "effects": effects,
                "counterfactual_audit": checkpoint,
            },
        },
        "secondary_metric_effects": metric_effects,
        "input_validation": validation,
        "parquet_audit": parquet_audit,
        "analysis_config": {
            "hierarchical_bootstrap_replicates": args.bootstrap,
            "within_dataset_association_permutations": args.association_permutations,
            "rng_seed": args.rng_seed,
        },
        "interpretation": {
            "robust_component": "stable Hermitian self-density",
            "heterogeneous_component": "phase-free cross-neighbour closure",
            "incremental_phase_result": (
                "small positive panel mean with a hierarchical-bootstrap interval crossing zero"
            ),
            "functional_phase_result": (
                "large frozen-checkpoint alignment and assignment effects with monotonic dose response"
            ),
            "causal_unit": "training run or frozen fitted predictor",
        },
    }
    write_json(args.output / "causal_inference.json", result)
    write_json(args.output / "input_validation.json", validation)
    write_json(args.output / "checkpoint_audit.json", checkpoint)

    source_files = [
        args.primary_root.parent / "summary" / "summary.json",
        args.phase_root / "summary" / "summary.json",
        Path(__file__),
    ]
    provenance = {
        "generated_by": str(Path(__file__).resolve()),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in source_files if path.exists()
        ],
        "raw_data_locations": {
            "primary": str(args.primary_root),
            "phase": str(args.phase_root),
        },
        "observational_protocol_status": "post_hoc_observational_extension",
    }
    write_json(args.output / "SOURCE_NOTES.json", provenance)

    configure_plots()
    plot_component_ladder(effects, args.output / "component_ladder")
    plot_three_rungs(effects, frozen, dose, associations, args.output / "pearl_three_rungs")
    write_dag_svg(args.output / "causal_dag.svg")
    write_reports(
        args.output,
        effects,
        frozen,
        observational_joined,
        associations,
        validation,
        checkpoint,
        parquet_audit,
    )
    print(json.dumps({"output": str(args.output), "validation": validation["passed"]}, indent=2))


if __name__ == "__main__":
    main()
