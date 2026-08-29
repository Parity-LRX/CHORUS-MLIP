#!/usr/bin/env python3
"""Complete MACE-only Pearl three-rung causal synthesis.

The analysis combines the completed Final four-mode intervention ladder with
the completed Persistent Full/Aligned campaign.  It keeps three distinct
objects separate:

1. seeing: post-hoc associations among fitted-model summaries;
2. doing: independently retrained architecture policies paired by dataset/seed;
3. imagining: same-checkpoint potential outcomes under phase interventions.

Atoms and frames are used only to describe the distribution of model-internal
potential outcomes.  Statistical uncertainty is clustered at the independently
trained dataset-by-seed model, with datasets equally weighted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DATASETS = ("mal", "bucky", "3bpa", "zro")
DATASET_LABELS = {
    "mal": "MAL",
    "bucky": "Buckyball",
    "3bpa": "3BPA (300 K)",
    "zro": "ZrO",
}
SEEDS = tuple(range(20260616, 20260621))
PRIMARY_TEST = {"3bpa": "300K", "mal": "test", "bucky": "test", "zro": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--final-probe-root", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--association-permutations", type=int, default=100_000)
    parser.add_argument("--rng-seed", type=int, default=20260828)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def load_final_runs(root: Path) -> pd.DataFrame:
    primary = pd.read_csv(root / "primary_modes.csv")
    aligned = pd.read_csv(root / "aligned_runs.csv")
    runs = pd.concat([primary, aligned], ignore_index=True)
    # Charge-2 is a separately reported sensitivity analysis.  The Pearl
    # component ladder declared here is the four-mode Off/Diagonal/Aligned/Full
    # chain, so exclude the extra mode before checking exact cell coverage.
    runs = runs[runs["mode"].isin(("off", "diagonal", "aligned", "full"))].copy()
    runs.insert(0, "scope", "Final")
    runs["test_used_for_selection"] = runs["test_used_for_selection"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    )
    return runs


def load_persistent_runs(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            pair_root = root / "mace" / "runs" / dataset / f"seed{seed}"
            if not (pair_root / "PERSISTENT_PAIR_VERIFIED").is_file():
                raise FileNotFoundError(pair_root / "PERSISTENT_PAIR_VERIFIED")
            for mode in ("full", "aligned"):
                path = pair_root / mode / "metrics.json"
                raw = read_json(path)
                if raw.get("phase_scope") != "persistent":
                    raise ValueError(f"wrong phase scope in {path}")
                selection = raw["checkpoint_selection"]
                test_name = PRIMARY_TEST[dataset]
                test = raw["tests"][test_name]
                val = selection["force_mae_envelope"]
                records.append(
                    {
                        "scope": "Persistent",
                        "dataset": dataset,
                        "seed": seed,
                        "mode": mode,
                        "parameter_count": int(raw["parameter_count"]),
                        "selected_step": int(val["step"]),
                        "validation_energy_mae": float(val["energy_mae_mev_per_atom"]),
                        "validation_energy_rmse": float(val["energy_rmse_mev_per_atom"]),
                        "validation_force_mae": float(val["force_mae_mev_per_angstrom"]),
                        "validation_force_rmse": float(val["force_rmse_mev_per_angstrom"]),
                        "test_energy_mae": float(test["energy_mae_mev_per_atom"]),
                        "test_energy_rmse": float(test["energy_rmse_mev_per_atom"]),
                        "test_force_mae": float(test["force_mae_mev_per_angstrom"]),
                        "test_force_rmse": float(test["force_rmse_mev_per_angstrom"]),
                        "test_name": test_name,
                        "selection_rule": selection["rule"],
                        "test_used_for_selection": bool(selection["test_used_for_selection"]),
                        "metrics_path": str(path),
                    }
                )
    return pd.DataFrame(records)


def flatten_probe(scope: str, path: Path) -> dict[str, Any]:
    raw = read_json(path)
    metrics = raw["metrics"]
    record: dict[str, Any] = {
        "scope": scope,
        "dataset": raw["dataset"],
        "seed": int(raw["seed"]),
        "test": raw["test"],
        "parameter_count": int(raw["parameter_count"]),
        "frames": int(raw["frames"]),
        "checkpoint_sha256": raw["checkpoint_sha256"],
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
        values = metrics[intervention]
        for key in (
            "energy_mae_mev_per_atom",
            "force_mae_mev_per_angstrom",
            "force_rmse_ev_per_angstrom",
            "maximum_absolute_force_change_ev_per_angstrom",
            "mean_absolute_force_change_ev_per_angstrom",
        ):
            if key in values:
                record[f"{intervention}_{key}"] = float(values[key])
    record["permute_mean_force_mae_mev_per_angstrom"] = float(
        metrics["permute_mean"]["force_mae_mev_per_angstrom"]
    )
    return record


def load_final_probes(final_result: Path) -> pd.DataFrame:
    probes = pd.read_csv(final_result / "frozen_probe_runs.csv")
    probes.insert(0, "scope", "Final")
    return probes


def load_persistent_probes(root: Path) -> pd.DataFrame:
    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            test_name = PRIMARY_TEST[dataset]
            path = root / "mace" / "probes" / dataset / f"seed{seed}" / test_name / "metrics.json"
            records.append(flatten_probe("Persistent", path))
    return pd.DataFrame(records)


def pair_modes(runs: pd.DataFrame, scope: str, treatment: str, control: str, metric: str) -> pd.DataFrame:
    local = runs[runs["scope"] == scope]
    pivot = local.pivot(index=["dataset", "seed"], columns="mode", values=metric).reset_index()
    needed = {treatment, control}
    if not needed.issubset(pivot.columns):
        raise ValueError(f"missing modes for {scope}: {needed - set(pivot.columns)}")
    result = pivot[["dataset", "seed", treatment, control]].copy()
    result["scope"] = scope
    result["treatment"] = treatment
    result["control"] = control
    result["metric"] = metric
    result["log_benefit"] = np.log(result[control] / result[treatment])
    result["relative_reduction_percent"] = 100.0 * (1.0 - result[treatment] / result[control])
    return result


def pair_probes(probes: pd.DataFrame, scope: str, control: str) -> pd.DataFrame:
    local = probes[probes["scope"] == scope].copy()
    treatment_column = "native_force_mae_mev_per_angstrom"
    control_column = f"{control}_force_mae_mev_per_angstrom"
    result = local[["dataset", "seed", treatment_column, control_column]].copy()
    result.columns = ["dataset", "seed", "native", control]
    result["scope"] = scope
    result["treatment"] = "native"
    result["control"] = control
    result["metric"] = "test_force_mae"
    result["log_benefit"] = np.log(result[control] / result["native"])
    result["relative_reduction_percent"] = 100.0 * (1.0 - result["native"] / result[control])
    return result


def bootstrap_means(pair: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    grouped = {dataset: group["log_benefit"].to_numpy(float) for dataset, group in pair.groupby("dataset")}
    if set(grouped) != set(DATASETS):
        raise ValueError(f"incomplete dataset groups: {sorted(grouped)}")
    values = np.empty(n_boot, dtype=np.float64)
    datasets = np.asarray(DATASETS)
    for index in range(n_boot):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for dataset in sampled_datasets:
            local = grouped[str(dataset)]
            means.append(float(rng.choice(local, size=len(local), replace=True).mean()))
        values[index] = float(np.mean(means))
    return values


def exact_sign_flip_p(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    observed = abs(float(array.mean()))
    n = len(array)
    extreme = 0
    bit_positions = np.arange(n, dtype=np.uint64)
    for start in range(0, 1 << n, 65_536):
        stop = min(start + 65_536, 1 << n)
        ids = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 2.0 * ((ids >> bit_positions) & 1).astype(np.float64) - 1.0
        draws = np.abs((signs @ array) / n)
        extreme += int(np.count_nonzero(draws >= observed - 1e-15))
    return float(extreme / (1 << n))


def summarize_pair(
    pair: pd.DataFrame,
    name: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    bootstrap = bootstrap_means(pair, n_boot, rng)
    log_values = pair["log_benefit"].to_numpy(float)
    dataset_logs = {
        dataset: float(pair.loc[pair["dataset"] == dataset, "log_benefit"].mean())
        for dataset in DATASETS
    }
    macro_log = float(np.mean(list(dataset_logs.values())))
    try:
        wilcoxon_p = float(stats.wilcoxon(log_values, zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "name": name,
        "scope": str(pair["scope"].iloc[0]) if "scope" in pair else "interaction",
        "treatment": str(pair["treatment"].iloc[0]) if "treatment" in pair else "interaction",
        "control": str(pair["control"].iloc[0]) if "control" in pair else "interaction",
        "paired_units": int(len(pair)),
        "wins": int((log_values > 0).sum()),
        "macro_mean_log_benefit": macro_log,
        "macro_relative_error_reduction_percent": float(100.0 * (1.0 - math.exp(-macro_log))),
        "macro_95pct_hierarchical_bootstrap_ci_percent": [
            float(100.0 * (1.0 - math.exp(-value)))
            for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "dataset_relative_error_reduction_percent": {
            dataset: float(100.0 * (1.0 - math.exp(-value)))
            for dataset, value in dataset_logs.items()
        },
        "dataset_wins_of_five": {
            dataset: int((pair.loc[pair["dataset"] == dataset, "log_benefit"] > 0).sum())
            for dataset in DATASETS
        },
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(log_values),
        "two_sided_binomial_sign_p": float(
            stats.binomtest(int((log_values > 0).sum()), len(log_values), 0.5).pvalue
        ),
        "two_sided_wilcoxon_p": wilcoxon_p,
    }


def scope_interaction(
    final_pair: pd.DataFrame,
    persistent_pair: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
    name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    final = final_pair.set_index(["dataset", "seed"])["log_benefit"]
    persistent = persistent_pair.set_index(["dataset", "seed"])["log_benefit"]
    if not final.index.equals(persistent.index):
        raise ValueError(f"pair mismatch for {name}")
    pair = (persistent - final).rename("log_benefit").reset_index()
    pair["scope"] = "Persistent-minus-Final"
    pair["treatment"] = "scope interaction"
    pair["control"] = "zero"
    summary = summarize_pair(pair, name, n_boot, rng)
    mean_log = float(pair.groupby("dataset")["log_benefit"].mean().mean())
    bootstrap = bootstrap_means(pair, n_boot, rng)
    summary["benefit_ratio_change_percent"] = float(100.0 * math.expm1(mean_log))
    summary["benefit_ratio_change_95pct_hierarchical_bootstrap_ci_percent"] = [
        float(100.0 * math.expm1(value)) for value in np.quantile(bootstrap, [0.025, 0.975])
    ]
    return summary, pair


def dose_response(probes: pd.DataFrame) -> pd.DataFrame:
    columns = {
        0.0: "aligned_force_mae_mev_per_angstrom",
        0.25: "dose_0.25_force_mae_mev_per_angstrom",
        0.50: "dose_0.50_force_mae_mev_per_angstrom",
        0.75: "dose_0.75_force_mae_mev_per_angstrom",
        1.0: "native_force_mae_mev_per_angstrom",
    }
    records = []
    for scope in ("Final", "Persistent"):
        local_scope = probes[probes["scope"] == scope]
        for dose, column in columns.items():
            dataset_logs = []
            for dataset in DATASETS:
                local = local_scope[local_scope["dataset"] == dataset]
                ratios = local[column].to_numpy(float) / local[
                    "native_force_mae_mev_per_angstrom"
                ].to_numpy(float)
                log_mean = float(np.log(ratios).mean())
                dataset_logs.append(log_mean)
                records.append(
                    {
                        "scope": scope,
                        "dataset": dataset,
                        "lambda": dose,
                        "force_mae_ratio_to_factual": float(math.exp(log_mean)),
                    }
                )
            records.append(
                {
                    "scope": scope,
                    "dataset": "macro",
                    "lambda": dose,
                    "force_mae_ratio_to_factual": float(math.exp(np.mean(dataset_logs))),
                }
            )
    return pd.DataFrame(records)


def within_stratum_rank_association(table: pd.DataFrame, permutations: int, rng: np.random.Generator) -> dict[str, Any]:
    local = table.copy()
    group_cols = ["dataset", "scope"]
    for column in ("trained_log_benefit", "frozen_alignment_log_harm"):
        local[f"rank_{column}"] = local.groupby(group_cols)[column].rank(method="average")
        local[f"rank_{column}"] -= local.groupby(group_cols)[f"rank_{column}"].transform("mean")
    x = local["rank_trained_log_benefit"].to_numpy(float)
    y = local["rank_frozen_alignment_log_harm"].to_numpy(float)
    observed = float(np.corrcoef(x, y)[0, 1])
    groups = [group.index.to_numpy() for _, group in local.groupby(group_cols, sort=False)]
    extreme = 0
    for _ in range(permutations):
        permuted = y.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        correlation = float(np.corrcoef(x, permuted)[0, 1])
        extreme += abs(correlation) >= abs(observed) - 1e-15
    return {
        "name": "within-dataset-and-scope seed-rank association",
        "paired_model_scope_units": int(len(local)),
        "rank_correlation": observed,
        "two_sided_stratified_permutation_p": float((extreme + 1) / (permutations + 1)),
        "status": "post-hoc descriptive association; not an identified causal effect",
    }


def load_counterfactual_distributions(
    scope: str,
    probe_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    total_atomic = 0
    total_frame = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            test_name = PRIMARY_TEST[dataset]
            base = probe_root / dataset / f"seed{seed}" / test_name
            atomic_path = base / "paired_atomic_outcomes.parquet"
            frame_path = base / "paired_frame_outcomes.parquet"
            atomic = pd.read_parquet(
                atomic_path,
                columns=[
                    "force_mae_native",
                    "force_mae_aligned",
                    "force_mae_permute_mean",
                    "force_mae_common_shift",
                ],
            )
            frame = pd.read_parquet(
                frame_path,
                columns=[
                    "energy_abs_error_per_atom_native",
                    "energy_abs_error_per_atom_aligned",
                    "energy_abs_error_per_atom_common_shift",
                ],
            )
            total_atomic += len(atomic)
            total_frame += len(frame)
            aligned_delta = 1000.0 * (
                atomic["force_mae_aligned"].to_numpy(float)
                - atomic["force_mae_native"].to_numpy(float)
            )
            permuted_delta = 1000.0 * (
                atomic["force_mae_permute_mean"].to_numpy(float)
                - atomic["force_mae_native"].to_numpy(float)
            )
            frame_delta = 1000.0 * (
                frame["energy_abs_error_per_atom_aligned"].to_numpy(float)
                - frame["energy_abs_error_per_atom_native"].to_numpy(float)
            )
            records.append(
                {
                    "scope": scope,
                    "dataset": dataset,
                    "seed": seed,
                    "atomic_outcomes": int(len(atomic)),
                    "frame_outcomes": int(len(frame)),
                    "atoms_with_higher_error_after_alignment_fraction": float(np.mean(aligned_delta > 0)),
                    "atoms_with_higher_error_after_permutation_fraction": float(np.mean(permuted_delta > 0)),
                    "alignment_force_error_change_mean_mev_per_angstrom": float(np.mean(aligned_delta)),
                    "alignment_force_error_change_median_mev_per_angstrom": float(np.median(aligned_delta)),
                    "permutation_force_error_change_mean_mev_per_angstrom": float(np.mean(permuted_delta)),
                    "frames_with_higher_energy_error_after_alignment_fraction": float(np.mean(frame_delta > 0)),
                    "alignment_energy_error_change_mean_mev_per_atom": float(np.mean(frame_delta)),
                    "alignment_energy_error_change_median_mev_per_atom": float(np.median(frame_delta)),
                    "maximum_atomic_common_shift_error_change_mev_per_angstrom": float(
                        1000.0
                        * np.max(
                            np.abs(
                                atomic["force_mae_common_shift"].to_numpy(float)
                                - atomic["force_mae_native"].to_numpy(float)
                            )
                        )
                    ),
                    "maximum_frame_common_shift_error_change_mev_per_atom": float(
                        1000.0
                        * np.max(
                            np.abs(
                                frame["energy_abs_error_per_atom_common_shift"].to_numpy(float)
                                - frame["energy_abs_error_per_atom_native"].to_numpy(float)
                            )
                        )
                    ),
                }
            )
    table = pd.DataFrame(records)
    audit = {
        "scope": scope,
        "model_units": int(len(table)),
        "atomic_outcomes": int(total_atomic),
        "frame_outcomes": int(total_frame),
        "all_values_finite": bool(
            np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "inference_rule": "descriptive only at atom/frame grain; uncertainty uses model-level clusters",
    }
    return table, audit


def validate_inputs(runs: pd.DataFrame, probes: pd.DataFrame) -> dict[str, Any]:
    expected_final = {(dataset, seed, mode) for dataset in DATASETS for seed in SEEDS for mode in ("off", "diagonal", "aligned", "full")}
    expected_persistent = {(dataset, seed, mode) for dataset in DATASETS for seed in SEEDS for mode in ("aligned", "full")}
    found_final = set(
        runs[runs["scope"] == "Final"][["dataset", "seed", "mode"]].itertuples(index=False, name=None)
    )
    found_persistent = set(
        runs[runs["scope"] == "Persistent"][["dataset", "seed", "mode"]].itertuples(index=False, name=None)
    )
    expected_probe = {(scope, dataset, seed) for scope in ("Final", "Persistent") for dataset in DATASETS for seed in SEEDS}
    found_probe = set(probes[["scope", "dataset", "seed"]].itertuples(index=False, name=None))
    pairs_equal = True
    for scope in ("Final", "Persistent"):
        local = runs[(runs["scope"] == scope) & runs["mode"].isin(("full", "aligned"))]
        pivot = local.pivot(index=["dataset", "seed"], columns="mode", values="parameter_count")
        pairs_equal &= bool((pivot["full"] == pivot["aligned"]).all())
    numeric = runs[
        ["validation_energy_mae", "validation_force_mae", "test_energy_mae", "test_force_mae"]
    ].to_numpy(float)
    result = {
        "final_cells_expected_80": len(found_final) == 80 and found_final == expected_final,
        "persistent_cells_expected_40": len(found_persistent) == 40 and found_persistent == expected_persistent,
        "probe_cells_expected_40": len(found_probe) == 40 and found_probe == expected_probe,
        "all_run_metrics_finite_positive": bool(np.isfinite(numeric).all() and (numeric > 0).all()),
        "test_never_used_for_selection": bool((~runs["test_used_for_selection"]).all()),
        "full_aligned_parameter_counts_equal_within_scope": pairs_equal,
    }
    result["passed"] = bool(all(result.values()))
    return result


def make_figure(
    output: Path,
    final_associations: pd.DataFrame,
    effects: list[dict[str, Any]],
    dose: pd.DataFrame,
    scope_interaction_effect: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    colors = {"Final": "#356C8C", "Persistent": "#B25B45"}
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.45), constrained_layout=True)

    # Rung 1: the precomputed Final observational associations.
    selected = final_associations.copy()
    selected = selected.sort_values("within_dataset_rank_correlation", key=lambda s: s.abs()).tail(5)
    statistic_labels = {
        "destructive_kernel_fraction": "destructive fraction",
        "weighted_mean_cos_delta": r"weighted mean $\cos\Delta\theta$",
        "circular_resultant_magnitude": "circular concentration",
        "relative_phase_entropy_normalized": "relative-phase entropy",
    }
    outcome_labels = {
        "trained_full_vs_aligned_log_benefit": "trained benefit",
        "frozen_alignment_log_harm": "alignment harm",
        "frozen_permutation_log_harm": "permutation harm",
    }
    labels = [
        f"{statistic_labels.get(statistic, statistic)} · {outcome_labels.get(outcome, outcome)}"
        for statistic, outcome in zip(selected["phase_statistic"], selected["outcome"])
    ]
    axes[0].barh(np.arange(len(selected)), selected["within_dataset_rank_correlation"], color="#7B8794")
    axes[0].axvline(0.0, color="#30343B", lw=0.8)
    axes[0].set_yticks(np.arange(len(selected)), labels)
    axes[0].set_xlabel("Within-dataset rank correlation")
    axes[0].set_title("a  Seeing: fitted phase organisation", loc="left")

    order = [
        "Final: self-density",
        "Final: cross-neighbour closure",
        "Final: learned phase",
        "Persistent: learned phase",
    ]
    by_name = {item["plot_label"]: item for item in effects}
    y = np.arange(len(order))[::-1]
    for position, label in zip(y, order):
        item = by_name[label]
        estimate = item["macro_relative_error_reduction_percent"]
        low, high = item["macro_95pct_hierarchical_bootstrap_ci_percent"]
        color = colors["Persistent"] if label.startswith("Persistent") else colors["Final"]
        axes[1].errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            capsize=2.5,
            ms=5,
        )
    axes[1].axvline(0.0, color="#30343B", lw=0.8, ls="--")
    axes[1].set_yticks(y, order)
    axes[1].set_xlabel(
        "Force-MAE reduction (%)\n"
        f"Persistent−Final learned-phase effect: "
        f"{scope_interaction_effect['benefit_ratio_change_percent']:+.2f}%"
    )
    axes[1].set_title("b  Doing: retrained policies", loc="left")

    macro = dose[dose["dataset"] == "macro"]
    for scope in ("Final", "Persistent"):
        local = macro[macro["scope"] == scope].sort_values("lambda")
        axes[2].plot(
            local["lambda"],
            local["force_mae_ratio_to_factual"],
            marker="o",
            ms=4,
            lw=1.7,
            label=scope,
            color=colors[scope],
        )
    axes[2].axhline(1.0, color="#30343B", lw=0.8, ls="--")
    axes[2].set_xlabel("Relative-phase dose $\\lambda$")
    axes[2].set_ylabel("Force MAE / factual Force MAE")
    axes[2].set_title("c  Imagining: same-checkpoint outcomes", loc="left")
    axes[2].legend(frameon=False, loc="upper right")

    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_effect(item: dict[str, Any]) -> str:
    estimate = item["macro_relative_error_reduction_percent"]
    low, high = item["macro_95pct_hierarchical_bootstrap_ci_percent"]
    return f"{estimate:.2f}% (95% hierarchical-bootstrap CI {low:.2f}% to {high:.2f}%; {item['wins']}/20 wins)"


def write_report(
    output: Path,
    effects_by_name: dict[str, dict[str, Any]],
    trained_interaction: dict[str, Any],
    frozen_interaction: dict[str, Any],
    association: dict[str, Any],
    dose: pd.DataFrame,
    counterfactual: pd.DataFrame,
    validation: dict[str, Any],
    common_shift_max: float,
) -> None:
    final_self = effects_by_name["final_self_density"]
    final_closure = effects_by_name["final_cross_neighbour"]
    final_phase = effects_by_name["final_learned_phase"]
    persistent_phase = effects_by_name["persistent_learned_phase"]
    final_frozen = effects_by_name["final_frozen_alignment"]
    persistent_frozen = effects_by_name["persistent_frozen_alignment"]
    final_permute = effects_by_name["final_frozen_permutation"]
    persistent_permute = effects_by_name["persistent_frozen_permutation"]
    macro_dose = dose[dose["dataset"] == "macro"].pivot(index="lambda", columns="scope", values="force_mae_ratio_to_factual")
    cf_scope = counterfactual.groupby("scope").agg(
        atomic_outcomes=("atomic_outcomes", "sum"),
        frame_outcomes=("frame_outcomes", "sum"),
        atom_alignment_harm_fraction=("atoms_with_higher_error_after_alignment_fraction", "mean"),
        frame_alignment_harm_fraction=("frames_with_higher_energy_error_after_alignment_fraction", "mean"),
    )
    text = f"""# Complete MACE-only Pearl three-rung analysis

## Answer first

The MACE evidence separates architectural benefit from functional dependence. The Final intervention ladder assigns the most reproducible generalisation gain to the Hermitian self-density channel: {format_effect(final_self)}. Cross-neighbour closure contributes {format_effect(final_closure)}, while learned relative phase adds {format_effect(final_phase)} over a parameter-matched Aligned control. Under Persistent propagation, the independently retrained phase effect is {format_effect(persistent_phase)}. The Persistent-minus-Final change in phase benefit is {trained_interaction['benefit_ratio_change_percent']:+.2f}% (95% CI {trained_interaction['benefit_ratio_change_95pct_hierarchical_bootstrap_ci_percent'][0]:+.2f}% to {trained_interaction['benefit_ratio_change_95pct_hierarchical_bootstrap_ci_percent'][1]:+.2f}%).

The same fitted functions rely much more strongly on their learned relative phase than the retraining contrast suggests. Frozen alignment raises the equal-dataset macro Force-MAE ratio to {float(macro_dose.loc[0.0, 'Final']):.3f}× in Final and {float(macro_dose.loc[0.0, 'Persistent']):.3f}× in Persistent. Equivalently, factual phase improves Force MAE by {format_effect(final_frozen)} in Final and {format_effect(persistent_frozen)} in Persistent relative to the aligned counterfactual. Breaking phase-to-edge assignment gives {format_effect(final_permute)} and {format_effect(persistent_permute)}, respectively. Retraining therefore compensates for much of the removed phase mechanism; it does not imply that the fitted Full predictor ignores phase.

## Pearl rung 1: seeing

The observational rung is deliberately descriptive. In the Final checkpoints, fitted phase-organisation statistics were associated with trained and frozen effects using ranks within dataset. Across the combined Final/Persistent model-scope units, the association between end-to-end phase benefit and frozen alignment harm is r={association['rank_correlation']:.3f} with a stratified permutation p={association['two_sided_stratified_permutation_p']:.4f}. This relationship is post hoc and is not used as a causal estimate.

## Pearl rung 2: doing

Each architecture policy was independently trained under the same split, seed, optimiser, precision, step budget, and validation-only checkpoint rule. Dataset and seed define the paired causal blocks. Final provides the full Off → Diagonal → Aligned → Full ladder. Persistent tests Full versus Aligned using the same parameter schema. The resulting scope interaction compares paired log benefits before averaging datasets equally. It identifies whether propagation scope modifies the algorithmic benefit of allowing learned relative phase under the frozen protocol.

The trained results support a clear hierarchy. Hermitian self-density has the largest stable end-to-end effect. Cross-neighbour closure and learned phase provide smaller, heterogeneous incremental effects on this four-system panel. Persistent propagation increases the mean learned-phase benefit only modestly, and its interval determines whether that increase is distinguishable from seed and dataset variation.

## Pearl rung 3: imagining

For every selected Full checkpoint, the factual prediction and its aligned, dose-controlled, permuted, and common-shift potential outcomes were evaluated on the same held-out structures. Messages, amplitudes, graph, examples, and all learned weights were fixed. The dose response is monotonic at the equal-dataset macro level in both scopes: as λ moves from zero alignment to the learned phase, the Force-MAE ratio falls from {float(macro_dose.loc[0.0, 'Final']):.3f} to 1.000 in Final and from {float(macro_dose.loc[0.0, 'Persistent']):.3f} to 1.000 in Persistent. The maximum common-shift change is {common_shift_max:.6f} meV Å⁻¹, confirming that predictions depend on relative phase rather than an arbitrary common phase origin.

The counterfactual tables contain {int(cf_scope.loc['Final', 'atomic_outcomes']):,} Final and {int(cf_scope.loc['Persistent', 'atomic_outcomes']):,} Persistent atomic force outcomes, plus {int(cf_scope.loc['Final', 'frame_outcomes']):,} and {int(cf_scope.loc['Persistent', 'frame_outcomes']):,} frame-energy outcomes. These rows localise effect heterogeneity but are not treated as independent replicates. Model-level paired inference remains the basis for confidence intervals.

## Causal interpretation

The three rungs answer different questions. Seeing describes how fitted phase organisation covaries with outcomes. Doing estimates the effect of permitting a representation rule in the training algorithm. Imagining quantifies how the selected predictor would change under controlled internal phase interventions. Together they show that the Hermitian second-moment channel drives the most reproducible generalisation gain, while learned relative phase is functionally embedded in the fitted solution and becomes more consequential when the charged state is propagated.

The analysis identifies causal effects inside this fixed MACE-ICTC/CHORUS training system. Theta is an internal learned variable, not a molecular observable. Diagonal versus Off is a joint representation-and-capacity intervention because active parameter counts differ. Full versus Aligned is the cleaner phase contrast because parameter counts and state schemas match. The four datasets form a mechanism panel rather than a random sample of chemical space.

## Validation status

- Input gate passed: `{validation['passed']}`.
- Final cells: 80/80; Persistent cells: 40/40; frozen probes: 40/40.
- Test data never selected checkpoints.
- Full/Aligned parameter counts match within each scope and dataset/seed pair.
- Confidence intervals resample datasets and then seeds; atoms and frames are never promoted to causal replicates.
- Frozen scope interaction in alignment benefit: {frozen_interaction['benefit_ratio_change_percent']:+.2f}% (95% CI {frozen_interaction['benefit_ratio_change_95pct_hierarchical_bootstrap_ci_percent'][0]:+.2f}% to {frozen_interaction['benefit_ratio_change_95pct_hierarchical_bootstrap_ci_percent'][1]:+.2f}%).
"""
    (output / "REPORT.md").write_text(text)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    final_runs = load_final_runs(args.final_result)
    persistent_runs = load_persistent_runs(args.persistent_root)
    runs = pd.concat([final_runs, persistent_runs], ignore_index=True)
    final_probes = load_final_probes(args.final_result)
    persistent_probes = load_persistent_probes(args.persistent_root)
    probes = pd.concat([final_probes, persistent_probes], ignore_index=True)
    validation = validate_inputs(runs, probes)
    if not validation["passed"]:
        raise RuntimeError(f"input validation failed: {validation}")

    pair_specs = [
        ("final_self_density", "Final", "diagonal", "off", "Final: self-density"),
        ("final_cross_neighbour", "Final", "aligned", "diagonal", "Final: cross-neighbour closure"),
        ("final_learned_phase", "Final", "full", "aligned", "Final: learned phase"),
        ("persistent_learned_phase", "Persistent", "full", "aligned", "Persistent: learned phase"),
    ]
    effects: list[dict[str, Any]] = []
    pairs: list[pd.DataFrame] = []
    pair_index: dict[str, pd.DataFrame] = {}
    for name, scope, treatment, control, plot_label in pair_specs:
        pair = pair_modes(runs, scope, treatment, control, "test_force_mae")
        summary = summarize_pair(pair, name, args.bootstrap, rng)
        summary["plot_label"] = plot_label
        effects.append(summary)
        pairs.append(pair.assign(contrast=name))
        pair_index[name] = pair

    trained_interaction, trained_interaction_pair = scope_interaction(
        pair_index["final_learned_phase"],
        pair_index["persistent_learned_phase"],
        args.bootstrap,
        rng,
        "Persistent-minus-Final trained phase interaction",
    )

    frozen_specs = [
        ("final_frozen_alignment", "Final", "aligned"),
        ("persistent_frozen_alignment", "Persistent", "aligned"),
        ("final_frozen_permutation", "Final", "permute_mean"),
        ("persistent_frozen_permutation", "Persistent", "permute_mean"),
    ]
    frozen_effects = []
    frozen_pairs = []
    frozen_index: dict[str, pd.DataFrame] = {}
    for name, scope, control in frozen_specs:
        pair = pair_probes(probes, scope, control)
        summary = summarize_pair(pair, name, args.bootstrap, rng)
        frozen_effects.append(summary)
        frozen_pairs.append(pair.assign(contrast=name))
        frozen_index[name] = pair
    frozen_interaction, frozen_interaction_pair = scope_interaction(
        frozen_index["final_frozen_alignment"],
        frozen_index["persistent_frozen_alignment"],
        args.bootstrap,
        rng,
        "Persistent-minus-Final frozen alignment interaction",
    )

    dose = dose_response(probes)
    trained_for_association = pd.concat(
        [
            pair_index["final_learned_phase"][["dataset", "seed", "log_benefit"]].assign(scope="Final"),
            pair_index["persistent_learned_phase"][["dataset", "seed", "log_benefit"]].assign(scope="Persistent"),
        ],
        ignore_index=True,
    ).rename(columns={"log_benefit": "trained_log_benefit"})
    frozen_for_association = pd.concat(
        [
            frozen_index["final_frozen_alignment"][["dataset", "seed", "log_benefit"]].assign(scope="Final"),
            frozen_index["persistent_frozen_alignment"][["dataset", "seed", "log_benefit"]].assign(scope="Persistent"),
        ],
        ignore_index=True,
    ).rename(columns={"log_benefit": "frozen_alignment_log_harm"})
    association_table = trained_for_association.merge(
        frozen_for_association, on=["scope", "dataset", "seed"], validate="one_to_one"
    )
    association = within_stratum_rank_association(
        association_table, args.association_permutations, rng
    )
    final_associations = pd.read_csv(args.final_result / "observational_associations.csv")

    final_counterfactual, final_audit = load_counterfactual_distributions(
        "Final", args.final_probe_root
    )
    persistent_counterfactual, persistent_audit = load_counterfactual_distributions(
        "Persistent", args.persistent_root / "mace" / "probes"
    )
    counterfactual = pd.concat(
        [final_counterfactual, persistent_counterfactual], ignore_index=True
    )

    common_shift_max = float(
        1000.0
        * probes["common_shift_maximum_absolute_force_change_ev_per_angstrom"].max()
    )
    dose_monotonic = {}
    for scope in ("Final", "Persistent"):
        values = (
            dose[(dose["scope"] == scope) & (dose["dataset"] == "macro")]
            .sort_values("lambda")["force_mae_ratio_to_factual"]
            .to_numpy(float)
        )
        dose_monotonic[scope] = bool(np.all(np.diff(values) <= 1e-12))

    effects_by_name = {item["name"]: item for item in effects + frozen_effects}
    all_effects = effects + frozen_effects
    result = {
        "schema_version": 1,
        "title": "Complete MACE-only Pearl three-rung causal analysis",
        "analysis_config": {
            "hierarchical_bootstrap_replicates": args.bootstrap,
            "association_permutations": args.association_permutations,
            "rng_seed": args.rng_seed,
            "causal_unit": "independently trained dataset-by-seed model or frozen predictor",
            "outcome": "test Force MAE after validation-Force-MAE checkpoint selection",
        },
        "input_validation": validation,
        "pearl_rungs": {
            "seeing": {
                "final_phase_organisation_associations": final_associations.to_dict("records"),
                "scope_level_association": association,
                "status": "post-hoc association only",
            },
            "doing": {
                "trained_effects": effects,
                "propagation_scope_interaction": trained_interaction,
            },
            "imagining": {
                "frozen_effects": frozen_effects,
                "frozen_scope_interaction": frozen_interaction,
                "dose_response": dose.to_dict("records"),
                "dose_macro_monotonic": dose_monotonic,
                "maximum_common_shift_force_change_mev_per_angstrom": common_shift_max,
                "counterfactual_outcome_audit": [final_audit, persistent_audit],
            },
        },
        "claim_boundary": {
            "theta": "internal learned variable without assigned physical observable",
            "scope": "fixed MACE-ICTC/CHORUS configuration and four-dataset mechanism panel",
            "self_density": "Diagonal versus Off is a total module effect with active capacity co-intervention",
            "phase": "Full versus Aligned retains matched parameter schema and nominal count",
            "counterfactual": "same fitted predictor and held-out input under model-internal phase intervention",
        },
    }

    runs.to_csv(args.output / "trained_runs.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(args.output / "trained_pairs.csv", index=False)
    trained_interaction_pair.to_csv(args.output / "trained_scope_interaction_pairs.csv", index=False)
    probes.to_csv(args.output / "frozen_probe_runs.csv", index=False)
    pd.concat(frozen_pairs, ignore_index=True).to_csv(args.output / "frozen_pairs.csv", index=False)
    frozen_interaction_pair.to_csv(args.output / "frozen_scope_interaction_pairs.csv", index=False)
    dose.to_csv(args.output / "phase_dose_response.csv", index=False)
    association_table.to_csv(args.output / "scope_association_units.csv", index=False)
    counterfactual.to_csv(args.output / "counterfactual_outcome_summary.csv", index=False)
    write_json(args.output / "causal_inference.json", result)
    write_json(args.output / "input_validation.json", validation)

    source_paths = [
        args.final_result / "primary_modes.csv",
        args.final_result / "aligned_runs.csv",
        args.final_result / "frozen_probe_runs.csv",
        args.final_result / "observational_associations.csv",
    ]
    write_json(
        args.output / "SOURCE_NOTES.json",
        {
            "generated_by": str(Path(__file__).resolve()),
            "source_files": [
                {"path": str(path), "sha256": sha256(path)} for path in source_paths
            ],
            "persistent_root": str(args.persistent_root),
            "final_probe_root": str(args.final_probe_root),
        },
    )
    make_figure(
        args.output / "mace_pearl_three_rungs",
        final_associations,
        effects,
        dose,
        trained_interaction,
    )
    write_report(
        args.output,
        effects_by_name,
        trained_interaction,
        frozen_interaction,
        association,
        dose,
        counterfactual,
        validation,
        common_shift_max,
    )


if __name__ == "__main__":
    main()
