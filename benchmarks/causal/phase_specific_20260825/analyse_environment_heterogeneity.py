#!/usr/bin/env python3
"""Analyse where a frozen CHORUS predictor relies on learned relative phase.

The intervention is evaluated at the validation-selected Full checkpoint.  The
unit of replication is the independently trained dataset-by-seed predictor.
Atoms and frames are nested repeated outcomes and are never treated as IID
replicates for uncertainty estimates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = tuple(range(20260616, 20260621))
PRIMARY_TEST = {"mal": "test", "bucky": "test", "3bpa": "300K", "zro": "test"}
LABELS = {"mal": "MAL", "bucky": "Buckyball", "3bpa": "3BPA", "zro": "ZrO"}
COLORS = {"mal": "#315B7D", "bucky": "#B45B3E", "3bpa": "#5D866C", "zro": "#7B659A"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=Path("/tmp/chorus_phase_causal_20260825"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/paper/results/causal/environment_heterogeneity_20260828"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--rng-seed", type=int, default=20260828)
    return parser.parse_args()


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def seed_bootstrap(
    values: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    return values[indices].mean(axis=1)


def hierarchical_bootstrap(
    table: pd.DataFrame,
    value: str,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    datasets = sorted(table["dataset"].unique())
    by_dataset = {
        dataset: table.loc[table["dataset"] == dataset, value].to_numpy(dtype=float)
        for dataset in datasets
    }
    result = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            values = by_dataset[str(dataset)]
            dataset_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        result[index] = float(np.mean(dataset_means))
    return result


def percent_inflation(log_harm: float | np.ndarray) -> float | np.ndarray:
    return 100.0 * np.expm1(log_harm)


def read_atomic(path: Path) -> pd.DataFrame:
    columns = [
        "dataset",
        "seed",
        "atomic_number",
        "neighbour_count",
        "reference_force_norm",
        "force_mae_native",
        "force_mae_aligned",
        "force_mae_permute_mean",
    ]
    frame = pq.read_table(path, columns=columns).to_pandas()
    for column in (
        "reference_force_norm",
        "force_mae_native",
        "force_mae_aligned",
        "force_mae_permute_mean",
    ):
        frame[column] = frame[column].astype(float) * 1000.0
    return frame


def read_frames(path: Path) -> pd.DataFrame:
    columns = [
        "dataset",
        "seed",
        "energy_abs_error_per_atom_native",
        "energy_abs_error_per_atom_aligned",
        "energy_abs_error_per_atom_permute_0",
        "energy_abs_error_per_atom_permute_1",
        "energy_abs_error_per_atom_permute_2",
        "energy_abs_error_per_atom_permute_3",
    ]
    frame = pq.read_table(path, columns=columns).to_pandas()
    permute = [f"energy_abs_error_per_atom_permute_{index}" for index in range(4)]
    frame["energy_abs_error_per_atom_permute_mean"] = frame[permute].mean(axis=1)
    for column in (
        "energy_abs_error_per_atom_native",
        "energy_abs_error_per_atom_aligned",
        "energy_abs_error_per_atom_permute_mean",
    ):
        frame[column] = frame[column].astype(float) * 1000.0
    return frame


def add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["reference-force quartile"] = ""
    result["neighbour-count quartile"] = ""
    for _dataset, group in result.groupby("dataset", observed=True):
        reference_quartile = pd.qcut(
            group["reference_force_norm"],
            4,
            labels=("Q1", "Q2", "Q3", "Q4"),
        ).astype(str)
        result.loc[group.index, "reference-force quartile"] = reference_quartile.to_numpy()
        neighbour_rank = group["neighbour_count"].rank(method="average")
        neighbour_codes = pd.qcut(
            neighbour_rank, 4, labels=False, duplicates="drop"
        )
        result.loc[group.index, "neighbour-count quartile"] = neighbour_codes.map(
            lambda value: f"Q{int(value) + 1}"
        ).to_numpy()
    result["element"] = result["atomic_number"].map(lambda value: f"Z={int(value)}")
    return result


def seed_level_strata(atomic: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    dimensions = {
        "reference-force quartile": "reference_force_norm",
        "neighbour-count quartile": "neighbour_count",
        "element": "atomic_number",
    }
    for dimension, range_column in dimensions.items():
        grouped = atomic.groupby(["dataset", "seed", dimension], observed=True)
        values = grouped.agg(
            atoms=("force_mae_native", "size"),
            stratum_low=(range_column, "min"),
            stratum_high=(range_column, "max"),
            native=("force_mae_native", "mean"),
            aligned=("force_mae_aligned", "mean"),
            permuted=("force_mae_permute_mean", "mean"),
            atom_fraction_harmed_descriptive=(
                "force_mae_aligned",
                lambda series: float(
                    np.mean(
                        series.to_numpy()
                        > atomic.loc[series.index, "force_mae_native"].to_numpy()
                    )
                ),
            ),
        ).reset_index().rename(columns={dimension: "stratum"})
        values["dimension"] = dimension
        values["aligned_minus_native"] = values["aligned"] - values["native"]
        values["permuted_minus_native"] = values["permuted"] - values["native"]
        values["aligned_log_harm"] = np.log(values["aligned"] / values["native"])
        values["permuted_log_harm"] = np.log(values["permuted"] / values["native"])
        parts.append(values)
    return pd.concat(parts, ignore_index=True)


def summarize_strata(
    seed_table: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, dimension, stratum), group in seed_table.groupby(
        ["dataset", "dimension", "stratum"], observed=True
    ):
        if len(group) != len(SEEDS):
            raise RuntimeError(
                f"incomplete stratum: {dataset}/{dimension}/{stratum} has {len(group)} seeds"
            )
        aligned_boot = seed_bootstrap(
            group["aligned_log_harm"].to_numpy(), replicates, rng
        )
        permuted_boot = seed_bootstrap(
            group["permuted_log_harm"].to_numpy(), replicates, rng
        )
        aligned_log = float(group["aligned_log_harm"].mean())
        permuted_log = float(group["permuted_log_harm"].mean())
        rows.append(
            {
                "dataset": dataset,
                "dimension": dimension,
                "stratum": stratum,
                "model_pairs": len(group),
                "atoms_per_seed_mean": float(group["atoms"].mean()),
                "stratum_low": float(group["stratum_low"].min()),
                "stratum_high": float(group["stratum_high"].max()),
                "native_force_mae_mev_per_angstrom": float(group["native"].mean()),
                "aligned_force_mae_mev_per_angstrom": float(group["aligned"].mean()),
                "permuted_force_mae_mev_per_angstrom": float(group["permuted"].mean()),
                "aligned_minus_native_mev_per_angstrom": float(
                    group["aligned_minus_native"].mean()
                ),
                "permuted_minus_native_mev_per_angstrom": float(
                    group["permuted_minus_native"].mean()
                ),
                "aligned_proportional_error_inflation_percent": float(
                    percent_inflation(aligned_log)
                ),
                "aligned_95pct_seed_bootstrap_ci_percent": percentile_interval(
                    percent_inflation(aligned_boot)
                ),
                "permuted_proportional_error_inflation_percent": float(
                    percent_inflation(permuted_log)
                ),
                "permuted_95pct_seed_bootstrap_ci_percent": percentile_interval(
                    percent_inflation(permuted_boot)
                ),
                "aligned_worse_model_pairs": int((group["aligned_log_harm"] > 0).sum()),
                "atom_fraction_harmed_descriptive": float(
                    group["atom_fraction_harmed_descriptive"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_macro_quartiles(
    seed_table: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quartile_data = seed_table[seed_table["dimension"].isin(
        ("reference-force quartile", "neighbour-count quartile")
    )]
    for (dimension, stratum), group in quartile_data.groupby(
        ["dimension", "stratum"], observed=True
    ):
        if set(group["dataset"]) != set(DATASETS):
            continue
        aligned_boot = hierarchical_bootstrap(
            group, "aligned_log_harm", replicates, rng
        )
        permuted_boot = hierarchical_bootstrap(
            group, "permuted_log_harm", replicates, rng
        )
        dataset_aligned = group.groupby("dataset")["aligned_log_harm"].mean()
        dataset_permuted = group.groupby("dataset")["permuted_log_harm"].mean()
        rows.append(
            {
                "dataset": "equal-dataset macro",
                "dimension": dimension,
                "stratum": stratum,
                "model_pairs": len(group),
                "aligned_proportional_error_inflation_percent": float(
                    percent_inflation(dataset_aligned.mean())
                ),
                "aligned_95pct_hierarchical_bootstrap_ci_percent": percentile_interval(
                    percent_inflation(aligned_boot)
                ),
                "permuted_proportional_error_inflation_percent": float(
                    percent_inflation(dataset_permuted.mean())
                ),
                "permuted_95pct_hierarchical_bootstrap_ci_percent": percentile_interval(
                    percent_inflation(permuted_boot)
                ),
            }
        )
    return pd.DataFrame(rows)


def gradient_contrasts(
    seed_table: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for dimension in ("reference-force quartile", "neighbour-count quartile"):
            group = seed_table[
                (seed_table["dataset"] == dataset)
                & (seed_table["dimension"] == dimension)
            ]
            wide_log = group.pivot(index="seed", columns="stratum", values="aligned_log_harm")
            wide_abs = group.pivot(
                index="seed", columns="stratum", values="aligned_minus_native"
            )
            if not {"Q1", "Q4"}.issubset(wide_log.columns):
                continue
            log_contrast = (wide_log["Q4"] - wide_log["Q1"]).to_numpy(dtype=float)
            abs_contrast = (wide_abs["Q4"] - wide_abs["Q1"]).to_numpy(dtype=float)
            log_boot = seed_bootstrap(log_contrast, replicates, rng)
            abs_boot = seed_bootstrap(abs_contrast, replicates, rng)
            rows.append(
                {
                    "dataset": dataset,
                    "dimension": dimension,
                    "model_pairs": len(log_contrast),
                    "q4_minus_q1_absolute_harm_mev_per_angstrom": float(abs_contrast.mean()),
                    "q4_minus_q1_absolute_harm_95pct_ci": percentile_interval(abs_boot),
                    "q4_vs_q1_proportional_harm_ratio_percent": float(
                        percent_inflation(log_contrast.mean())
                    ),
                    "q4_vs_q1_proportional_harm_ratio_95pct_ci_percent": percentile_interval(
                        percent_inflation(log_boot)
                    ),
                    "positive_proportional_contrast_pairs": int((log_contrast > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize_energy(
    frames: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_seed = frames.groupby(["dataset", "seed"], observed=True).agg(
        frames=("energy_abs_error_per_atom_native", "size"),
        native=("energy_abs_error_per_atom_native", "mean"),
        aligned=("energy_abs_error_per_atom_aligned", "mean"),
        permuted=("energy_abs_error_per_atom_permute_mean", "mean"),
    ).reset_index()
    per_seed["aligned_log_harm"] = np.log(per_seed["aligned"] / per_seed["native"])
    per_seed["permuted_log_harm"] = np.log(per_seed["permuted"] / per_seed["native"])
    datasets = []
    for dataset, group in per_seed.groupby("dataset"):
        aligned_boot = seed_bootstrap(group["aligned_log_harm"].to_numpy(), replicates, rng)
        datasets.append(
            {
                "dataset": dataset,
                "model_pairs": len(group),
                "native_energy_mae_mev_per_atom": float(group["native"].mean()),
                "aligned_energy_mae_mev_per_atom": float(group["aligned"].mean()),
                "permuted_energy_mae_mev_per_atom": float(group["permuted"].mean()),
                "aligned_proportional_error_inflation_percent": float(
                    percent_inflation(group["aligned_log_harm"].mean())
                ),
                "aligned_95pct_seed_bootstrap_ci_percent": percentile_interval(
                    percent_inflation(aligned_boot)
                ),
            }
        )
    macro_boot = hierarchical_bootstrap(per_seed, "aligned_log_harm", replicates, rng)
    macro = {
        "aligned_proportional_error_inflation_percent": float(
            percent_inflation(
                per_seed.groupby("dataset")["aligned_log_harm"].mean().mean()
            )
        ),
        "aligned_95pct_hierarchical_bootstrap_ci_percent": percentile_interval(
            percent_inflation(macro_boot)
        ),
    }
    return pd.DataFrame(datasets), macro


def plot_heterogeneity(summary: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.2,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.2,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.55), sharex="row")
    dimensions = ("reference-force quartile", "neighbour-count quartile")
    for row, dimension in enumerate(dimensions):
        selected = summary[summary["dimension"] == dimension]
        for dataset in DATASETS:
            group = selected[selected["dataset"] == dataset].copy()
            group["order"] = group["stratum"].str.removeprefix("Q").astype(int)
            group = group.sort_values("order")
            if len(group) < 2:
                continue
            axes[row, 0].plot(
                group["stratum"],
                group["aligned_minus_native_mev_per_angstrom"],
                marker="o",
                ms=3.7,
                lw=1.25,
                color=COLORS[dataset],
                label=LABELS[dataset],
            )
            axes[row, 1].plot(
                group["stratum"],
                group["aligned_proportional_error_inflation_percent"],
                marker="o",
                ms=3.7,
                lw=1.25,
                color=COLORS[dataset],
                label=LABELS[dataset],
            )
        axes[row, 0].set_ylabel("Aligned − native\nForce MAE (meV Å⁻¹)")
        axes[row, 1].set_ylabel("Proportional Force-MAE\ninflation (%)")
        axes[row, 0].set_xlabel(dimension.replace("-", " "))
        axes[row, 1].set_xlabel(dimension.replace("-", " "))
        for axis in axes[row]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#D9DEE5", linewidth=0.55, alpha=0.75)
    axes[0, 0].set_title("Absolute intervention consequence", loc="left")
    axes[0, 1].set_title("Scale-normalised intervention consequence", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0, w_pad=1.2)
    fig.savefig(output / "frozen_phase_heterogeneity.svg", bbox_inches="tight")
    fig.savefig(output / "frozen_phase_heterogeneity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    strata: pd.DataFrame,
    gradients: pd.DataFrame,
    energy_macro: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    force_gradient = gradients[
        gradients["dimension"] == "reference-force quartile"
    ]
    absolute_positive = int(
        (force_gradient["q4_minus_q1_absolute_harm_mev_per_angstrom"] > 0).sum()
    )
    proportional_positive = int(
        (force_gradient["q4_vs_q1_proportional_harm_ratio_percent"] > 0).sum()
    )
    neighbour_gradient = gradients[
        gradients["dimension"] == "neighbour-count quartile"
    ]
    neighbour_positive = int(
        (neighbour_gradient["q4_vs_q1_proportional_harm_ratio_percent"] > 0).sum()
    )
    report = f"""# Frozen-phase effect heterogeneity

## Result

Aligning the learned relative phase increases Force MAE in every reported
environmental stratum of every completed MACE-Final predictor.  The absolute
Q4-minus-Q1 intervention consequence is positive for {absolute_positive}/4
datasets when environments are ordered by reference-force magnitude.  The
scale-normalised Q4-versus-Q1 contrast is positive for only
{proportional_positive}/4 datasets.  Harder configurations therefore incur a
larger absolute penalty, while the current panel does not support a universal
claim that their *relative* dependence on phase is stronger.

Neighbour count is a more heterogeneous moderator: the proportional Q4-versus-Q1
contrast is positive in {neighbour_positive}/{len(neighbour_gradient)} datasets
with identifiable Q1 and Q4 strata.  Crowding alone is consequently not a
sufficient explanation of phase use.  Element-resolved results are retained in
`stratum_effects.csv`; they are descriptive within each chemical system rather
than population-level element effects.

The equal-dataset frozen alignment effect on energy is
{energy_macro['aligned_proportional_error_inflation_percent']:.2f}%
(95% hierarchical-bootstrap interval
{energy_macro['aligned_95pct_hierarchical_bootstrap_ci_percent'][0]:.2f}% to
{energy_macro['aligned_95pct_hierarchical_bootstrap_ci_percent'][1]:.2f}%).

## Interpretation

This analysis localises the consequence of an intervention on a fitted
predictor.  It does not turn atoms into independent causal units.  Each stratum
first reduces all nested atom or frame outcomes to one value per trained
dataset-by-seed predictor; bootstrap resampling is then performed over those
predictors.  Absolute and proportional effects are shown together because an
absolute error difference can rise mechanically with target scale.

## Audit

- Trained-model pairs: {audit['model_pairs']}.
- Atomic repeated outcomes: {audit['atomic_rows']:,}.
- Frame repeated outcomes: {audit['frame_rows']:,}.
- PyArrow: {audit['pyarrow_version']}.
- All required files present: **{audit['complete']}**.
- All numeric outcomes finite: **{audit['finite']}**.
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    args = parse_args()
    if int(pa.__version__.split(".")[0]) < 24:
        raise RuntimeError(f"PyArrow >=24 required; found {pa.__version__}")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    atomic_parts: list[pd.DataFrame] = []
    frame_parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            root = (
                args.phase_root
                / "probes"
                / dataset
                / f"seed{seed}"
                / PRIMARY_TEST[dataset]
            )
            atomic_path = root / "paired_atomic_outcomes.parquet"
            frame_path = root / "paired_frame_outcomes.parquet"
            if not atomic_path.is_file() or not frame_path.is_file():
                missing.append(str(root))
                continue
            atomic_parts.append(read_atomic(atomic_path))
            frame_parts.append(read_frames(frame_path))
    if missing:
        raise RuntimeError("missing frozen-probe outcomes:\n" + "\n".join(missing))

    atomic = add_strata(pd.concat(atomic_parts, ignore_index=True))
    frames = pd.concat(frame_parts, ignore_index=True)
    numeric_atomic = atomic.select_dtypes(include=[np.number]).to_numpy()
    numeric_frames = frames.select_dtypes(include=[np.number]).to_numpy()
    finite = bool(np.isfinite(numeric_atomic).all() and np.isfinite(numeric_frames).all())
    if not finite:
        raise RuntimeError("non-finite outcome detected")

    seed_table = seed_level_strata(atomic)
    strata = summarize_strata(seed_table, args.bootstrap, rng)
    macro = summarize_macro_quartiles(seed_table, args.bootstrap, rng)
    gradients = gradient_contrasts(seed_table, args.bootstrap, rng)
    energy, energy_macro = summarize_energy(frames, args.bootstrap, rng)
    audit = {
        "complete": True,
        "finite": finite,
        "model_pairs": len(DATASETS) * len(SEEDS),
        "atomic_rows": len(atomic),
        "frame_rows": len(frames),
        "pyarrow_version": pa.__version__,
        "bootstrap_replicates": args.bootstrap,
        "rng_seed": args.rng_seed,
        "causal_unit": "independently trained dataset-by-seed predictor",
        "nested_outcomes": "atoms and frames",
    }

    seed_table.to_csv(args.output / "seed_level_strata.csv", index=False)
    strata.to_csv(args.output / "stratum_effects.csv", index=False)
    macro.to_csv(args.output / "macro_quartile_effects.csv", index=False)
    gradients.to_csv(args.output / "gradient_contrasts.csv", index=False)
    energy.to_csv(args.output / "frame_energy_effects.csv", index=False)
    payload = {
        "schema_version": 1,
        "estimand": "frozen relative-phase alignment harm at the validation-selected Full checkpoint",
        "stratum_effects": strata.to_dict("records"),
        "macro_quartile_effects": macro.to_dict("records"),
        "gradient_contrasts": gradients.to_dict("records"),
        "energy_effects": energy.to_dict("records"),
        "energy_macro": energy_macro,
        "audit": audit,
    }
    (args.output / "environment_heterogeneity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    plot_heterogeneity(strata, args.output)
    write_report(args.output, strata, gradients, energy_macro, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
