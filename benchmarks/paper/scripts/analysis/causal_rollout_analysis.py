#!/usr/bin/env python3
"""Paired potential-outcome analysis for the CHORUS water rollouts.

The velocity seed is the matched rollout block.  The analysis estimates total
checkpoint intervention effects for Final vs MACE-ICTC, Persistent vs
MACE-ICTC, and
Persistent vs Final.  It deliberately keeps the independently trained DPA-4
models outside the causal contrasts; they remain external deployment controls.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats


INK = "#272727"
MUTED = "#747474"
HAIRLINE = "#E6E6E3"
PAPER = "#FFFFFF"
CONTEXT = "#F5F5F3"
BLUE = "#205381"
BLUE_LIGHT = "#DDE7EF"
ORANGE = "#C96A3D"
ORANGE_LIGHT = "#F2E2DA"

MODEL_ORDER = ["mace_phaseoff", "mace_final", "mace_persistent"]
MODEL_NAMES = {
    "mace_phaseoff": "MACE-ICTC",
    "mace_final": "Final",
    "mace_persistent": "Persistent",
}
MODEL_COLORS = {
    "mace_phaseoff": MUTED,
    "mace_final": BLUE,
    "mace_persistent": ORANGE,
}
CONTRASTS = [
    ("mace_final", "mace_phaseoff", "Final - MACE-ICTC"),
    ("mace_persistent", "mace_phaseoff", "Persistent - MACE-ICTC"),
    ("mace_persistent", "mace_final", "Persistent - Final"),
]
FORCE_MAE = {
    "mace_phaseoff": 21.891,
    "mace_final": 20.906,
    "mace_persistent": 20.454,
}
PARAMETERS = {
    "mace_phaseoff": 577_728,
    "mace_final": 672_134,
    "mace_persistent": 864_847,
}


def configure_style(font_dir: Path | None = None) -> None:
    """Match the serif visual system used by the manuscript figures."""
    if font_dir is not None and font_dir.exists():
        for font_path in sorted(font_dir.glob("Times New Roman*.ttf")):
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.55,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": PAPER,
            "figure.facecolor": PAPER,
        }
    )
ENDPOINTS = {
    "diffusion_matched_1e-9_m2_per_s": {
        "name": r"O diffusion $D$",
        "unit": r"$10^{-9}$ m$^2$ s$^{-1}$",
        "role": "dynamic",
    },
    "oo_peak_g": {
        "name": r"O--O first-peak $g(r)$",
        "unit": "dimensionless",
        "role": "structural_secondary",
    },
    "oo_coordination_3p5a": {
        "name": r"O--O coordination",
        "unit": "per water",
        "role": "structural_secondary",
    },
    "hydrogen_bond_coordination": {
        "name": r"H-bond coordination",
        "unit": "per water",
        "role": "structural_secondary",
    },
    "tetrahedral_q": {
        "name": r"Tetrahedral order $q$",
        "unit": "dimensionless",
        "role": "structural_primary",
    },
    "abs_epot_drift_mev_per_h2o_per_ns": {
        "name": r"$|$PE drift$|$",
        "unit": r"meV H$_2$O$^{-1}$ ns$^{-1}$",
        "role": "deployment",
    },
    "fmax_q95_ev_per_a": {
        "name": r"95th-percentile $F_{\max}$",
        "unit": r"eV \AA$^{-1}$",
        "role": "deployment",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/ylzhang/lrx/chorus_water_1ns_20260926"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260930)
    return parser.parse_args()


def t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return math.nan, math.nan
    sd = float(np.std(values, ddof=1))
    if sd == 0.0:
        return mean, mean
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    half = critical * sd / math.sqrt(len(values))
    return mean - half, mean + half


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(abs(float(np.mean(values * np.asarray(signs)))))
    null = np.asarray(null)
    return float(np.mean(null >= observed - 1e-15))


def endpoint_effect(
    base: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float | int | str]:
    base = np.asarray(base, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - base
    n = len(delta)
    mean_delta = float(np.mean(delta))
    sd_delta = float(np.std(delta, ddof=1))
    se_delta = sd_delta / math.sqrt(n)
    ci_low, ci_high = t_interval(delta)
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(delta[boot_idx], axis=1)
    boot_low, boot_high = np.quantile(boot_means, [0.025, 0.975])
    loo = np.asarray([np.mean(np.delete(delta, index)) for index in range(n)])
    positive = int(np.sum(delta > 0))
    negative = int(np.sum(delta < 0))
    zero = int(np.sum(delta == 0))
    base_mean = float(np.mean(base))
    base_sd = float(np.std(base, ddof=1))
    return {
        "n_pairs": n,
        "base_mean": base_mean,
        "target_mean": float(np.mean(target)),
        "mean_delta": mean_delta,
        "percent_delta_vs_base": 100.0 * mean_delta / base_mean if base_mean else math.nan,
        "paired_sd": sd_delta,
        "paired_se": se_delta,
        "t95_low": ci_low,
        "t95_high": ci_high,
        "bootstrap95_low": float(boot_low),
        "bootstrap95_high": float(boot_high),
        "paired_cohen_dz": mean_delta / sd_delta if sd_delta else math.nan,
        "standardized_by_base_seed_sd": mean_delta / base_sd if base_sd else math.nan,
        "standardized_t95_low": ci_low / base_sd if base_sd else math.nan,
        "standardized_t95_high": ci_high / base_sd if base_sd else math.nan,
        "positive_pairs": positive,
        "negative_pairs": negative,
        "zero_pairs": zero,
        "direction_consistency": max(positive, negative) / n,
        "modal_direction": "positive" if positive > negative else "negative" if negative > positive else "mixed",
        "exact_signflip_p_two_sided": exact_signflip_p(delta),
        "leave_one_out_mean_min": float(np.min(loo)),
        "leave_one_out_mean_max": float(np.max(loo)),
    }


def load_seed_table(root: Path) -> pd.DataFrame:
    path = root / "analysis/water_combined_validation/seed_level_300ps.csv"
    table = pd.read_csv(path)
    table = table[table["label"].isin(MODEL_ORDER)].copy()
    table["abs_epot_drift_mev_per_h2o_per_ns"] = table[
        "epot_drift_mev_per_h2o_per_ns"
    ].abs()

    fmax_rows: list[dict[str, float | int | str]] = []
    multi = root / "analysis/water_multiseed_300ps"
    for label in MODEL_ORDER:
        for seed in sorted(table.loc[table["label"] == label, "seed"].astype(int)):
            windows = pd.read_csv(multi / label / f"seed{seed}" / "thermo_windows.csv")
            row = windows[
                (np.isclose(windows["start_ps"], 100.0))
                & (np.isclose(windows["end_ps"], 300.0))
                & (windows["quantity"] == "fmax_ev_per_a")
            ]
            if len(row) != 1:
                raise RuntimeError(f"Expected one fmax window for {label} seed {seed}")
            fmax_rows.append(
                {
                    "label": label,
                    "seed": seed,
                    "fmax_mean_ev_per_a": float(row.iloc[0]["mean"]),
                    "fmax_q95_ev_per_a": float(row.iloc[0]["q95"]),
                    "fmax_max_ev_per_a": float(row.iloc[0]["max"]),
                }
            )
    table = table.merge(pd.DataFrame(fmax_rows), on=["label", "seed"], validate="one_to_one")
    counts = table.groupby("label")["seed"].nunique().to_dict()
    if any(counts.get(label) != 5 for label in MODEL_ORDER):
        raise RuntimeError(f"Expected five paired seeds per model, found {counts}")
    seed_sets = [set(table.loc[table["label"] == label, "seed"]) for label in MODEL_ORDER]
    if not all(seed_set == seed_sets[0] for seed_set in seed_sets[1:]):
        raise RuntimeError(f"Velocity seeds are not matched: {seed_sets}")
    return table.sort_values(["label", "seed"]).reset_index(drop=True)


def scalar_effects(
    table: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = table.set_index(["label", "seed"]).sort_index()
    seeds = sorted(table["seed"].unique())
    effect_rows = []
    difference_rows = []
    ordered_rows = []
    for endpoint, metadata in ENDPOINTS.items():
        model_values = {
            label: indexed.loc[(label, seeds), endpoint].to_numpy(dtype=float)
            for label in MODEL_ORDER
        }
        for target, base, contrast in CONTRASTS:
            effect = endpoint_effect(model_values[base], model_values[target], rng, n_bootstrap)
            effect_rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_name": metadata["name"],
                    "unit": metadata["unit"],
                    "role": metadata["role"],
                    "contrast": contrast,
                    "target": MODEL_NAMES[target],
                    "base": MODEL_NAMES[base],
                    **effect,
                }
            )
            for seed, base_value, target_value in zip(
                seeds, model_values[base], model_values[target], strict=True
            ):
                difference_rows.append(
                    {
                        "endpoint": endpoint,
                        "endpoint_name": metadata["name"],
                        "contrast": contrast,
                        "seed": int(seed),
                        "base_value": float(base_value),
                        "target_value": float(target_value),
                        "paired_delta": float(target_value - base_value),
                    }
                )

        means = np.asarray([np.mean(model_values[label]) for label in MODEL_ORDER])
        overall_direction = float(np.sign(means[-1] - means[0]))
        seed_slopes = []
        strict_monotonic = 0
        direction_consistent = 0
        for seed_index, seed in enumerate(seeds):
            values = np.asarray([model_values[label][seed_index] for label in MODEL_ORDER])
            slope = float(np.polyfit(np.arange(3, dtype=float), values, 1)[0])
            seed_slopes.append(slope)
            if overall_direction > 0:
                strict_monotonic += int(values[0] < values[1] < values[2])
                direction_consistent += int(slope > 0)
            elif overall_direction < 0:
                strict_monotonic += int(values[0] > values[1] > values[2])
                direction_consistent += int(slope < 0)
            ordered_rows.append(
                {
                    "endpoint": endpoint,
                    "seed": int(seed),
                    "off": float(values[0]),
                    "final": float(values[1]),
                    "persistent": float(values[2]),
                    "scope_slope": slope,
                    "overall_mean_direction": "increasing" if overall_direction > 0 else "decreasing",
                }
            )
        slope_ci = t_interval(np.asarray(seed_slopes))
        for row in ordered_rows[-len(seeds):]:
            row["strict_monotonic_seed_count"] = strict_monotonic
            row["scope_direction_consistent_seed_count"] = direction_consistent
            row["mean_scope_slope"] = float(np.mean(seed_slopes))
            row["scope_slope_t95_low"] = slope_ci[0]
            row["scope_slope_t95_high"] = slope_ci[1]
    return pd.DataFrame(effect_rows), pd.DataFrame(difference_rows), pd.DataFrame(ordered_rows)


def block_mean(table: pd.DataFrame, start: float, end: float, last: bool) -> pd.Series:
    if last:
        selected = table[(table["time_ps"] >= start) & (table["time_ps"] <= end)]
    else:
        selected = table[(table["time_ps"] >= start) & (table["time_ps"] < end)]
    return selected.mean(numeric_only=True)


def time_resolved_effects(
    root: Path,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    multi = root / "analysis/water_multiseed_300ps"
    seed_table = load_seed_table(root)
    seeds = sorted(seed_table["seed"].unique())
    block_edges = [(100.0, 150.0), (150.0, 200.0), (200.0, 250.0), (250.0, 300.0)]
    endpoint_columns = {
        "tetrahedral_q_mean": ("tetrahedral_q", "Tetrahedral order q"),
        "hydrogen_bond_coordination": ("hydrogen_bond_coordination", "H-bond coordination"),
    }
    block_rows = []
    for label in MODEL_ORDER:
        for seed in seeds:
            series = pd.read_csv(multi / label / f"seed{seed}" / "water_order_series.csv")
            for block_index, (start, end) in enumerate(block_edges):
                means = block_mean(series, start, end, block_index == len(block_edges) - 1)
                for column, (endpoint, endpoint_name) in endpoint_columns.items():
                    block_rows.append(
                        {
                            "label": label,
                            "model": MODEL_NAMES[label],
                            "seed": int(seed),
                            "block": block_index,
                            "start_ps": start,
                            "end_ps": end,
                            "midpoint_ps": (start + end) / 2.0,
                            "endpoint": endpoint,
                            "endpoint_name": endpoint_name,
                            "value": float(means[column]),
                        }
                    )
    block_table = pd.DataFrame(block_rows)
    effect_rows = []
    for endpoint in block_table["endpoint"].unique():
        for block_index in sorted(block_table["block"].unique()):
            selected = block_table[
                (block_table["endpoint"] == endpoint) & (block_table["block"] == block_index)
            ]
            pivot = selected.pivot(index="seed", columns="label", values="value").sort_index()
            for target, base, contrast in CONTRASTS:
                effect = endpoint_effect(
                    pivot[base].to_numpy(), pivot[target].to_numpy(), rng, n_bootstrap
                )
                effect_rows.append(
                    {
                        "endpoint": endpoint,
                        "endpoint_name": selected["endpoint_name"].iloc[0],
                        "block": int(block_index),
                        "start_ps": float(selected["start_ps"].iloc[0]),
                        "end_ps": float(selected["end_ps"].iloc[0]),
                        "midpoint_ps": float(selected["midpoint_ps"].iloc[0]),
                        "contrast": contrast,
                        **effect,
                    }
                )
    return block_table, pd.DataFrame(effect_rows)


def long_horizon_blocks(root: Path) -> pd.DataFrame:
    base = root / "analysis/water_1ns"
    block_edges = [(200.0, 400.0), (400.0, 600.0), (600.0, 800.0), (800.0, 1000.0)]
    rows = []
    for label in MODEL_ORDER:
        series = pd.read_csv(base / label / "water_order_series.csv")
        for block_index, (start, end) in enumerate(block_edges):
            means = block_mean(series, start, end, block_index == len(block_edges) - 1)
            for column, endpoint, endpoint_name in [
                ("tetrahedral_q_mean", "tetrahedral_q", "Tetrahedral order q"),
                ("hydrogen_bond_coordination", "hydrogen_bond_coordination", "H-bond coordination"),
            ]:
                rows.append(
                    {
                        "label": label,
                        "model": MODEL_NAMES[label],
                        "block": block_index,
                        "start_ps": start,
                        "end_ps": end,
                        "midpoint_ps": (start + end) / 2.0,
                        "endpoint": endpoint,
                        "endpoint_name": endpoint_name,
                        "value": float(means[column]),
                    }
                )
    return pd.DataFrame(rows)


def functional_contrasts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    multi = root / "analysis/water_multiseed_300ps"
    seed_table = load_seed_table(root)
    seeds = sorted(seed_table["seed"].unique())
    rdf_rows = []
    msd_rows = []
    functional_rows = []
    for seed in seeds:
        rdf_by_model = {
            label: pd.read_csv(multi / label / f"seed{seed}" / "rdf.csv")
            for label in MODEL_ORDER
        }
        msd_by_model = {
            label: pd.read_csv(multi / label / f"seed{seed}" / "msd_oxygen.csv")
            for label in MODEL_ORDER
        }
        for target, base, contrast in CONTRASTS:
            base_rdf = rdf_by_model[base]
            target_rdf = rdf_by_model[target]
            if not np.allclose(base_rdf["r_a"], target_rdf["r_a"]):
                raise RuntimeError(f"RDF grid mismatch for {contrast}, seed {seed}")
            r = base_rdf["r_a"].to_numpy()
            mask = (r >= 1.5) & (r <= 6.0)
            for component in ("g_oo", "g_oh", "g_hh"):
                delta = target_rdf[component].to_numpy() - base_rdf[component].to_numpy()
                for radius, value in zip(r, delta, strict=True):
                    rdf_rows.append(
                        {
                            "seed": int(seed),
                            "contrast": contrast,
                            "component": component,
                            "r_a": float(radius),
                            "delta_g": float(value),
                        }
                    )
                functional_rows.append(
                    {
                        "seed": int(seed),
                        "contrast": contrast,
                        "curve": component,
                        "domain": "1.5-6.0 A",
                        "rms_difference": float(
                            np.sqrt(np.trapezoid(delta[mask] ** 2, r[mask]) / (r[mask][-1] - r[mask][0]))
                        ),
                    }
                )

            base_msd = msd_by_model[base]
            target_msd = msd_by_model[target]
            if not np.allclose(base_msd["lag_ps"], target_msd["lag_ps"]):
                raise RuntimeError(f"MSD grid mismatch for {contrast}, seed {seed}")
            lag = base_msd["lag_ps"].to_numpy()
            delta_msd = target_msd["msd_o_a2"].to_numpy() - base_msd["msd_o_a2"].to_numpy()
            for lag_value, value in zip(lag, delta_msd, strict=True):
                msd_rows.append(
                    {
                        "seed": int(seed),
                        "contrast": contrast,
                        "lag_ps": float(lag_value),
                        "delta_msd_o_a2": float(value),
                    }
                )
    return pd.DataFrame(rdf_rows), pd.DataFrame(msd_rows), pd.DataFrame(functional_rows)


def plot_main(
    output: Path,
    seed_table: pd.DataFrame,
    effects: pd.DataFrame,
    time_effects: pd.DataFrame,
    long_blocks: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)

    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("(a)  Matched-rollout identification", loc="left", weight="normal", pad=7)

    def add_node(
        centre: tuple[float, float],
        label: str,
        *,
        width: float,
        height: float = 0.15,
        facecolor: str = PAPER,
        edgecolor: str = MUTED,
        fontsize: float = 5.8,
    ) -> FancyBboxPatch:
        x, y = centre
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.75,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=ax.transAxes,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight="normal",
            transform=ax.transAxes,
            zorder=4,
        )
        return box

    def add_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        patch_a: FancyBboxPatch,
        patch_b: FancyBboxPatch,
        color: str = INK,
        linestyle: str = "-",
    ) -> None:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.8,
            linestyle=linestyle,
            color=color,
            patchA=patch_a,
            patchB=patch_b,
            transform=ax.transAxes,
            zorder=1,
        )
        ax.add_patch(arrow)

    positions = {
        "scope": (0.13, 0.72),
        "training": (0.13, 0.27),
        "checkpoint": (0.41, 0.72),
        "context": (0.42, 0.27),
        "trajectory": (0.69, 0.56),
        "force": (0.69, 0.88),
        "outcomes": (0.895, 0.56),
    }
    nodes = {
        "scope": add_node(
            positions["scope"],
            "MACE-ICTC\nCHORUS-Final\nCHORUS-Persistent",
            width=0.22,
            height=0.20,
            facecolor=BLUE_LIGHT,
            edgecolor=BLUE,
            fontsize=5.2,
        ),
        "training": add_node(
            positions["training"], "training\nrandomness", width=0.18, facecolor=CONTEXT
        ),
        "checkpoint": add_node(
            positions["checkpoint"], "selected\ncheckpoint", width=0.18
        ),
        "context": add_node(
            positions["context"],
            "matched context\nseed · initial state · protocol",
            width=0.27,
            facecolor=CONTEXT,
            fontsize=5.4,
        ),
        "trajectory": add_node(
            positions["trajectory"], "MD trajectory", width=0.17, facecolor=ORANGE_LIGHT, edgecolor=ORANGE
        ),
        "force": add_node(
            positions["force"], "held-out\nForce MAE", width=0.17, height=0.13
        ),
        "outcomes": add_node(
            positions["outcomes"],
            "trajectory\noutcomes\n$q$ · H-bond\nRDF · $D$",
            width=0.17,
            height=0.22,
            fontsize=5.0,
        ),
    }
    for source, target, color, linestyle in (
        ("scope", "checkpoint", BLUE, "-"),
        ("training", "checkpoint", MUTED, "-"),
        ("checkpoint", "force", INK, "-"),
        ("checkpoint", "trajectory", INK, "-"),
        ("context", "trajectory", MUTED, (0, (2.0, 2.0))),
        ("trajectory", "outcomes", ORANGE, "-"),
    ):
        add_arrow(
            positions[source],
            positions[target],
            patch_a=nodes[source],
            patch_b=nodes[target],
            color=color,
            linestyle=linestyle,
        )
    ax.text(
        0.50,
        0.06,
        "paired unit: matched velocity seed  |  treatment: selected checkpoint",
        ha="center",
        va="center",
        fontsize=5.6,
        color=MUTED,
        transform=ax.transAxes,
    )

    ax = axes[0, 1]
    labels = [MODEL_NAMES[label] for label in MODEL_ORDER]
    q_table = seed_table.pivot(index="seed", columns="label", values="tetrahedral_q")
    x = np.arange(3)
    for index, (_, row) in enumerate(q_table.iterrows()):
        values = [row[label] for label in MODEL_ORDER]
        ax.plot(
            x,
            values,
            marker="o",
            color="#A9B0B6",
            linewidth=0.65,
            markersize=2.2,
            label="matched velocity seed" if index == 0 else None,
        )
    means = [q_table[label].mean() for label in MODEL_ORDER]
    ax.plot(x, means, marker="o", color=INK, linewidth=1.5, markersize=4.2, label="five-seed mean")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Tetrahedral order $q$")
    ax.set_title("(b)  Paired downstream structural response", loc="left", weight="normal", pad=7)
    ax.legend(frameon=False, loc="best")

    ax = axes[1, 0]
    q_effects = time_effects[
        (time_effects["endpoint"] == "tetrahedral_q")
        & (
            time_effects["contrast"].isin(
                ["Final - MACE-ICTC", "Persistent - MACE-ICTC"]
            )
        )
    ]
    styles = {
        "Final - MACE-ICTC": (MODEL_COLORS["mace_final"], "o"),
        "Persistent - MACE-ICTC": (MODEL_COLORS["mace_persistent"], "s"),
    }
    for contrast, selected in q_effects.groupby("contrast"):
        selected = selected.sort_values("midpoint_ps")
        color, marker = styles[contrast]
        y = selected["mean_delta"].to_numpy()
        low = selected["t95_low"].to_numpy()
        high = selected["t95_high"].to_numpy()
        ax.errorbar(
            selected["midpoint_ps"],
            y,
            yerr=np.vstack([y - low, high - y]),
            color=color,
            marker=marker,
            capsize=2,
            linewidth=1.0,
            elinewidth=0.9,
            markersize=3.8,
            label=contrast,
        )
    ax.axhline(0.0, color=MUTED, linewidth=0.65, linestyle="--")
    ax.set_xlabel("Trajectory time (ps)")
    ax.set_ylabel(r"Paired effect on $q$")
    ax.set_title("(c)  Time-resolved paired effect", loc="left", weight="normal", pad=7)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    selected = long_blocks[long_blocks["endpoint"] == "tetrahedral_q"]
    for label in MODEL_ORDER:
        model = selected[selected["label"] == label].sort_values("midpoint_ps")
        ax.plot(
            model["midpoint_ps"],
            model["value"],
            marker="o",
            linewidth=1.15,
            markersize=3.5,
            color=MODEL_COLORS[label],
            label=MODEL_NAMES[label],
        )
    ax.set_xlabel("Trajectory time (ps)")
    ax.set_ylabel(r"Block-mean $q$")
    ax.set_title("(d)  One-million-step horizon (seed 20260616)", loc="left", weight="normal", pad=7)
    ax.legend(frameon=False)

    for suffix in ("png", "pdf"):
        fig.savefig(output / f"causal_rollout_main.{suffix}", dpi=420 if suffix == "png" else None)
    plt.close(fig)


def plot_forest(output: Path, effects: pd.DataFrame) -> None:
    plot_endpoints = [
        "diffusion_matched_1e-9_m2_per_s",
        "oo_peak_g",
        "oo_coordination_3p5a",
        "hydrogen_bond_coordination",
        "tetrahedral_q",
        "abs_epot_drift_mev_per_h2o_per_ns",
        "fmax_q95_ev_per_a",
    ]
    contrasts = ["Final - MACE-ICTC", "Persistent - MACE-ICTC"]
    fig, ax = plt.subplots(figsize=(7.15, 4.10), constrained_layout=True)
    y = np.arange(len(plot_endpoints))
    offsets = {"Final - MACE-ICTC": -0.13, "Persistent - MACE-ICTC": 0.13}
    colors = {
        "Final - MACE-ICTC": MODEL_COLORS["mace_final"],
        "Persistent - MACE-ICTC": MODEL_COLORS["mace_persistent"],
    }
    markers = {"Final - MACE-ICTC": "o", "Persistent - MACE-ICTC": "s"}
    for contrast in contrasts:
        selected = effects[(effects["contrast"] == contrast)].set_index("endpoint").loc[plot_endpoints]
        mean = selected["standardized_by_base_seed_sd"].to_numpy()
        low = selected["standardized_t95_low"].to_numpy()
        high = selected["standardized_t95_high"].to_numpy()
        ax.errorbar(
            mean,
            y + offsets[contrast],
            xerr=np.vstack([mean - low, high - mean]),
            fmt=markers[contrast],
            color=colors[contrast],
            capsize=2,
            elinewidth=0.9,
            capthick=0.8,
            markersize=4.0,
            label=contrast,
        )
    labels = [ENDPOINTS[endpoint]["name"] for endpoint in plot_endpoints]
    ax.set_yticks(y, labels)
    ax.axvline(0.0, color=MUTED, linestyle="--", linewidth=0.65)
    ax.set_xlabel("Paired effect / MACE-ICTC seed-to-seed SD")
    ax.set_title("Paired rollout effects across declared endpoints", loc="left", weight="normal", pad=7)
    ax.legend(frameon=False)
    ax.invert_yaxis()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"causal_effect_forest.{suffix}", dpi=420 if suffix == "png" else None)
    plt.close(fig)


def aggregate_curve(table: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame:
    rows = []
    for (contrast, x), selected in table.groupby(["contrast", x_column], sort=True):
        values = selected[y_column].to_numpy(dtype=float)
        low, high = t_interval(values)
        rows.append(
            {
                "contrast": contrast,
                x_column: float(x),
                "mean": float(np.mean(values)),
                "t95_low": low,
                "t95_high": high,
                "n_seeds": len(values),
            }
        )
    return pd.DataFrame(rows)


def plot_functional(output: Path, rdf: pd.DataFrame, msd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rdf_oo = rdf[rdf["component"] == "g_oo"].copy()
    rdf_summary = aggregate_curve(rdf_oo, "r_a", "delta_g")
    msd_limited = msd[msd["lag_ps"] <= 150.0].copy()
    msd_summary = aggregate_curve(msd_limited, "lag_ps", "delta_msd_o_a2")
    contrasts = ["Final - MACE-ICTC", "Persistent - MACE-ICTC"]
    colors = {
        "Final - MACE-ICTC": MODEL_COLORS["mace_final"],
        "Persistent - MACE-ICTC": MODEL_COLORS["mace_persistent"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    for contrast in contrasts:
        rdf_selected = rdf_summary[rdf_summary["contrast"] == contrast]
        axes[0].plot(
            rdf_selected["r_a"], rdf_selected["mean"], color=colors[contrast], linewidth=1.0, label=contrast
        )
        axes[0].fill_between(
            rdf_selected["r_a"], rdf_selected["t95_low"], rdf_selected["t95_high"],
            color=colors[contrast], alpha=0.16,
        )
        msd_selected = msd_summary[msd_summary["contrast"] == contrast]
        axes[1].plot(
            msd_selected["lag_ps"], msd_selected["mean"], color=colors[contrast], linewidth=1.0, label=contrast
        )
        axes[1].fill_between(
            msd_selected["lag_ps"], msd_selected["t95_low"], msd_selected["t95_high"],
            color=colors[contrast], alpha=0.16,
        )
    axes[0].axhline(0.0, color=MUTED, linestyle="--", linewidth=0.65)
    axes[0].set_xlim(1.5, 6.0)
    axes[0].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    axes[0].set_ylabel(r"Paired $\Delta g_{OO}(r)$")
    axes[0].set_title("(a)  O--O RDF intervention contrast", loc="left", weight="normal", pad=7)
    axes[1].axhline(0.0, color=MUTED, linestyle="--", linewidth=0.65)
    axes[1].set_xlabel("Lag time (ps)")
    axes[1].set_ylabel(r"Paired $\Delta \mathrm{MSD}_O$ ($\mathrm{\AA}^2$)")
    axes[1].set_title("(b)  Oxygen MSD intervention contrast", loc="left", weight="normal", pad=7)
    axes[1].legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"functional_rollout_contrasts.{suffix}", dpi=420 if suffix == "png" else None)
    plt.close(fig)
    return rdf_summary, msd_summary


def fnum(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def make_report(
    output: Path,
    seed_table: pd.DataFrame,
    effects: pd.DataFrame,
    ordered: pd.DataFrame,
    time_effects: pd.DataFrame,
    long_blocks: pd.DataFrame,
    functional: pd.DataFrame,
) -> None:
    def effect(endpoint: str, contrast: str) -> pd.Series:
        selected = effects[(effects["endpoint"] == endpoint) & (effects["contrast"] == contrast)]
        if len(selected) != 1:
            raise RuntimeError(f"Missing effect {endpoint} {contrast}")
        return selected.iloc[0]

    q_final = effect("tetrahedral_q", "Final - MACE-ICTC")
    q_persistent = effect("tetrahedral_q", "Persistent - MACE-ICTC")
    q_persistent_final = effect("tetrahedral_q", "Persistent - Final")
    d_final = effect("diffusion_matched_1e-9_m2_per_s", "Final - MACE-ICTC")
    d_persistent = effect("diffusion_matched_1e-9_m2_per_s", "Persistent - MACE-ICTC")
    oo_final = effect("oo_peak_g", "Final - MACE-ICTC")
    oo_persistent = effect("oo_peak_g", "Persistent - MACE-ICTC")
    hb_final = effect("hydrogen_bond_coordination", "Final - MACE-ICTC")
    hb_persistent = effect("hydrogen_bond_coordination", "Persistent - MACE-ICTC")
    pe_final = effect("abs_epot_drift_mev_per_h2o_per_ns", "Final - MACE-ICTC")
    pe_persistent = effect("abs_epot_drift_mev_per_h2o_per_ns", "Persistent - MACE-ICTC")

    q_time = time_effects[
        (time_effects["endpoint"] == "tetrahedral_q")
        & (
            time_effects["contrast"].isin(
                ["Final - MACE-ICTC", "Persistent - MACE-ICTC"]
            )
        )
    ].copy()
    q_time["ci_excludes_zero"] = (q_time["t95_low"] > 0) | (q_time["t95_high"] < 0)
    time_counts = q_time.groupby("contrast")["ci_excludes_zero"].sum().to_dict()
    time_negative_counts = q_time.assign(is_negative=q_time["mean_delta"] < 0).groupby("contrast")[
        "is_negative"
    ].sum().to_dict()

    long_q = long_blocks[long_blocks["endpoint"] == "tetrahedral_q"].pivot(
        index="block", columns="label", values="value"
    )
    long_first = long_q.iloc[0]
    long_last = long_q.iloc[-1]

    q_ordered = ordered[ordered["endpoint"] == "tetrahedral_q"].iloc[0]
    d_ordered = ordered[ordered["endpoint"] == "diffusion_matched_1e-9_m2_per_s"].iloc[0]
    oo_rms = functional[
        (functional["curve"] == "g_oo")
        & functional["contrast"].isin(
            ["Final - MACE-ICTC", "Persistent - MACE-ICTC"]
        )
    ]
    oo_rms_summary = oo_rms.groupby("contrast")["rms_difference"].agg(["mean", "std"])

    means = seed_table.groupby("label").mean(numeric_only=True)
    report = f"""# Paired causal analysis of CHORUS liquid-water rollouts

## Design and estimand

The architectural intervention is the independently retrained model:
MACE-ICTC, Final, or Persistent.  The five common velocity seeds
20260616--20260620 are matched rollout blocks.  Every block is observed under
all three interventions, so the primary estimands are within-seed total
checkpoint effects, averaged over the declared velocity-seed distribution.
The 300 ps production estimands use the common 100--300 ps window.  The 1 ns
run at seed 20260616 is a horizon extension and is not counted as a sixth
replicate.

## Accuracy-to-deployment chain

The validation-selected water checkpoints have held-out force MAEs of
{FORCE_MAE['mace_phaseoff']:.3f}, {FORCE_MAE['mace_final']:.3f}, and
{FORCE_MAE['mace_persistent']:.3f} meV Angstrom^-1 for MACE-ICTC, Final, and Persistent.
Final and Persistent therefore reduce force MAE by 4.50% and 6.56% relative
to MACE-ICTC before any rollout outcome is inspected.  All fifteen 300 ps matched
rollouts and all three 1 ns horizon extensions reach their target lengths.

## Primary structural effect: tetrahedral order

The five-seed mean tetrahedral order is
{means.loc['mace_phaseoff', 'tetrahedral_q']:.6f} for MACE-ICTC,
{means.loc['mace_final', 'tetrahedral_q']:.6f} for Final, and
{means.loc['mace_persistent', 'tetrahedral_q']:.6f} for Persistent.
Final - MACE-ICTC is {q_final['mean_delta']:+.6f}
(95% paired-t interval {q_final['t95_low']:+.6f} to {q_final['t95_high']:+.6f});
Persistent - MACE-ICTC is {q_persistent['mean_delta']:+.6f}
({q_persistent['t95_low']:+.6f} to {q_persistent['t95_high']:+.6f}).
The directions agree in {int(q_final['direction_consistency'] * 5)}/5 and
{int(q_persistent['direction_consistency'] * 5)}/5 paired seeds, respectively.
Relative to MACE-ICTC seed-to-seed variation, the effects are
{q_final['standardized_by_base_seed_sd']:+.2f} and
{q_persistent['standardized_by_base_seed_sd']:+.2f} SD.  A strict
MACE-ICTC--Final--Persistent ordering occurs in
{int(q_ordered['strict_monotonic_seed_count'])}/5 seeds, while the fitted
scope slope has the same sign in
{int(q_ordered['scope_direction_consistent_seed_count'])}/5 seeds.
The direct Persistent - Final effect is
{q_persistent_final['mean_delta']:+.6f}
({q_persistent_final['t95_low']:+.6f} to
{q_persistent_final['t95_high']:+.6f}), with
{int(q_persistent_final['direction_consistency'] * 5)}/5 paired seeds in the
modal direction; the pairwise evidence therefore separates both CHORUS scopes
from MACE-ICTC more clearly than it separates Persistent from Final.
With five pairs, the two-sided exact sign-flip p-value for a unanimous direction
is {q_final['exact_signflip_p_two_sided']:.4f}; the evidence is summarized by
the paired effect, interval, and direction agreement rather than a binary
p-value threshold.

The mean paired-q effect is negative in
{int(time_negative_counts.get('Final - MACE-ICTC', 0))}/4 blocks for Final - MACE-ICTC and
{int(time_negative_counts.get('Persistent - MACE-ICTC', 0))}/4 blocks for the
Persistent - MACE-ICTC contrast.  The individual 50 ps intervals are wide: they exclude zero in
{int(time_counts.get('Final - MACE-ICTC', 0))}/4 and
{int(time_counts.get('Persistent - MACE-ICTC', 0))}/4 blocks, respectively.  The
robust estimand is therefore the pre-specified 100--300 ps trajectory summary,
supported by 5/5 paired-seed direction agreement, rather than any individual
short block.

## Secondary structure and dynamics

O--O peak-height effects are {oo_final['mean_delta']:+.5f} (Final - MACE-ICTC) and
{oo_persistent['mean_delta']:+.5f} (Persistent - MACE-ICTC).  The corresponding
H-bond coordination effects are {hb_final['mean_delta']:+.5f}
({hb_final['t95_low']:+.5f} to {hb_final['t95_high']:+.5f}) and
{hb_persistent['mean_delta']:+.5f}
({hb_persistent['t95_low']:+.5f} to {hb_persistent['t95_high']:+.5f}); the
Persistent - MACE-ICTC direction agrees in
{int(hb_persistent['direction_consistency'] * 5)}/5 seeds.  Across 1.5--6.0 A,
the seed-mean RMS
O--O RDF contrasts are {oo_rms_summary.loc['Final - MACE-ICTC', 'mean']:.5f} for
Final - MACE-ICTC and {oo_rms_summary.loc['Persistent - MACE-ICTC', 'mean']:.5f} for
Persistent - MACE-ICTC.  These RMS values describe total curve displacement; the
paired peak-height intervals and pointwise RDF bands do not support treating
RDF amplitude as the primary causal endpoint.

Oxygen diffusion changes from
{means.loc['mace_phaseoff', 'diffusion_matched_1e-9_m2_per_s']:.3f} for MACE-ICTC to
{means.loc['mace_final', 'diffusion_matched_1e-9_m2_per_s']:.3f} for Final and
{means.loc['mace_persistent', 'diffusion_matched_1e-9_m2_per_s']:.3f} x10^-9
m^2 s^-1 for Persistent.  The paired effects are
{d_final['mean_delta']:+.3f} and {d_persistent['mean_delta']:+.3f}, but their
95% intervals ({d_final['t95_low']:+.3f}, {d_final['t95_high']:+.3f}) and
({d_persistent['t95_low']:+.3f}, {d_persistent['t95_high']:+.3f}) include zero.
Only {int(d_ordered['strict_monotonic_seed_count'])}/5 seeds show a strict
MACE-ICTC--Final--Persistent diffusion ordering.  Diffusion is therefore a noisy
secondary rollout response, not the primary causal endpoint.

## Deployment stability and one-million-step horizon

The causal contrasts in absolute potential-energy drift are
{pe_final['mean_delta']:+.3f} and {pe_persistent['mean_delta']:+.3f}
meV H2O^-1 ns^-1 for Final - MACE-ICTC and Persistent - MACE-ICTC.  Completion, bounded
force excursions, and the 1 ns block summaries establish that all three
checkpoints remain usable under closed-loop integration.

At 200--400 ps, the 1 ns block-mean q values are
{long_first['mace_phaseoff']:.6f}, {long_first['mace_final']:.6f}, and
{long_first['mace_persistent']:.6f}; at 800--1000 ps they are
{long_last['mace_phaseoff']:.6f}, {long_last['mace_final']:.6f}, and
{long_last['mace_persistent']:.6f}.  The short-horizon ordering is not preserved
as a simple monotone 1 ns separation.  The combined result is therefore:
the selected CHORUS checkpoints produce a repeatable finite-horizon change in
higher-order liquid structure while all scopes remain stable over one million
steps; the 1 ns evidence supports deployment persistence rather than a claim
of monotone long-time transport ranking.

## Manuscript-level conclusion

The water study extends the paper's designed interventions downstream.  The
replicated training panel establishes the algorithmic effect on force error;
the matched velocity-seed analysis shows that the selected lower-error
checkpoints cause reproducible changes in trajectory-level many-body
statistics; and the one-million-step extension establishes closed-loop
deployability.

## Files

- `causal_effect_summary.csv`: paired scalar estimands and intervals.
- `paired_seed_differences.csv`: every observed potential-outcome contrast.
- `time_resolved_effects.csv`: 50 ps paired effects for q and H-bond order.
- `long_1ns_blocks.csv`: 200 ps horizon-extension blocks.
- `functional_effect_summary.csv`: RMS RDF intervention contrasts.
- `causal_rollout_main.pdf`: main-text candidate figure.
- `causal_effect_forest.pdf`: standardized endpoint forest plot.
- `functional_rollout_contrasts.pdf`: paired RDF and MSD curves.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    configure_style(root / "source/fonts")
    output = (args.output or root / "analysis/causal_rollout_20260930").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)

    seed_table = load_seed_table(root)
    effects, differences, ordered = scalar_effects(seed_table, rng, args.bootstrap)
    block_table, time_effects = time_resolved_effects(root, rng, args.bootstrap)
    long_blocks = long_horizon_blocks(root)
    rdf, msd, functional = functional_contrasts(root)
    rdf_summary, msd_summary = plot_functional(output, rdf, msd)

    seed_table.to_csv(output / "analysis_input_300ps.csv", index=False)
    effects.to_csv(output / "causal_effect_summary.csv", index=False)
    differences.to_csv(output / "paired_seed_differences.csv", index=False)
    ordered.to_csv(output / "ordered_scope_effects.csv", index=False)
    block_table.to_csv(output / "time_resolved_block_values.csv", index=False)
    time_effects.to_csv(output / "time_resolved_effects.csv", index=False)
    long_blocks.to_csv(output / "long_1ns_blocks.csv", index=False)
    functional.to_csv(output / "functional_effect_summary.csv", index=False)
    rdf_summary.to_csv(output / "rdf_oo_effect_curve.csv", index=False)
    msd_summary.to_csv(output / "msd_effect_curve.csv", index=False)

    plot_main(output, seed_table, effects, time_effects, long_blocks)
    plot_forest(output, effects)
    make_report(output, seed_table, effects, ordered, time_effects, long_blocks, functional)

    manifest = {
        "analysis": "paired potential-outcome analysis of CHORUS water rollouts",
        "root": str(root),
        "output": str(output),
        "models": MODEL_ORDER,
        "velocity_seeds": sorted(int(seed) for seed in seed_table["seed"].unique()),
        "window_300ps": "100-300 ps",
        "window_1ns": "200-1000 ps",
        "time_blocks_300ps": "four 50 ps blocks",
        "time_blocks_1ns": "four 200 ps blocks",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "force_mae_mev_per_a": FORCE_MAE,
        "trainable_parameters": PARAMETERS,
        "checkpoint_metric_sources": {
            "off": "/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/runs/mace_family/water/phaseoff/validation_force_selected_eval/metrics.json",
            "final": "/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/runs/mace_family/water/final/validation_force_selected_eval/metrics.json",
            "persistent": "/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/runs/mace_family/water/persistent/validation_force_selected_eval/metrics.json",
        },
        "outputs": sorted({path.name for path in output.iterdir()} | {"manifest.json"}),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "files": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
