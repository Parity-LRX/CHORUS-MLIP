#!/usr/bin/env python3
"""Build paper figures from completed training, NTK, and MD parity records."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_ORDER = [
    "mace_e3nn",
    "mace_cueq",
    "ictd_bridge_u_eager",
    "ictd_bridge_u_makefx",
    "ictd_cueq_makefx",
]

MODE_LABELS = {
    "mace_e3nn": "MACE e3nn",
    "mace_cueq": "MACE cuEq",
    "ictd_bridge_u_eager": "ICTC eager",
    "ictd_bridge_u_makefx": "ICTC compiled",
    "ictd_cueq_makefx": "ICTC+cuEq compiled",
    "ictd_bridge_u": "ICTC",
    "ictd_cueq": "ICTC+cuEq",
}

MODE_COLORS = {
    "mace_e3nn": "#4C78A8",
    "mace_cueq": "#F58518",
    "ictd_bridge_u_eager": "#54A24B",
    "ictd_bridge_u_makefx": "#B279A2",
    "ictd_cueq_makefx": "#E45756",
    "ictd_bridge_u": "#54A24B",
    "ictd_cueq": "#E45756",
}

DATASET_LABELS = {
    "revised_benzene": "Benzene",
    "revised_ethanol": "Ethanol",
    "revised_aspirin": "Aspirin",
    "cheng_water": "Water",
}

SYSTEM_ORDER = ["revised_benzene", "revised_ethanol", "revised_aspirin", "cheng_water"]
NTK_SYSTEM_ORDER = ["revised_benzene", "revised_ethanol", "revised_aspirin", "cheng_water"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_outputs(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")


def copy_to_paper(fig_stem: Path, paper_fig_dir: Path, artifact_fig_dir: Path) -> None:
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    artifact_fig_dir.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg"]:
        src = fig_stem.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, paper_fig_dir / src.name)
            shutil.copy2(src, artifact_fig_dir / src.name)


def load_training_curves(convergence_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for directory in convergence_dirs:
        curves = read_csv(directory / "curves.csv")
        frames.append(curves)
    curves = pd.concat(frames, ignore_index=True)
    curves = curves[curves["epoch"] >= 0].copy()
    curves["dataset"] = pd.Categorical(curves["dataset"], SYSTEM_ORDER, ordered=True)
    curves["mode"] = pd.Categorical(curves["mode"], MODE_ORDER, ordered=True)
    return curves.sort_values(["dataset", "mode", "seed", "epoch"])


def load_training_aggregates(convergence_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for directory in convergence_dirs:
        agg = read_csv(directory / "aggregate_by_mode.csv")
        frames.append(agg)
    agg = pd.concat(frames, ignore_index=True)
    agg["dataset"] = pd.Categorical(agg["dataset"], SYSTEM_ORDER, ordered=True)
    agg["mode"] = pd.Categorical(agg["mode"], MODE_ORDER, ordered=True)
    return agg.sort_values(["dataset", "mode"])


def plot_training_convergence(curves: pd.DataFrame, out_stem: Path) -> None:
    fig, axes = plt.subplots(2, len(SYSTEM_ORDER), figsize=(13.6, 5.6), sharex="col", sharey=False)
    metrics = [
        ("force_rmse_eV_A", "Force RMSE (eV A$^{-1}$)", None),
        ("energy_rmse_eV_atom", "Energy RMSE (eV atom$^{-1}$)", 1.0e-4),
    ]
    for col, dataset in enumerate(SYSTEM_ORDER):
        dcurves = curves[curves["dataset"] == dataset]
        for row, (metric, ylabel, floor) in enumerate(metrics):
            ax = axes[row, col]
            for mode in MODE_ORDER:
                mcurves = dcurves[dcurves["mode"] == mode].copy()
                if mcurves.empty:
                    continue
                if floor is not None:
                    mcurves.loc[mcurves[metric] <= floor, metric] = np.nan
                by_epoch = (
                    mcurves.groupby("epoch", observed=True)[metric]
                    .agg(["mean", "std"])
                    .dropna(subset=["mean"])
                    .reset_index()
                    .sort_values("epoch")
                )
                if by_epoch.empty:
                    continue
                x = by_epoch["epoch"].to_numpy(dtype=float)
                y = by_epoch["mean"].to_numpy(dtype=float)
                s = by_epoch["std"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, y, color=MODE_COLORS[mode], lw=1.9, label=MODE_LABELS[mode])
                ax.fill_between(x, np.maximum(y - s, floor or 1e-12), y + s, color=MODE_COLORS[mode], alpha=0.12, lw=0)
            ax.set_yscale("log")
            if floor is not None:
                ax.set_ylim(bottom=floor * 0.85)
            if row == 0:
                ax.set_title(DATASET_LABELS[dataset])
            if row == 1:
                ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(
        0.5,
        -0.015,
        r"Mean over three seeds; shaded bands show one standard deviation. Energy points at or below $10^{-4}$ eV atom$^{-1}$ are omitted to avoid log-scale artifacts from rounded small values.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    write_outputs(fig, out_stem)
    plt.close(fig)


def plot_training_best(agg: pd.DataFrame, out_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.8), sharex=False)
    metrics = [
        ("best_force_rmse_eV_A_mean", "best_force_rmse_eV_A_std", "Best force RMSE (eV A$^{-1}$)"),
        ("final_energy_rmse_eV_atom_mean", "final_energy_rmse_eV_atom_std", "Final energy RMSE (eV atom$^{-1}$)"),
    ]
    x = np.arange(len(SYSTEM_ORDER))
    width = 0.15
    offsets = (np.arange(len(MODE_ORDER)) - (len(MODE_ORDER) - 1) / 2.0) * width
    for ax, (mean_col, std_col, ylabel) in zip(axes, metrics):
        for i, mode in enumerate(MODE_ORDER):
            rows = agg[agg["mode"] == mode].set_index("dataset")
            means = [rows.loc[dataset, mean_col] for dataset in SYSTEM_ORDER]
            stds = [rows.loc[dataset, std_col] for dataset in SYSTEM_ORDER]
            ax.bar(
                x + offsets[i],
                means,
                width=width,
                yerr=stds,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
                capsize=2,
                linewidth=0.4,
                edgecolor="white",
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in SYSTEM_ORDER])
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.text(
        0.5,
        -0.02,
        "Force bars report best validation RMSE during 300 epochs; energy bars report final epoch RMSE to avoid best-value artifacts from rounded log entries.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    write_outputs(fig, out_stem)
    plt.close(fig)


def infer_dataset_from_run_id(run_id: str) -> str:
    if run_id.startswith("cheng_water"):
        return "cheng_water"
    for dataset in SYSTEM_ORDER:
        if run_id.startswith(dataset):
            return dataset
    return run_id.rsplit("_batch", 1)[0]


def load_ntk_runs(rmd17_dir: Path, water_dir: Path) -> pd.DataFrame:
    frames = []
    for directory in [rmd17_dir, water_dir]:
        runs = read_csv(directory / "aggregate" / "ntk_runs.csv")
        frames.append(runs)
    runs = pd.concat(frames, ignore_index=True)
    runs["dataset"] = runs["run_id"].map(infer_dataset_from_run_id)
    runs["dataset"] = pd.Categorical(runs["dataset"], NTK_SYSTEM_ORDER, ordered=True)
    runs["mode"] = pd.Categorical(runs["mode"], ["mace_e3nn", "mace_cueq", "ictd_bridge_u", "ictd_cueq"], ordered=True)
    return runs.sort_values(["dataset", "mode", "batch_index"])


def plot_ntk(ntk: pd.DataFrame, out_stem: Path, summary_csv: Path) -> None:
    agg = (
        ntk.groupby(["dataset", "mode"], observed=True)
        .agg(
            lambda_min_pos_mean=("lambda_min_pos", "mean"),
            lambda_min_pos_std=("lambda_min_pos", "std"),
            lambda_max_mean=("lambda_max", "mean"),
            lambda_max_std=("lambda_max", "std"),
            kappa_pos_mean=("kappa_pos", "mean"),
            kappa_pos_std=("kappa_pos", "std"),
            trace_mean=("trace", "mean"),
            trace_std=("trace", "std"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.0), sharex=True)
    metrics = [
        ("lambda_min_pos_mean", "lambda_min_pos_std", r"$\lambda_{\min}^+(K)$"),
        ("lambda_max_mean", "lambda_max_std", r"$\lambda_{\max}(K)$"),
        ("kappa_pos_mean", "kappa_pos_std", r"$\kappa^+(K)$"),
        ("trace_mean", "trace_std", r"$\mathrm{tr}(K)$"),
    ]
    mode_order = ["mace_e3nn", "mace_cueq", "ictd_bridge_u", "ictd_cueq"]
    x = np.arange(len(NTK_SYSTEM_ORDER))
    width = 0.18
    offsets = (np.arange(len(mode_order)) - (len(mode_order) - 1) / 2.0) * width
    for ax, (mean_col, std_col, ylabel) in zip(axes.flat, metrics):
        for i, mode in enumerate(mode_order):
            rows = agg[agg["mode"] == mode].set_index("dataset")
            means = [rows.loc[dataset, mean_col] if dataset in rows.index else np.nan for dataset in NTK_SYSTEM_ORDER]
            stds = [rows.loc[dataset, std_col] if dataset in rows.index else np.nan for dataset in NTK_SYSTEM_ORDER]
            ax.bar(
                x + offsets[i],
                means,
                width=width,
                yerr=stds,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
                capsize=2,
                linewidth=0.4,
                edgecolor="white",
            )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in NTK_SYSTEM_ORDER], rotation=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(
        0.5,
        -0.015,
        "Empirical weighted-output NTK Gram spectra at initialization. Each bar is a mean over three sampled batches; error bars are standard deviations.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    write_outputs(fig, out_stem)
    plt.close(fig)
    agg.to_csv(summary_csv, index=False)


def plot_md_parity(summary_json: Path, out_stem: Path) -> None:
    rows = json.loads(summary_json.read_text())
    df = pd.DataFrame(rows)
    df["case_label"] = df["case"].map(lambda x: x.replace("_", " ").title())
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    x = np.arange(len(df))
    axes[0].bar(x - 0.15, df["same_frame_energy_abs_max_eV"], width=0.3, color="#4C78A8", label="max |dE|")
    axes[0].bar(x + 0.15, df["same_frame_force_rms_eV_A"], width=0.3, color="#54A24B", label="force RMS")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Same-frame error")
    axes[0].legend(
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0.0,
    )
    axes[1].bar(x, df["independent_traj_position_rms_max_A"], width=0.55, color="#B279A2")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Max trajectory RMS position error (A)")
    axes[2].bar(x - 0.16, df["native_ms_per_step_including_eval"], width=0.32, color="#4C78A8", label="native")
    axes[2].bar(x + 0.16, df["ictd_ms_per_step_including_eval"], width=0.32, color="#E45756", label="ICTC")
    axes[2].set_ylabel("ms per MD step")
    axes[2].legend(
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0.0,
    )
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["case_label"], rotation=25, ha="right")
    fig.text(
        0.5,
        -0.03,
        "MACE-OFF23 small, fp64, ASE VelocityVerlet, 10000 steps at 0.25 fs. Timing includes ASE calculator overhead.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    write_outputs(fig, out_stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=None,
        help="Optional manuscript directory to also receive copied figures/artifacts.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/paper/results/training/paper_training_summary_20260618"))
    args = parser.parse_args()

    setup_style()
    repo = args.repo_root.resolve()
    archive_dir = repo / "benchmarks/paper"
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    convergence_dirs = [
        repo / "benchmarks/paper/results/training/md17_convergence_benzene_20260617",
        repo / "benchmarks/paper/results/training/md17_convergence_ethanol_0to300_20260617",
        repo / "benchmarks/paper/results/training/md17_convergence_aspirin_20260617",
        repo / "benchmarks/paper/results/training/water_cheng_convergence_20260618",
    ]
    curves = load_training_curves(convergence_dirs)
    agg = load_training_aggregates(convergence_dirs)
    curves.to_csv(out_dir / "matched_training_curves_combined.csv", index=False)
    agg.to_csv(out_dir / "matched_training_aggregate_combined.csv", index=False)
    # Backward-compatible names used by older manuscript drafts and artifact scripts.
    curves.to_csv(out_dir / "md17_training_curves_combined.csv", index=False)
    agg.to_csv(out_dir / "md17_training_aggregate_combined.csv", index=False)

    plot_training_convergence(curves, fig_dir / "matched_training_convergence")
    plot_training_best(agg, fig_dir / "matched_training_best_rmse")
    # Backward-compatible figure names.
    plot_training_convergence(curves, fig_dir / "md17_training_convergence")
    plot_training_best(agg, fig_dir / "md17_training_best_rmse")

    ntk = load_ntk_runs(
        repo / "benchmarks/paper/results/training/ntk_multisystem_with_mace_cueq_20260617",
        repo / "benchmarks/paper/results/training/ntk_water_cheng_with_mace_cueq_20260617",
    )
    ntk.to_csv(out_dir / "ntk_runs_combined.csv", index=False)
    plot_ntk(ntk, fig_dir / "ntk_spectrum_diagnostics", out_dir / "ntk_by_system_mode.csv")

    md_summary = repo / "benchmarks/paper/results/model/md_parity_off23_long_10000_20260615/compare_float64_summary.json"
    shutil.copy2(md_summary, out_dir / "md_parity_off23_long_10000_summary.json")
    plot_md_parity(md_summary, fig_dir / "off23_md_parity_long")

    archive_fig_dir = archive_dir / "figures"
    archive_data_dir = archive_dir / "results/training"
    for stem in [
        fig_dir / "matched_training_convergence",
        fig_dir / "matched_training_best_rmse",
        fig_dir / "md17_training_convergence",
        fig_dir / "md17_training_best_rmse",
        fig_dir / "ntk_spectrum_diagnostics",
        fig_dir / "off23_md_parity_long",
    ]:
        archive_fig_dir.mkdir(parents=True, exist_ok=True)
        for suffix in [".png", ".pdf", ".svg"]:
            src = stem.with_suffix(suffix)
            if src.exists():
                shutil.copy2(src, archive_fig_dir / src.name)
    for file in out_dir.glob("*.csv"):
        shutil.copy2(file, archive_data_dir / file.name)
    shutil.copy2(out_dir / "md_parity_off23_long_10000_summary.json", archive_data_dir / "md_parity_off23_long_10000_summary.json")

    if args.paper_dir is not None:
        paper_dir = args.paper_dir.resolve()
        paper_fig_dir = paper_dir / "figures"
        artifact_fig_dir = paper_dir / "benchmark_artifacts/figures"
        artifact_data_dir = paper_dir / "benchmark_artifacts/results/training"
        artifact_data_dir.mkdir(parents=True, exist_ok=True)
        for stem in [
            fig_dir / "matched_training_convergence",
            fig_dir / "matched_training_best_rmse",
            fig_dir / "md17_training_convergence",
            fig_dir / "md17_training_best_rmse",
            fig_dir / "ntk_spectrum_diagnostics",
            fig_dir / "off23_md_parity_long",
        ]:
            copy_to_paper(stem, paper_fig_dir, artifact_fig_dir)
        for file in out_dir.glob("*.csv"):
            shutil.copy2(file, artifact_data_dir / file.name)
        shutil.copy2(out_dir / "md_parity_off23_long_10000_summary.json", artifact_data_dir / "md_parity_off23_long_10000_summary.json")

    print(f"wrote figures and summary CSVs to {out_dir}")


if __name__ == "__main__":
    main()
