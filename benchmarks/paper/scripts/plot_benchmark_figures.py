#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKEND_LABELS = {
    "e3nn": "e3nn",
    "cartnn": "cartnn (Cartesian-3j)",
    "ictd": "ICTC eager",
    "ictd_eager": "ICTC eager",
    "ictd_compile": "ICTC compile",
    "ictd_compile_fwbw": "ICTC compile",
    "ictd_compiled": "ICTC compile",
    "ictd_aoti": "ICTC AOTI",
}

BACKEND_COLORS = {
    "e3nn": "#4a4a4a",
    "cartnn": "#c43c39",
    "ictd": "#9ecae1",
    "ictd_eager": "#9ecae1",
    "ictd_compile": "#08519c",
    "ictd_compile_fwbw": "#08519c",
    "ictd_compiled": "#08519c",
    "ictd_aoti": "#41ab5d",
}

BACKEND_MARKERS = {
    "e3nn": "s",
    "cartnn": "o",
    "ictd": "^",
    "ictd_eager": "^",
    "ictd_compile": "D",
    "ictd_compile_fwbw": "D",
    "ictd_compiled": "D",
    "ictd_aoti": "P",
}

WHOLE_MODEL_LABELS = {
    "MACE e3nn": "MACE e3nn",
    "MACE cuEq": "MACE cuEq",
    "ICTC eager": "ICTC eager",
    "ICTC compiled": "ICTC compiled",
    "ICTC+cuEq compiled": "ICTC+cuEq compiled",
}

WHOLE_MODEL_COLORS = {
    "MACE e3nn": "#4a4a4a",
    "MACE cuEq": "#d95f02",
    "ICTC eager": "#9ecae1",
    "ICTC compiled": "#08519c",
    "ICTC+cuEq compiled": "#41ab5d",
}

WHOLE_MODEL_MARKERS = {
    "MACE e3nn": "s",
    "MACE cuEq": "o",
    "ICTC eager": "^",
    "ICTC compiled": "D",
    "ICTC+cuEq compiled": "P",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["hidden_lmax", "max_ell", "channels", "edges"]:
        df[col] = df[col].astype(int)
    for col in ["total_ms", "forward_ms", "backward_ms", "edges_per_s", "peak_mem_gb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["config"] = df["hidden_lmax"].astype(str) + "/" + df["max_ell"].astype(str)
    return df


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].eq("ok")].copy()


def format_equiv_atom_tick(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def plot_matched_fusion(aoti: pd.DataFrame, fwbw: pd.DataFrame, outdir: Path) -> None:
    configs = ["1/1", "1/2", "2/2", "2/3", "3/3"]
    x = np.arange(len(configs))

    forward = ok(aoti)
    forward = forward[
        (forward["dtype"] == "float32")
        & (forward["mode"] == "forward_only")
        & (forward["channels"] == 64)
        & (forward["edges"] == 100000)
        & (forward["backend"].isin(["ictd_compile", "e3nn", "cartnn"]))
    ].copy()
    forward_pivot = (
        forward.pivot_table(index="config", columns="backend", values="total_ms", aggfunc="mean")
        .rename(columns={"ictd_compile": "ictd"})
        .reindex(configs)
    )

    train = ok(fwbw)
    train = train[
        (train["dtype"] == "float32")
        & (train["mode"] == "forward_backward")
        & (train["channels"] == 64)
        & (train["edges"] == 100000)
        & (train["backend"].isin(["ictd_compile_fwbw", "e3nn", "cartnn"]))
    ].copy()
    train_pivot = (
        train.pivot_table(index="config", columns="backend", values="total_ms", aggfunc="mean")
        .rename(columns={"ictd_compile_fwbw": "ictd"})
        .reindex(configs)
    )

    pivots = [
        ("Forward only", forward_pivot, "forward time per call (ms, log)"),
        ("Forward + backward", train_pivot, "forward+backward total time (ms, log)"),
    ]
    backend_order = ["e3nn", "cartnn", "ictd"]
    labels = {"e3nn": "e3nn", "cartnn": "cartnn (Cartesian-3j)", "ictd": "ICTC"}
    colors = {"e3nn": BACKEND_COLORS["e3nn"], "cartnn": BACKEND_COLORS["cartnn"], "ictd": BACKEND_COLORS["ictd_compile"]}
    markers = {"e3nn": BACKEND_MARKERS["e3nn"], "cartnn": BACKEND_MARKERS["cartnn"], "ictd": BACKEND_MARKERS["ictd_compile"]}

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.6, 7.3),
        gridspec_kw={"width_ratios": [1.35, 1.0], "height_ratios": [1.0, 1.0]},
        sharex="col",
    )
    width = 0.22
    offsets = np.linspace(-1, 1, len(backend_order)) * width

    for row, (workload, pivot, ylabel) in enumerate(pivots):
        ax = axes[row, 0]
        for i, backend in enumerate(backend_order):
            vals = pivot[backend].to_numpy(dtype=float) if backend in pivot else np.full(len(configs), np.nan)
            mask = np.isfinite(vals)
            ax.bar(
                x[mask] + offsets[i],
                vals[mask],
                width=width,
                label=labels[backend],
                color=colors[backend],
                edgecolor="white",
                linewidth=0.35,
            )
            for xi, yi, finite in zip(x, vals, mask):
                if not finite:
                    ax.text(
                        xi + offsets[i],
                        0.08,
                        "OOM",
                        transform=ax.get_xaxis_transform(),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=colors[backend],
                    )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(ord('a') + row * 2)}) {workload}: time")
        ax.grid(True, axis="y", which="both", alpha=0.28)
        if row == 0:
            ax.legend(frameon=False, ncol=1, loc="upper left")

        ax = axes[row, 1]
        baseline = pivot["e3nn"]
        max_speed = 1.0
        for backend in ["cartnn", "ictd"]:
            speed = baseline / pivot[backend]
            max_speed = max(max_speed, np.nanmax(speed.to_numpy(dtype=float)))
            ax.plot(
                x,
                speed,
                marker=markers[backend],
                color=colors[backend],
                linewidth=2.0,
                markersize=6,
                label=labels[backend],
            )
            if backend == "ictd":
                for xi, yi in zip(x, speed):
                    if np.isfinite(yi):
                        ax.annotate(
                            f"{yi:.2f}",
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(0, 7),
                            ha="center",
                            fontsize=8,
                            color=colors[backend],
                        )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_ylabel("speedup vs e3nn ($t_{e3nn}/t$)")
        ax.set_title(f"({chr(ord('b') + row * 2)}) {workload}: speedup")
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.legend(frameon=False, loc="upper left")
        ax.set_ylim(0, max(2.2, max_speed * 1.25))

    for ax in axes[-1, :]:
        ax.set_xlabel("hidden L / edge L")

    fig.suptitle("Operator tensor product on RTX 4090: fp32, C=64, E=100k directed edges", y=1.01, fontsize=12)
    fig.tight_layout()
    save_all(fig, outdir, "operator_matched_fusion_fp32_c64_e100k")


def plot_forward_scaling(aoti: pd.DataFrame, outdir: Path) -> None:
    df = ok(aoti)
    df = df[(df["dtype"] == "float32") & (df["mode"] == "forward_only") & (df["channels"] == 64)].copy()
    configs = ["1/1", "1/2", "2/2", "2/3", "3/3"]
    backends = ["e3nn", "cartnn", "ictd_compile", "ictd_aoti"]
    edge_ticks = [1e4, 5e4, 1e5, 5e5]
    atom_ticks = [tick / 50 for tick in edge_ticks]

    fig, axes = plt.subplots(1, len(configs), figsize=(14.2, 3.3), sharey=True)
    for ax, cfg in zip(axes, configs):
        sub = df[df["config"] == cfg]
        for backend in backends:
            b = sub[sub["backend"] == backend].sort_values("edges")
            if b.empty:
                continue
            ax.plot(
                b["edges"],
                b["edges_per_s"] / 1e6,
                marker=BACKEND_MARKERS[backend],
                color=BACKEND_COLORS[backend],
                linewidth=1.7,
                markersize=5,
                label=BACKEND_LABELS[backend],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(8e3, 6e5)
        ax.set_title(f"L {cfg}")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("directed edges")
        secax = ax.secondary_xaxis(
            "top",
            functions=(lambda x: x / 50.0, lambda x: x * 50.0),
        )
        secax.set_xscale("log")
        secax.set_xticks(atom_ticks)
        secax.set_xticklabels(["200", "1k", "2k", "10k"])
        secax.tick_params(labelsize=7, pad=1)
    axes[0].set_ylabel("throughput (million edges/s, log)")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Forward throughput scaling, fp32, C=64", y=1.03, fontsize=12)
    fig.tight_layout()
    save_all(fig, outdir, "operator_forward_throughput_scaling_fp32_c64")


def combined_eager_compiled(eager: pd.DataFrame, compiled: pd.DataFrame) -> pd.DataFrame:
    eager = eager.copy()
    compiled = compiled.copy()
    compiled["backend"] = "ictd_compiled"
    cols = eager.columns
    return pd.concat([eager, compiled[cols]], ignore_index=True)


def plot_regime_map(eager: pd.DataFrame, compiled: pd.DataFrame, outdir: Path) -> None:
    df = ok(combined_eager_compiled(eager, compiled))
    df = df[(df["channels"] == 64) & (df["edges"] == 100000)].copy()
    configs = ["1/1", "1/2", "2/2", "2/3", "3/3"]
    regimes = [("float32", "forward_only"), ("float32", "forward_backward"), ("float64", "forward_only"), ("float64", "forward_backward")]
    titles = ["fp32 forward", "fp32 forward+backward", "fp64 forward", "fp64 forward+backward"]
    backends = ["e3nn", "cartnn", "ictd", "ictd_compiled"]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.3), sharex=True)
    x = np.arange(len(configs))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(backends)) * width
    for ax, (dtype, mode), title in zip(axes.flat, regimes, titles):
        sub = df[(df["dtype"] == dtype) & (df["mode"] == mode)]
        pivot = sub.pivot_table(index="config", columns="backend", values="total_ms", aggfunc="mean").reindex(configs)
        for i, backend in enumerate(backends):
            if backend not in pivot:
                continue
            vals = pivot[backend].to_numpy(dtype=float)
            mask = np.isfinite(vals)
            ax.bar(
                x[mask] + offsets[i],
                vals[mask],
                width=width,
                color=BACKEND_COLORS[backend],
                label=BACKEND_LABELS[backend],
                edgecolor="white",
                linewidth=0.35,
            )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.grid(True, axis="y", which="both", alpha=0.25)
    axes[0, 0].set_ylabel("total time per call (ms, log)")
    axes[1, 0].set_ylabel("total time per call (ms, log)")
    axes[1, 0].set_xlabel("hidden L / edge L")
    axes[1, 1].set_xlabel("hidden L / edge L")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Operator regimes at C=64, E=100k", y=1.08, fontsize=12)
    fig.tight_layout()
    save_all(fig, outdir, "operator_regime_map_c64_e100k")


def plot_memory(eager: pd.DataFrame, compiled: pd.DataFrame, outdir: Path) -> None:
    df = ok(combined_eager_compiled(eager, compiled))
    df = df[(df["channels"] == 64) & (df["edges"] == 100000)].copy()
    configs = ["1/1", "1/2", "2/2", "2/3", "3/3"]
    regimes = [("float32", "forward_only"), ("float32", "forward_backward"), ("float64", "forward_only"), ("float64", "forward_backward")]
    titles = ["fp32 forward", "fp32 forward+backward", "fp64 forward", "fp64 forward+backward"]
    backends = ["e3nn", "cartnn", "ictd", "ictd_compiled"]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.3), sharex=True)
    x = np.arange(len(configs))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(backends)) * width
    for ax, (dtype, mode), title in zip(axes.flat, regimes, titles):
        sub = df[(df["dtype"] == dtype) & (df["mode"] == mode)]
        pivot = sub.pivot_table(index="config", columns="backend", values="peak_mem_gb", aggfunc="mean").reindex(configs)
        for i, backend in enumerate(backends):
            if backend not in pivot:
                continue
            vals = pivot[backend].to_numpy(dtype=float)
            mask = np.isfinite(vals)
            ax.bar(
                x[mask] + offsets[i],
                vals[mask],
                width=width,
                color=BACKEND_COLORS[backend],
                label=BACKEND_LABELS[backend],
                edgecolor="white",
                linewidth=0.35,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("peak allocated memory (GB)")
    axes[1, 0].set_ylabel("peak allocated memory (GB)")
    axes[1, 0].set_xlabel("hidden L / edge L")
    axes[1, 1].set_xlabel("hidden L / edge L")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Peak GPU memory at C=64, E=100k", y=1.08, fontsize=12)
    fig.tight_layout()
    save_all(fig, outdir, "operator_memory_c64_e100k")


def plot_oom_summary(eager: pd.DataFrame, aoti: pd.DataFrame, compiled: pd.DataFrame, outdir: Path) -> None:
    frames = [
        eager.assign(source="eager grid"),
        aoti.assign(source="matched-fusion fwd"),
        compiled.assign(source="compiled grid"),
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] != "ok"].copy()
    if df.empty:
        return
    counts = df.groupby(["source", "backend", "status"]).size().reset_index(name="count")
    counts["label"] = counts["backend"] + " / " + counts["status"]
    sources = list(counts["source"].drop_duplicates())
    fig, axes = plt.subplots(1, len(sources), figsize=(4.4 * len(sources), 3.6), sharey=False)
    if len(sources) == 1:
        axes = [axes]
    for ax, source in zip(axes, sources):
        sub = counts[counts["source"] == source].sort_values(["backend", "status"])
        colors = [BACKEND_COLORS.get(b, "#888888") for b in sub["backend"]]
        ax.barh(np.arange(len(sub)), sub["count"], color=colors)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["label"])
        ax.set_title(source)
        ax.set_xlabel("non-ok cells")
        ax.grid(True, axis="x", alpha=0.25)
    axes[0].set_ylabel("backend/status")
    fig.suptitle("OOM/error cells in benchmark sweeps", y=1.03, fontsize=12)
    fig.tight_layout()
    save_all(fig, outdir, "operator_non_ok_summary")


def read_whole_model_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["hidden_lmax", "max_ell", "atoms", "directed_edges"]:
        df[col] = df[col].astype(int)
    for col in ["equiv_atoms50", "time_ms", "throughput_equiv_atoms50_s", "speedup_vs_mace_e3nn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["config"] = df["hidden_lmax"].astype(str) + "/" + df["max_ell"].astype(str)
    return df


def plot_whole_model_fixed_configs(df: pd.DataFrame, outdir: Path) -> None:
    configs = ["1/1", "1/2", "2/2", "2/3"]
    tasks = ["train", "inference"]
    modes = ["MACE e3nn", "MACE cuEq", "ICTC eager", "ICTC compiled", "ICTC+cuEq compiled"]
    tick_values = sorted(df["equiv_atoms50"].dropna().unique())
    tick_labels = [format_equiv_atom_tick(v) for v in tick_values]

    plot_specs = [
        (
            "throughput_equiv_atoms50_s",
            r"throughput ($10^3$ equiv. atoms s$^{-1}$)",
            "whole_model_fixed_configs_throughput_channels64",
            modes,
        ),
        (
            "speedup_vs_mace_e3nn",
            "speedup vs MACE e3nn",
            "whole_model_fixed_configs_speedup_channels64",
            modes[1:],
        ),
    ]
    for metric, ylabel, stem, plotted_modes in plot_specs:
        fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.0), sharex=True)
        handles = []
        labels = []
        for row, task in enumerate(tasks):
            for col, cfg in enumerate(configs):
                ax = axes[row, col]
                sub = df[(df["task"] == task) & (df["config"] == cfg)].copy()
                panel_label = chr(ord("a") + row * len(configs) + col)
                ax.text(
                    0.02,
                    0.98,
                    panel_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )
                task_label = "Training" if task == "train" else "Inference"
                hidden_l, max_ell = cfg.split("/")
                ax.set_title(
                    rf"{task_label}, $(\ell_\mathrm{{hidden}},\ell_\max)=({hidden_l},{max_ell})$",
                    pad=6,
                )
                for mode in plotted_modes:
                    m = sub[sub["mode"] == mode].sort_values("equiv_atoms50")
                    if m.empty:
                        continue
                    y = m[metric] / 1000.0 if metric == "throughput_equiv_atoms50_s" else m[metric]
                    (line,) = ax.plot(
                        m["equiv_atoms50"],
                        y,
                        marker=WHOLE_MODEL_MARKERS[mode],
                        color=WHOLE_MODEL_COLORS[mode],
                        linewidth=1.7,
                        markersize=4.8,
                        label=WHOLE_MODEL_LABELS[mode],
                    )
                    if row == 0 and col == 0:
                        handles.append(line)
                        labels.append(WHOLE_MODEL_LABELS[mode])
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_xticks(tick_values)
                ax.set_xticklabels(tick_labels)
                ax.grid(True, which="both", alpha=0.24)
                if metric == "speedup_vs_mace_e3nn":
                    ax.axhline(1.0, color="#4a4a4a", linestyle="--", linewidth=0.9, alpha=0.75)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 1:
                    ax.set_xlabel("equiv. atoms per graph")
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle(
            "Whole-model synthetic benchmark on RTX 4090: fp32, C=64, 50-neighbor-equivalent scaling",
            y=1.075,
            fontsize=12,
        )
        fig.tight_layout()
        save_all(fig, outdir, stem)


def main() -> None:
    paper_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", type=Path, default=paper_dir / "results/operator")
    parser.add_argument(
        "--whole-model-dir",
        type=Path,
        default=paper_dir / "results/model",
    )
    parser.add_argument("--outdir", type=Path, default=paper_dir / "figures")
    args = parser.parse_args()
    setup_style()

    eager = read_csv(args.bench_dir / "operator_cartnn_vs_ictd.csv")
    aoti = read_csv(args.bench_dir / "operator_aoti_fwd.csv")
    compiled = read_csv(args.bench_dir / "operator_ictd_compiled.csv")
    fwbw_path = args.bench_dir / "operator_compile_fwbw_flat.csv"
    if not fwbw_path.exists():
        fwbw_path = args.bench_dir / "compile_fwbw_flat" / "operator_compile_fwbw_flat.csv"
    if not fwbw_path.exists():
        fwbw_path = Path(__file__).with_name("operator_compile_fwbw_flat.csv")
    compile_fwbw = read_csv(fwbw_path)

    plot_matched_fusion(aoti, compile_fwbw, args.outdir)
    plot_forward_scaling(aoti, args.outdir)
    plot_regime_map(eager, compiled, args.outdir)
    plot_memory(eager, compiled, args.outdir)
    plot_oom_summary(eager, aoti, compiled, args.outdir)
    fixed = read_whole_model_csv(args.whole_model_dir / "selected_fixed_configs_channels64.csv")
    plot_whole_model_fixed_configs(fixed, args.outdir)


if __name__ == "__main__":
    main()
