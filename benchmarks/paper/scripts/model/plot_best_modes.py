#!/usr/bin/env python3
"""Plot benchmark curves for the five comparison modes.

Input files are the benchmark harness CSV/JSON pairs copied into
``benchmark_results/raw``. The script emits best-of-config curves and combined
fixed-configuration curves for the available ``(hidden_lmax, max_ell)`` configs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
FIG = ROOT / "figures"
CHANNELS = 64

TRAIN_MODES = {
    "mace_torch_e3nn": "MACE e3nn",
    "mace_torch_cueq": "MACE cuEq",
    "mace_ictc_bridge_u_eager": "ICTC eager",
    "mace_ictc_bridge_u_makefx_train": "ICTC compiled",
    "mace_ictc_cueq_product_makefx_train": "ICTC+cuEq compiled",
}

INFER_MODES = {
    "mace_torch_e3nn": "MACE e3nn",
    "mace_torch_cueq": "MACE cuEq",
    "mace_ictc_bridge_u_eager": "ICTC eager",
    "mace_ictc_bridge_u_aoti": "ICTC compiled",
    "mace_ictc_cueq_product_aoti": "ICTC+cuEq compiled",
}

STYLE = {
    "MACE e3nn": {"color": "#4C78A8", "marker": "o"},
    "MACE cuEq": {"color": "#F58518", "marker": "s"},
    "ICTC eager": {"color": "#54A24B", "marker": "^"},
    "ICTC compiled": {"color": "#B279A2", "marker": "D"},
    "ICTC+cuEq compiled": {"color": "#E45756", "marker": "P"},
}

AVG_DEGREE = 16
EQUIV_NEIGHBORS = 50
ATOM_TICKS = [512, 1024, 2048, 4096, 8192]
EDGE_TICKS = [atoms * AVG_DEGREE for atoms in ATOM_TICKS]
EQUIV_ATOM_TICKS = [edges / EQUIV_NEIGHBORS for edges in EDGE_TICKS]
NOTE = (
    "RTX 4090, FP32, 64 channels, avg. directed degree 16. "
    "Each point is the fastest successful run over "
    r"$(\ell_\mathrm{hidden}, \ell_\mathrm{max}) \in \{(1,2),(2,3)\}$; "
    "compiled = make_fx for training and AOTI for inference. "
    "Atom counts/throughput are converted from directed edges assuming 50 directed neighbors per atom."
)

FIXED_CONFIGS = [("1", "1"), ("1", "2"), ("2", "2"), ("2", "3")]
FIXED_NOTE = (
    "RTX 4090, FP32, 64 channels, avg. directed degree 16. "
    "Columns are fixed configurations; compiled = make_fx for training and AOTI for inference. "
    "Axes are converted from measured directed-edge counts to 50-neighbor-equivalent atoms. "
    "Missing points are failed/OOM runs."
)


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(RAW.glob("mace_ictc_vs_mace_bench_*.csv")):
        json_path = csv_path.with_suffix(".json")
        if not json_path.exists():
            continue
        meta = json.loads(json_path.read_text())["meta"]
        if int(meta.get("channels", -1)) != CHANNELS:
            continue
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_source"] = csv_path.name
                row["_channels"] = str(meta["channels"])
                row["_avg_degree"] = str(meta["avg_degree"])
                row["_dtype"] = str(meta["dtype"])
                row["_device"] = str(meta["device"])
                rows.append(row)
    return rows


def summarize_candidates(
    candidates: list[dict[str, str]],
    *,
    task: str,
    label: str,
    raw_mode: str,
    atom: int,
) -> dict[str, str] | None:
    if not candidates:
        return None
    best = min(candidates, key=lambda r: float(r["time_ms"]))
    time_ms = float(best["time_ms"])
    avg_degree = int(best["_avg_degree"])
    directed_edges = atom * avg_degree
    throughput_atoms = atom / (time_ms / 1000.0)
    throughput_edges = directed_edges / (time_ms / 1000.0)
    equiv_atoms50 = directed_edges / float(EQUIV_NEIGHBORS)
    throughput_equiv_atoms50 = throughput_edges / float(EQUIV_NEIGHBORS)
    return {
        "task": task,
        "mode": label,
        "raw_mode": raw_mode,
        "atoms": str(atom),
        "avg_degree": str(avg_degree),
        "directed_edges": str(directed_edges),
        "equiv_atoms50": f"{equiv_atoms50:.6g}",
        "time_ms": f"{time_ms:.6g}",
        "throughput_atoms_s": f"{throughput_atoms:.6g}",
        "throughput_edges_s": f"{throughput_edges:.6g}",
        "throughput_equiv_atoms50_s": f"{throughput_equiv_atoms50:.6g}",
        "hidden_lmax": best["hidden_lmax"],
        "max_ell": best["max_ell"],
        "compile_s": best["compile_s"],
        "source": best["_source"],
    }


def add_speedups(rows: list[dict[str, str]]) -> None:
    by_key = {(r["task"], r["atoms"], r["mode"]): r for r in rows}
    for row in rows:
        baseline = by_key.get((row["task"], row["atoms"], "MACE e3nn"))
        if baseline is None:
            row["speedup_vs_mace_e3nn"] = ""
            continue
        row["speedup_vs_mace_e3nn"] = f"{float(baseline['time_ms']) / float(row['time_ms']):.6g}"


def select_best(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for task, modes in (("train", TRAIN_MODES), ("inference", INFER_MODES)):
        atoms = sorted({int(r["atoms"]) for r in rows if r["task"] == task})
        for atom in atoms:
            for raw_mode, label in modes.items():
                candidates = [
                    r
                    for r in rows
                    if r["task"] == task
                    and r["mode"] == raw_mode
                    and r["status"] == "ok"
                    and r["time_ms"]
                    and int(r["atoms"]) == atom
                ]
                summary = summarize_candidates(
                    candidates,
                    task=task,
                    label=label,
                    raw_mode=raw_mode,
                    atom=atom,
                )
                if summary is not None:
                    out.append(summary)

    add_speedups(out)
    return out


def select_fixed_config(rows: list[dict[str, str]], hidden_lmax: str, max_ell: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for task, modes in (("train", TRAIN_MODES), ("inference", INFER_MODES)):
        atoms = sorted({int(r["atoms"]) for r in rows if r["task"] == task})
        for atom in atoms:
            for raw_mode, label in modes.items():
                candidates = [
                    r
                    for r in rows
                    if r["task"] == task
                    and r["mode"] == raw_mode
                    and r["status"] == "ok"
                    and r["time_ms"]
                    and int(r["atoms"]) == atom
                    and r["hidden_lmax"] == hidden_lmax
                    and r["max_ell"] == max_ell
                ]
                summary = summarize_candidates(
                    candidates,
                    task=task,
                    label=label,
                    raw_mode=raw_mode,
                    atom=atom,
                )
                if summary is not None:
                    out.append(summary)

    add_speedups(out)
    return out


def write_selected(rows: list[dict[str, str]], filename: str) -> Path:
    out = ROOT / filename
    fields = [
        "task",
        "mode",
        "raw_mode",
        "atoms",
        "avg_degree",
        "directed_edges",
        "equiv_atoms50",
        "time_ms",
        "throughput_atoms_s",
        "throughput_edges_s",
        "throughput_equiv_atoms50_s",
        "speedup_vs_mace_e3nn",
        "hidden_lmax",
        "max_ell",
        "compile_s",
        "source",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.4,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def rows_for(rows: list[dict[str, str]], task: str, modes: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = {label: [] for label in modes.values()}
    for row in rows:
        if row["task"] == task and row["mode"] in by_label:
            by_label[row["mode"]].append(row)
    for values in by_label.values():
        values.sort(key=lambda r: int(r["directed_edges"]))
    return by_label


def format_equiv_atom_tick(value: float, _pos: int) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def format_axes(ax, *, ylabel: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(EQUIV_ATOM_TICKS[0] * 0.9, EQUIV_ATOM_TICKS[-1] * 1.1)
    ax.set_xticks(EQUIV_ATOM_TICKS)
    ax.set_xlabel("Equivalent atoms per graph (50 directed neighbors)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.7)
    ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.45)
    ax.xaxis.set_major_formatter(FuncFormatter(format_equiv_atom_tick))
    ax.tick_params(direction="out", length=3, width=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def panel_title(panel: str, title: str, hidden_lmax: str | None = None, max_ell: str | None = None) -> str:
    if hidden_lmax is None or max_ell is None:
        return rf"$\bf{{{panel}}}$   {title}"
    config = rf"$(\ell_\mathrm{{hidden}},\ell_\mathrm{{max}})=({hidden_lmax},{max_ell})$"
    return rf"$\bf{{{panel}}}$   {title}, {config}"


def plot_task_panel(
    ax,
    *,
    rows: list[dict[str, str]],
    task: str,
    modes: dict[str, str],
    metric: str,
    ylabel: str,
) -> None:
    for label, values in rows_for(rows, task, modes).items():
        if metric == "speedup_vs_mace_e3nn" and label == "MACE e3nn":
            continue
        points = [
            (float(r["equiv_atoms50"]), float(r[metric]))
            for r in values
            if r.get(metric) not in (None, "")
        ]
        if not points:
            continue
        style = STYLE[label]
        scale = 1000.0 if metric == "throughput_equiv_atoms50_s" else 1.0
        ax.plot(
            [point[0] for point in points],
            [point[1] / scale for point in points],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
        )
    if metric == "speedup_vs_mace_e3nn":
        ax.axhline(1.0, color="#4C78A8", linewidth=1.0, linestyle="--", alpha=0.75)
    format_axes(ax, ylabel=ylabel)


def add_legend(fig, axes, *, ncol: int) -> None:
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    seen = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=ncol,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.55,
        borderaxespad=0.0,
    )


def plot_throughput(rows: list[dict[str, str]], *, stem: str, note: str = NOTE) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), sharex=True)
    for ax, task, modes, title, panel in [
        (axes[0], "train", TRAIN_MODES, "Training step throughput", "a"),
        (axes[1], "inference", INFER_MODES, "Inference throughput", "b"),
    ]:
        for label, values in rows_for(rows, task, modes).items():
            if not values:
                continue
            style = STYLE[label]
            ax.plot(
                [float(r["equiv_atoms50"]) for r in values],
                [float(r["throughput_equiv_atoms50_s"]) / 1000.0 for r in values],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
        format_axes(ax, ylabel="Throughput (10$^3$ equiv. atoms s$^{-1}$)")
        ax.set_title(panel_title(panel, title), loc="left", pad=4)

    add_legend(fig, axes, ncol=5)
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=6.2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.84), w_pad=1.2)
    return save_all(fig, stem)


def plot_speedup(rows: list[dict[str, str]], *, stem: str, note: str = NOTE) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), sharex=True, sharey=True)
    for ax, task, modes, title, panel in [
        (axes[0], "train", TRAIN_MODES, "Training speedup", "a"),
        (axes[1], "inference", INFER_MODES, "Inference speedup", "b"),
    ]:
        for label, values in rows_for(rows, task, modes).items():
            if label == "MACE e3nn" or not values:
                continue
            style = STYLE[label]
            ax.plot(
                [float(r["equiv_atoms50"]) for r in values],
                [float(r["speedup_vs_mace_e3nn"]) for r in values],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
        ax.axhline(1.0, color="#4C78A8", linewidth=1.0, linestyle="--", alpha=0.75)
        format_axes(ax, ylabel="Speedup vs MACE e3nn")
        ax.set_title(panel_title(panel, title), loc="left", pad=4)

    add_legend(fig, axes, ncol=4)
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=6.2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.84), w_pad=1.2)
    return save_all(fig, stem)


def plot_fixed_config_grid(
    rows_by_config: dict[tuple[str, str], list[dict[str, str]]],
    *,
    metric: str,
    stem: str,
) -> list[Path]:
    ylabel = (
        "Throughput (10$^3$ equiv. atoms s$^{-1}$)"
        if metric == "throughput_equiv_atoms50_s"
        else "Speedup vs MACE e3nn"
    )
    fig, axes = plt.subplots(2, len(FIXED_CONFIGS), figsize=(10.2, 4.45), sharex=True)
    for col_idx, (hidden_lmax, max_ell) in enumerate(FIXED_CONFIGS):
        cfg_rows = rows_by_config[(hidden_lmax, max_ell)]
        for row_idx, (task, modes, title) in enumerate(
            [
                ("train", TRAIN_MODES, "Training"),
                ("inference", INFER_MODES, "Inference"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            plot_task_panel(
                ax,
                rows=cfg_rows,
                task=task,
                modes=modes,
                metric=metric,
                ylabel=ylabel,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=6.8)
            panel = "abcdefgh"[row_idx * len(FIXED_CONFIGS) + col_idx]
            ax.set_title(panel_title(panel, title, hidden_lmax, max_ell), loc="left", pad=4)

    add_legend(fig, axes.ravel(), ncol=5 if metric == "throughput_equiv_atoms50_s" else 4)
    fig.supylabel(ylabel, x=0.025, fontsize=8.2)
    fig.supxlabel("Equivalent atoms per graph (50 directed neighbors)", y=0.062, fontsize=8.2)
    fig.text(0.5, 0.01, FIXED_NOTE, ha="center", va="bottom", fontsize=6.2)
    fig.tight_layout(rect=(0.035, 0.095, 1, 0.89), h_pad=1.0, w_pad=0.95)
    return save_all(fig, stem)


def save_all(fig, stem: str) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        path = FIG / f"{stem}.{suffix}"
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    setup_matplotlib()
    raw_rows = load_rows()
    rows = select_best(raw_rows)
    selected_path = write_selected(rows, "selected_best_modes_channels64.csv")
    paths = plot_throughput(rows, stem="best_modes_equiv50_throughput_channels64") + plot_speedup(
        rows, stem="best_modes_equiv50_speedup_channels64"
    )
    print("selected:", selected_path)

    fixed_rows: list[dict[str, str]] = []
    rows_by_config: dict[tuple[str, str], list[dict[str, str]]] = {}
    for hidden_lmax, max_ell in FIXED_CONFIGS:
        cfg_rows = select_fixed_config(raw_rows, hidden_lmax, max_ell)
        fixed_rows.extend(cfg_rows)
        rows_by_config[(hidden_lmax, max_ell)] = cfg_rows

    paths += plot_fixed_config_grid(
        rows_by_config,
        metric="throughput_equiv_atoms50_s",
        stem="fixed_configs_equiv50_throughput_channels64",
    )
    paths += plot_fixed_config_grid(
        rows_by_config,
        metric="speedup_vs_mace_e3nn",
        stem="fixed_configs_equiv50_speedup_channels64",
    )

    fixed_path = write_selected(fixed_rows, "selected_fixed_configs_channels64.csv")
    print("selected:", fixed_path)
    for path in paths:
        print("figure:", path)


if __name__ == "__main__":
    main()
