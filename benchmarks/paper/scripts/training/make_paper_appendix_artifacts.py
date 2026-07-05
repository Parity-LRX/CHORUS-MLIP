#!/usr/bin/env python3
"""Generate appendix tables and MD trajectory-property figures for the paper."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path("/Users/sara/Desktop/code/MACE-ICTC")
PAPER = Path("/Users/sara/Desktop/code/mace_ictc_paper")
SUMMARY_DIR = REPO / "benchmarks/paper/results/training/paper_training_summary_20260618"
PAPER_ARTIFACT_TRAIN = PAPER / "benchmark_artifacts/results/training"
PAPER_ARTIFACT_FIG = PAPER / "benchmark_artifacts/figures"
PAPER_FIG = PAPER / "figures"
PAPER_TABLE = PAPER / "tables"

SMALL_MD_DIR = REPO / "benchmarks/paper/results/model/md_parity_off23_long_10000_20260615"
LARGE_MD_DIR = REPO / "benchmarks/paper/results/model/md_parity_off23_large_10000_20260616"

MODE_LABELS = {
    "mace_e3nn": "MACE e3nn",
    "mace_cueq": "MACE cuEq",
    "ictd_bridge_u_eager": "ICTC eager",
    "ictd_bridge_u_makefx": "ICTC compiled",
    "ictd_cueq_makefx": "ICTC+cuEq compiled",
    "ictd_bridge_u": "ICTC",
    "ictd_cueq": "ICTC+cuEq",
}

DATASET_LABELS = {
    "revised_benzene": "Benzene",
    "revised_ethanol": "Ethanol",
    "revised_aspirin": "Aspirin",
    "cheng_water": "Water",
}

CASE_LABELS = {
    "ethanol": "Ethanol",
    "acetic_acid": "Acetic acid",
    "acetamide": "Acetamide",
    "benzene": "Benzene",
    "benzene_64_grid": "Benzene 64-cell",
}

CASE_ORDER = ["ethanol", "acetic_acid", "acetamide", "benzene", "benzene_64_grid"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
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


def write_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")


def copy_fig(stem: Path) -> None:
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    PAPER_ARTIFACT_FIG.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg"]:
        src = stem.with_suffix(suffix)
        shutil.copy2(src, PAPER_FIG / src.name)
        shutil.copy2(src, PAPER_ARTIFACT_FIG / src.name)


def format_pm(mean: float, std: float, scale: float = 1.0, digits: int = 2) -> str:
    return f"${mean * scale:.{digits}f}\\pm{std * scale:.{digits}f}$"


def sci_tex(value: float, sig: int = 2) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"${mantissa:.{sig}f}\\times10^{{{exponent}}}$"


def escape_tex(text: str) -> str:
    return text.replace("_", "\\_")


def make_training_table() -> None:
    df = pd.read_csv(PAPER_ARTIFACT_TRAIN / "md17_training_aggregate_combined.csv")
    system_order = ["revised_benzene", "revised_ethanol", "revised_aspirin", "cheng_water"]
    mode_order = [
        "mace_e3nn",
        "mace_cueq",
        "ictd_bridge_u_eager",
        "ictd_bridge_u_makefx",
        "ictd_cueq_makefx",
    ]
    by_key = df.set_index(["mode", "dataset"])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Matched 300-epoch training summary for the three revised MD17 molecules and the public Cheng liquid-water split. Mode labels denote the training execution paths defined in Appendix~\ref{sec:appendix-numerical-result-tables}. $F_{\mathrm{best}}$ is the best validation force RMSE in meV~\AA$^{-1}$, and $E_{\mathrm{final}}$ is the final validation energy RMSE in meV atom$^{-1}$. Entries are mean $\pm$ standard deviation over three seeds.}",
        r"\label{tab:appendix-md17-training-summary}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth}C{0.10\textwidth}C{0.155\textwidth}C{0.155\textwidth}C{0.155\textwidth}C{0.155\textwidth}@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{Mode} & Metric & Benzene & Ethanol & Aspirin & Water \\",
        r"\midrule",
    ]
    for mode_index, mode in enumerate(mode_order):
        force_values = []
        energy_values = []
        for dataset in system_order:
            row = by_key.loc[(mode, dataset)]
            force_values.append(
                format_pm(row["best_force_rmse_eV_A_mean"], row["best_force_rmse_eV_A_std"], scale=1000, digits=2)
            )
            energy_values.append(
                format_pm(row["final_energy_rmse_eV_atom_mean"], row["final_energy_rmse_eV_atom_std"], scale=1000, digits=3)
            )
        if mode_index > 0:
            lines.append(r"\addlinespace[0.2em]")
        lines.append(" & ".join([MODE_LABELS[mode], r"$F_{\mathrm{best}}$"] + force_values) + r" \\")
        lines.append(
            " & ".join([r"", r"$E_{\mathrm{final}}$"] + energy_values) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    (PAPER_TABLE / "appendix_training_summary.tex").write_text("\n".join(lines))


def make_ntk_table() -> None:
    df = pd.read_csv(PAPER_ARTIFACT_TRAIN / "ntk_by_system_mode.csv")
    system_order = ["revised_benzene", "revised_ethanol", "revised_aspirin", "cheng_water"]
    mode_order = ["mace_e3nn", "mace_cueq", "ictd_bridge_u", "ictd_cueq"]
    df = df[df["dataset"].isin(system_order)].copy()
    by_key = df.set_index(["mode", "dataset"])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Empirical weighted-output NTK spectrum summary at initialization on the three revised MD17 systems and the Cheng liquid-water split. Mode labels denote the initialization-time parameterization and product backend, not compiled training graphs. Entries are means over three sampled batches; standard deviations are archived in the accompanying CSV records.}",
        r"\label{tab:appendix-ntk-spectrum-summary}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.17\textwidth}C{0.105\textwidth}C{0.155\textwidth}C{0.155\textwidth}C{0.155\textwidth}C{0.155\textwidth}@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{Mode} & Metric & Benzene & Ethanol & Aspirin & Water \\",
        r"\midrule",
    ]
    metrics = [
        ("lambda_min_pos_mean", r"$\lambda_{\min}^+$"),
        ("lambda_max_mean", r"$\lambda_{\max}$"),
        ("kappa_pos_mean", r"$\kappa^+$"),
        ("trace_mean", r"$\mathrm{tr}$"),
    ]
    for mode_index, mode in enumerate(mode_order):
        if mode_index > 0:
            lines.append(r"\addlinespace[0.2em]")
        for metric_index, (column, metric_label) in enumerate(metrics):
            values = [sci_tex(by_key.loc[(mode, dataset)][column]) for dataset in system_order]
            mode_label = MODE_LABELS[mode] if metric_index == 0 else ""
            lines.append(" & ".join([mode_label, metric_label] + values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    (PAPER_TABLE / "appendix_ntk_summary.tex").write_text("\n".join(lines))


def load_md_case(case: str) -> dict[str, np.ndarray | dict | str]:
    if case == "benzene_64_grid":
        directory = LARGE_MD_DIR
        suffix = case
    else:
        directory = SMALL_MD_DIR
        suffix = case
    native = np.load(directory / f"native_float64_{suffix}.npz")
    ictd = np.load(directory / f"ictd_float64_{suffix}.npz")
    on_native = np.load(directory / f"ictd_on_native_float64_{suffix}.npz")
    with (directory / f"compare_float64_{suffix}.json").open() as handle:
        summary = json.load(handle)
    return {"case": case, "native": native, "ictd": ictd, "on_native": on_native, "summary": summary}


def frame_steps(z: np.lib.npyio.NpzFile, summary: dict) -> np.ndarray:
    if "sample_steps" in z.files:
        return z["sample_steps"].astype(int)
    return np.arange(z["energies"].shape[0], dtype=int)


def make_md_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    for case in CASE_ORDER:
        data = load_md_case(case)
        native = data["native"]
        ictd = data["ictd"]
        on_native = data["on_native"]
        summary = data["summary"]
        steps = frame_steps(native, summary)
        atoms = int(summary["atoms"])
        dt_fs = float(summary["dt_fs"])
        time_ps = steps * dt_fs / 1000.0
        native_e = native["energies"]
        ictd_e = ictd["energies"]
        on_native_e = on_native["energies"]
        native_f = native["forces"]
        on_native_f = on_native["forces"]
        pos_rms = np.sqrt(np.mean((native["positions"] - ictd["positions"]) ** 2, axis=(1, 2)))
        vel_rms = np.sqrt(np.mean((native["velocities"] - ictd["velocities"]) ** 2, axis=(1, 2)))
        force_rms = np.sqrt(np.mean((native_f - on_native_f) ** 2, axis=(1, 2)))
        energy_err_per_atom_mev = np.abs(native_e - on_native_e) / atoms * 1000.0
        native_drift = (native_e - native_e[0]) / atoms * 1000.0
        ictd_drift = (ictd_e - ictd_e[0]) / atoms * 1000.0
        temp_native = native["temperatures"]
        temp_ictd = ictd["temperatures"]
        for i in range(len(steps)):
            rows.append(
                {
                    "case": case,
                    "case_label": CASE_LABELS[case],
                    "atoms": atoms,
                    "step": int(steps[i]),
                    "time_ps": float(time_ps[i]),
                    "native_energy_drift_meV_atom": float(native_drift[i]),
                    "ictd_energy_drift_meV_atom": float(ictd_drift[i]),
                    "native_temperature_K": float(temp_native[i]),
                    "ictd_temperature_K": float(temp_ictd[i]),
                    "same_frame_abs_energy_error_meV_atom": float(energy_err_per_atom_mev[i]),
                    "same_frame_force_rms_eV_A": float(force_rms[i]),
                    "independent_position_rms_A": float(pos_rms[i]),
                    "independent_velocity_rms_A_fs_units": float(vel_rms[i]),
                }
            )
        summaries.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "atoms": atoms,
                "recorded_frames": len(steps),
                "steps": int(summary["steps"]),
                "dt_fs": dt_fs,
                "native_temp_mean_K": float(np.mean(temp_native)),
                "ictd_temp_mean_K": float(np.mean(temp_ictd)),
                "native_energy_drift_final_meV_atom": float(native_drift[-1]),
                "ictd_energy_drift_final_meV_atom": float(ictd_drift[-1]),
                "same_frame_energy_abs_max_meV_atom": float(np.nanmax(energy_err_per_atom_mev)),
                "same_frame_force_rms_max_eV_A": float(np.nanmax(force_rms)),
                "independent_position_rms_max_A": float(np.nanmax(pos_rms)),
                "independent_velocity_rms_max_A_fs_units": float(np.nanmax(vel_rms)),
                "native_ms_per_step": float(summary["native_ms_per_step_including_eval"]),
                "ictd_ms_per_step": float(summary["ictd_ms_per_step_including_eval"]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def plot_md_properties(records: pd.DataFrame, stem: Path) -> None:
    colors = {
        "ethanol": "#4C78A8",
        "acetic_acid": "#F58518",
        "acetamide": "#54A24B",
        "benzene": "#B279A2",
        "benzene_64_grid": "#E45756",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.0))
    panels = [
        ("same_frame_abs_energy_error_meV_atom", "Same-frame $|\\Delta E|$ (meV atom$^{-1}$)", "log"),
        ("same_frame_force_rms_eV_A", "Same-frame force RMS (eV \\AA$^{-1}$)", "log"),
        ("independent_position_rms_A", "Independent trajectory RMS $\\Delta r$ (\\AA)", "log"),
        ("independent_velocity_rms_A_fs_units", "Independent trajectory RMS $\\Delta v$ (\\AA fs$^{-1}$)", "log"),
    ]
    for ax, (metric, ylabel, scale) in zip(axes.flat, panels):
        for case in CASE_ORDER:
            df = records[records["case"] == case]
            y = df[metric].to_numpy(float)
            if scale == "log":
                y = np.maximum(y, 1e-18)
            ax.plot(df["time_ps"], y, lw=1.7, color=colors[case], label=CASE_LABELS[case])
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(ylabel)
        if scale == "log":
            ax.set_yscale("log")
    axes[0, 0].set_title("Energy residual on native frames")
    axes[0, 1].set_title("Force residual on native frames")
    axes[1, 0].set_title("Independent trajectory positions")
    axes[1, 1].set_title("Independent trajectory velocities")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.text(
        0.5,
        -0.01,
        "MACE-OFF23 small, fp64, VelocityVerlet, 10000 steps at 0.25 fs. The 768-atom benzene grid is recorded every 20 steps.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    write_fig(fig, stem)
    plt.close(fig)


def make_md_tables(summary: pd.DataFrame) -> None:
    therm_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Long-MD trajectory-property summary for native MACE and MACE-ICTC. Energy drift is the final total-energy change per atom over 10000 velocity-Verlet steps and is reported in meV atom$^{-1}$.}",
        r"\label{tab:appendix-md-trajectory-properties}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.17\textwidth}C{0.06\textwidth}C{0.07\textwidth}C{0.125\textwidth}C{0.125\textwidth}C{0.14\textwidth}C{0.14\textwidth}@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{System} & Atoms & Frames & $\bar T_{\mathrm{native}}$ (K) & $\bar T_{\mathrm{ICTC}}$ (K) & Native drift & ICTC drift \\",
        r"\midrule",
    ]
    resid_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Long-MD checkpoint-correspondence residuals and wall-clock records. Same-frame energy errors are in meV atom$^{-1}$, force RMS values are in eV~\AA$^{-1}$, and trajectory position residuals are in \AA. Same-frame quantities compare ICTC evaluations on native MACE frames; trajectory residuals compare independently integrated native and ICTC trajectories.}",
        r"\label{tab:appendix-md-correspondence-properties}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth}C{0.16\textwidth}C{0.16\textwidth}C{0.16\textwidth}C{0.13\textwidth}C{0.13\textwidth}@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{System} & Max $|\Delta E|$/atom & Max force RMS & Max RMS $\Delta r$ & Native ms/step & ICTC ms/step \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        therm_lines.append(
            " & ".join(
                [
                    row["case_label"],
                    f"{int(row['atoms'])}",
                    f"{int(row['recorded_frames'])}",
                    f"${row['native_temp_mean_K']:.2f}$",
                    f"${row['ictd_temp_mean_K']:.2f}$",
                    f"${row['native_energy_drift_final_meV_atom']:.3f}$",
                    f"${row['ictd_energy_drift_final_meV_atom']:.3f}$",
                ]
            )
            + r" \\"
        )
        resid_lines.append(
            " & ".join(
                [
                    row["case_label"],
                    sci_tex(row["same_frame_energy_abs_max_meV_atom"]),
                    sci_tex(row["same_frame_force_rms_max_eV_A"]),
                    sci_tex(row["independent_position_rms_max_A"]),
                    f"${row['native_ms_per_step']:.2f}$",
                    f"${row['ictd_ms_per_step']:.2f}$",
                ]
            )
            + r" \\"
        )
    therm_lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    resid_lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    (PAPER_TABLE / "appendix_md_trajectory_properties.tex").write_text("\n".join(therm_lines))
    (PAPER_TABLE / "appendix_md_correspondence_properties.tex").write_text("\n".join(resid_lines))


def main() -> None:
    setup_style()
    PAPER_TABLE.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_ARTIFACT_TRAIN.mkdir(parents=True, exist_ok=True)

    make_training_table()
    make_ntk_table()

    records, summary = make_md_records()
    records_path = SUMMARY_DIR / "md_trajectory_properties.csv"
    summary_path = SUMMARY_DIR / "md_trajectory_summary.csv"
    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)
    shutil.copy2(records_path, PAPER_ARTIFACT_TRAIN / records_path.name)
    shutil.copy2(summary_path, PAPER_ARTIFACT_TRAIN / summary_path.name)

    fig_stem = SUMMARY_DIR / "figures/off23_md_trajectory_properties"
    plot_md_properties(records, fig_stem)
    copy_fig(fig_stem)
    make_md_tables(summary)
    print(f"wrote appendix tables to {PAPER_TABLE}")
    print(f"wrote MD records to {records_path} and {summary_path}")


if __name__ == "__main__":
    main()
