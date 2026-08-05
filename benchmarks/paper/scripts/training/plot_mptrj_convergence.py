"""Plot the matched MPtrj convergence comparison used in Figure 6.

The two source runs contain one value per epoch and one run per model.  The
figure therefore shows the observed curves without uncertainty bands.  Epochs
0--2 are excluded as the optimizer warm-up transient, consistently with the
manuscript discussion.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MACE_COLOR = "#4C78A8"
ICTC_COLOR = "#54A24B"

# epoch, ICTC force RMSE (eV/A), ICTC energy RMSE (eV/atom)
ICTC = np.array(
    [
        [0, 3.217689, 0.254843], [1, 11.951991, 0.315567], [2, 2.915186, 0.149165],
        [3, 0.185515, 0.120618], [4, 0.167617, 0.109500], [5, 0.148540, 0.100996],
        [6, 0.147686, 0.092524], [7, 0.136821, 0.085296], [8, 0.129905, 0.084352],
        [9, 0.128116, 0.080710], [10, 0.127056, 0.079523], [11, 0.124917, 0.078400],
        [12, 0.122652, 0.076406], [13, 0.122487, 0.075946], [14, 0.122265, 0.075086],
        [15, 0.121062, 0.074991], [16, 0.121414, 0.073976], [17, 0.121023, 0.073854],
        [18, 0.120731, 0.073599], [19, 0.120696, 0.073448], [20, 0.120461, 0.073272],
        [21, 0.120517, 0.073259], [22, 0.120292, 0.073168], [23, 0.120268, 0.073247],
        [24, 0.120316, 0.072997],
    ]
)

# epoch, MACE force RMSE (meV/A), MACE energy RMSE (meV/atom)
MACE = np.array(
    [
        [0, 459.28, 340.78], [1, 314.95, 243.54], [2, 281.26, 206.42],
        [3, 255.18, 192.94], [4, 247.56, 176.85], [5, 240.22, 173.90],
        [6, 233.87, 167.11], [7, 220.03, 162.20], [8, 219.77, 159.82],
        [9, 216.77, 158.23], [10, 216.83, 157.79], [11, 214.22, 155.99],
        [12, 212.34, 154.40], [13, 210.49, 153.77], [14, 212.06, 153.77],
        [15, 211.71, 153.36], [16, 211.44, 153.19], [17, 211.52, 152.72],
        [18, 211.34, 152.60], [19, 210.92, 152.42], [20, 210.95, 152.43],
        [21, 210.83, 152.37], [22, 210.86, 152.36], [23, 210.80, 152.28],
        [24, 210.83, 152.25],
    ],
    dtype=float,
)
MACE[:, 1:] /= 1000.0

START = 3
STEPS_PER_EPOCH = 74077


def epoch_to_million_steps(epoch: np.ndarray | float) -> np.ndarray | float:
    return (epoch + 1) * STEPS_PER_EPOCH / 1.0e6


def million_steps_to_epoch(steps: np.ndarray | float) -> np.ndarray | float:
    return steps * 1.0e6 / STEPS_PER_EPOCH - 1


def plot(out_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    mace = MACE[MACE[:, 0] >= START]
    ictc = ICTC[ICTC[:, 0] >= START]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.25), sharex=True)
    panels = [
        (1, r"Force RMSE (eV $\AA^{-1}$)", (0.105, 0.265)),
        (2, r"Energy RMSE (eV atom$^{-1}$)", (0.062, 0.202)),
    ]

    for ax, (idx, ylabel, ylim) in zip(axes, panels):
        ax.plot(
            mace[:, 0], mace[:, idx], color=MACE_COLOR, lw=2.1,
            marker="o", ms=3.2, markevery=3, label="Native MACE (e3nn)",
        )
        ax.plot(
            ictc[:, 0], ictc[:, idx], color=ICTC_COLOR, lw=2.1,
            marker="o", ms=3.2, markevery=3, label="ICTC",
        )
        ax.set_xlim(3, 27.6)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(3, 25, 3))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", lw=0.7, alpha=0.75)
        ax.tick_params(direction="out", length=3.5, width=0.8)

        step_axis = ax.secondary_xaxis(
            "top", functions=(epoch_to_million_steps, million_steps_to_epoch)
        )
        step_axis.set_xticks([0.4, 0.8, 1.2, 1.6, 1.85])
        step_axis.set_xticklabels(["0.4", "0.8", "1.2", "1.6", "1.85"])
        step_axis.set_xlabel(r"Cumulative optimizer steps ($\times10^6$)", fontsize=8, labelpad=4)
        step_axis.tick_params(labelsize=7.5, pad=2, length=3)

        mace_best_i = np.argmin(mace[:, idx])
        ictc_best_i = np.argmin(ictc[:, idx])
        mace_best = mace[mace_best_i, idx]
        ictc_best = ictc[ictc_best_i, idx]
        ax.scatter(mace[mace_best_i, 0], mace_best, s=32, facecolor="white", edgecolor=MACE_COLOR, lw=1.5, zorder=4)
        ax.scatter(ictc[ictc_best_i, 0], ictc_best, s=32, facecolor="white", edgecolor=ICTC_COLOR, lw=1.5, zorder=4)
        ax.text(24.55, mace[-1, idx], f"Native MACE  {mace_best:.3f}", color=MACE_COLOR, va="center", fontsize=8)
        ax.text(24.55, ictc[-1, idx], f"ICTC  {ictc_best:.3f}", color=ICTC_COLOR, va="center", fontsize=8)
        ratio = mace_best / ictc_best
        ax.text(
            0.97, 0.94, fr"Best-RMSE ratio (MACE/ICTC): {ratio:.2f}$\times$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#333333", bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#D0D0D0", "lw": 0.7},
        )

    fig.tight_layout(w_pad=2.4)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[2] / "figures" / "mptrj_training_convergence"
    plot(output)
    print(f"wrote {output.with_suffix('.png')}")
