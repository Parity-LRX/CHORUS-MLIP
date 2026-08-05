#!/usr/bin/env python3
"""Diagnose learned CHORUS phases and their functional interference.

The absolute phase ``theta`` is gauge dependent, so its histogram is reported
only as a descriptive diagnostic.  The primary statistics use the gauge-
invariant relative phase ``Delta theta = theta_ij - theta_ik`` for pairs of
edges that enter the same atomic environment.  The script also performs three
same-checkpoint interventions:

* ``zero``: set every phase to zero while retaining learned amplitudes;
* ``permute``: permute phases within each receiving atom, preserving the exact
  local phase marginal while breaking the learned phase--edge assignment;
* ``global-shift``: add a common angle to every phase (a U(1) sanity check).

Histograms alone show that constructive and destructive sectors are occupied;
performance changes under the interventions establish whether the learned
phase assignment is functionally used by the trained potential.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from chorus.data import H5Dataset, collate_fn_h5
    from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
except ModuleNotFoundError:  # compatibility with pre-rename formal checkpoints
    from mace_ictc.data import H5Dataset, collate_fn_h5
    from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


INTERVENTIONS = ("native", "zero", "permute", "global-shift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--elements", required=True, help="Comma-separated symbols")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=128,
        help="Maximum batches contributing to phase histograms",
    )
    parser.add_argument(
        "--intervention-batches",
        type=int,
        default=16,
        help="Leading batches used for force-bearing counterfactual forwards",
    )
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--global-shift", type=float, default=1.234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def move_batch(batch: tuple[Any, ...], device: torch.device, dtype: torch.dtype):
    if len(batch) == 11:
        batch = batch[:10]
    pos, atomic_numbers, batch_idx, force, target_e, src, dst, shifts, cell, stress = batch
    return (
        pos.to(device=device, dtype=dtype),
        atomic_numbers.to(device=device, dtype=torch.long),
        batch_idx.to(device=device, dtype=torch.long),
        force.to(device=device, dtype=dtype),
        target_e.to(device=device, dtype=dtype),
        src.to(device=device, dtype=torch.long),
        dst.to(device=device, dtype=torch.long),
        shifts.to(device=device, dtype=dtype),
        cell.to(device=device, dtype=dtype),
        stress.to(device=device, dtype=dtype),
    )


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def graph_sum(values: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = values.new_zeros(num_graphs)
    out.index_add_(0, batch_idx, values.reshape(-1))
    return out


def lookup_atomic_reference(
    atomic_numbers: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    matches = atomic_numbers[:, None] == keys.to(device=atomic_numbers.device)[None, :]
    if not bool(matches.any(dim=1).all().item()):
        missing = torch.unique(atomic_numbers[~matches.any(dim=1)]).tolist()
        raise ValueError(f"atomic reference energies are missing for Z={missing}")
    return matches.to(dtype=values.dtype) @ values.to(device=atomic_numbers.device)


def within_destination_permutation(edge_dst: torch.Tensor, layer: int) -> torch.Tensor:
    """Return a deterministic non-identity permutation inside each neighbor list."""
    permutation = torch.arange(edge_dst.numel(), device=edge_dst.device)
    if edge_dst.numel() == 0:
        return permutation
    _, counts = torch.unique_consecutive(edge_dst, return_counts=True)
    start = 0
    for count_tensor in counts:
        count = int(count_tensor.item())
        if count > 1:
            shift = 1 + (int(layer) % (count - 1))
            idx = torch.arange(start, start + count, device=edge_dst.device)
            permutation[start : start + count] = torch.roll(idx, shifts=shift)
        start += count
    return permutation


class ErrorAccumulator:
    def __init__(self) -> None:
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def add(self, error: torch.Tensor) -> None:
        values = error.detach().double().reshape(-1)
        self.sum_abs += float(values.abs().sum().item())
        self.sum_sq += float(values.square().sum().item())
        self.count += int(values.numel())

    def result(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "count": 0}
        return {
            "mae": self.sum_abs / self.count,
            "rmse": math.sqrt(self.sum_sq / self.count),
            "count": self.count,
        }


class PhaseHooks:
    """Capture phase/amplitude heads and optionally intervene on phase outputs."""

    def __init__(self, model: torch.nn.Module, global_shift: float) -> None:
        self.model = model
        self.global_shift = float(global_shift)
        self.mode = "native"
        self.edge_dst: torch.Tensor | None = None
        self.theta: dict[int, torch.Tensor] = {}
        self.amplitude_raw: dict[int, torch.Tensor] = {}
        self.interactions: dict[int, torch.nn.Module] = {}
        self.handles: list[Any] = []

        for layer, interaction in enumerate(model.interactions):
            phase_head = getattr(interaction, "phase_head", None)
            if phase_head is None:
                continue
            coefficient = str(getattr(interaction, "phase_coefficient", ""))
            if coefficient != "polar":
                raise RuntimeError(
                    "phase-angle diagnostics require polar coefficients; "
                    f"layer {layer} uses {coefficient!r}"
                )
            self.interactions[layer] = interaction
            self.handles.append(phase_head.register_forward_hook(self._phase_hook(layer)))
            amplitude_head = getattr(interaction, "phase_amplitude_head", None)
            if amplitude_head is not None:
                self.handles.append(
                    amplitude_head.register_forward_hook(self._amplitude_hook(layer))
                )
        if not self.interactions:
            raise RuntimeError("checkpoint contains no active polar phase head")

    def _phase_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor):
            self.theta[layer] = output.detach()
            if self.mode == "native":
                return output
            if self.mode == "zero":
                return torch.zeros_like(output)
            if self.mode == "global-shift":
                return output + self.global_shift
            if self.mode == "permute":
                if self.edge_dst is None or output.shape[0] != self.edge_dst.numel():
                    raise RuntimeError("phase permutation is missing the sorted edge destinations")
                permutation = within_destination_permutation(self.edge_dst, layer)
                return output.index_select(0, permutation)
            raise RuntimeError(f"unknown phase intervention {self.mode!r}")

        return hook

    def _amplitude_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor):
            self.amplitude_raw[layer] = output.detach()

        return hook

    def prepare(self, mode: str, edge_dst: torch.Tensor) -> None:
        if mode not in INTERVENTIONS:
            raise ValueError(f"unknown intervention {mode!r}")
        self.mode = mode
        self.edge_dst = edge_dst
        self.theta.clear()
        self.amplitude_raw.clear()

    def captured(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, theta in self.theta.items():
            raw = self.amplitude_raw.get(layer)
            amplitude = torch.ones_like(theta) if raw is None else F.softplus(raw)
            result[layer] = (theta, amplitude)
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class PhaseHistogramAccumulator:
    def __init__(self, bins: int) -> None:
        if bins < 12:
            raise ValueError("at least 12 angular bins are required")
        self.edges = np.linspace(-math.pi, math.pi, bins + 1, dtype=np.float64)
        self.absolute: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(bins, dtype=np.float64)
        )
        self.absolute_amplitude: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(bins, dtype=np.float64)
        )
        self.relative: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(bins, dtype=np.float64)
        )
        self.relative_amplitude: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(bins, dtype=np.float64)
        )
        self.stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "edge_count": 0.0,
                "pair_count": 0.0,
                "amplitude_sum": 0.0,
                "pair_amplitude_sum": 0.0,
                "constructive_kernel_mass": 0.0,
                "destructive_kernel_mass": 0.0,
                "weighted_cosine_sum": 0.0,
                "weighted_sine_sum": 0.0,
            }
        )

    def add(
        self,
        *,
        layer: int,
        interaction: torch.nn.Module,
        theta: torch.Tensor,
        amplitude: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> None:
        heads = int(interaction.phase_heads)
        angular = bool(interaction.phase_angular_channels)
        expected = heads * (int(interaction.target_lmax) + 1 if angular else 1)
        if theta.ndim != 2 or theta.shape[1] != expected:
            raise RuntimeError(
                f"layer {layer} phase shape {tuple(theta.shape)} does not match {expected} channels"
            )
        dst_cpu = edge_dst.detach().cpu().numpy()
        if dst_cpu.size:
            boundaries = np.concatenate(
                ([0], np.flatnonzero(dst_cpu[1:] != dst_cpu[:-1]) + 1, [dst_cpu.size])
            )
        else:
            boundaries = np.asarray([0], dtype=np.int64)
        pair_rows: list[np.ndarray] = []
        pair_cols: list[np.ndarray] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            count = int(end - start)
            if count < 2:
                continue
            row, col = np.triu_indices(count, k=1)
            pair_rows.append(row + int(start))
            pair_cols.append(col + int(start))
        if pair_rows:
            pair_row = np.concatenate(pair_rows)
            pair_col = np.concatenate(pair_cols)
        else:
            pair_row = np.empty(0, dtype=np.int64)
            pair_col = np.empty(0, dtype=np.int64)

        wrapped = wrap_angle(theta).detach().double().cpu().numpy()
        amp = amplitude.detach().double().cpu().numpy()
        for channel in range(expected):
            if angular:
                ell = channel // heads
                head = channel % heads
                key = f"layer{layer}.L{ell}.H{head}"
            else:
                key = f"layer{layer}.shared.H{channel}"
            angles = wrapped[:, channel]
            weights = amp[:, channel]
            self.absolute[key] += np.histogram(angles, bins=self.edges)[0]
            self.absolute_amplitude[key] += np.histogram(
                angles, bins=self.edges, weights=weights
            )[0]
            stats = self.stats[key]
            stats["edge_count"] += float(angles.size)
            stats["amplitude_sum"] += float(weights.sum())

            if pair_row.size:
                delta = np.angle(np.exp(1j * (angles[pair_row] - angles[pair_col])))
                pair_amp = weights[pair_row] * weights[pair_col]
                cosine = np.cos(delta)
                sine = np.sin(delta)
                self.relative[key] += np.histogram(delta, bins=self.edges)[0]
                self.relative_amplitude[key] += np.histogram(
                    delta, bins=self.edges, weights=pair_amp
                )[0]
                stats["pair_count"] += float(delta.size)
                stats["pair_amplitude_sum"] += float(pair_amp.sum())
                stats["constructive_kernel_mass"] += float(
                    np.sum(pair_amp * np.clip(cosine, 0.0, None))
                )
                stats["destructive_kernel_mass"] += float(
                    np.sum(pair_amp * np.clip(-cosine, 0.0, None))
                )
                stats["weighted_cosine_sum"] += float(np.sum(pair_amp * cosine))
                stats["weighted_sine_sum"] += float(np.sum(pair_amp * sine))

    @staticmethod
    def _normalize(values: np.ndarray) -> list[float]:
        total = float(values.sum())
        if total <= 0.0:
            return [0.0 for _ in values]
        return (values / total).tolist()

    def payload(self) -> dict[str, Any]:
        summaries: dict[str, dict[str, float]] = {}
        for key, raw in self.stats.items():
            pair_amp = max(raw["pair_amplitude_sum"], 1.0e-30)
            kernel_abs = max(
                raw["constructive_kernel_mass"] + raw["destructive_kernel_mass"],
                1.0e-30,
            )
            summaries[key] = {
                **raw,
                "weighted_mean_cos_delta": raw["weighted_cosine_sum"] / pair_amp,
                "weighted_mean_sin_delta": raw["weighted_sine_sum"] / pair_amp,
                "destructive_kernel_fraction": raw["destructive_kernel_mass"] / kernel_abs,
                "constructive_kernel_fraction": raw["constructive_kernel_mass"] / kernel_abs,
            }
        keys = sorted(self.stats)
        return {
            "bin_edges_radians": self.edges.tolist(),
            "channels": {
                key: {
                    "absolute_probability": self._normalize(self.absolute[key]),
                    "absolute_amplitude_weighted_probability": self._normalize(
                        self.absolute_amplitude[key]
                    ),
                    "relative_probability": self._normalize(self.relative[key]),
                    "relative_amplitude_weighted_probability": self._normalize(
                        self.relative_amplitude[key]
                    ),
                    "summary": summaries[key],
                }
                for key in keys
            },
        }


def model_energy_and_force(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    hooks: PhaseHooks,
    mode: str,
    atomic_energy_keys: torch.Tensor,
    atomic_energy_values: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
]:
    pos, atomic_numbers, batch_idx, _, _, src, dst, shifts, cell, _ = batch
    used_dst = dst if bool(getattr(model, "preserve_edge_order", False)) else dst[torch.argsort(dst)]
    hooks.prepare(mode, used_dst)
    differentiable_pos = pos.detach().clone().requires_grad_(True)
    atomic_energy = model(
        differentiable_pos,
        atomic_numbers,
        batch_idx,
        src,
        dst,
        shifts,
        cell,
    )
    if isinstance(atomic_energy, tuple):
        atomic_energy = atomic_energy[0]
    force = -torch.autograd.grad(
        atomic_energy.sum(), differentiable_pos, create_graph=False, retain_graph=False
    )[0]
    num_graphs = int(batch_idx.max().item()) + 1
    interaction_energy = graph_sum(atomic_energy, batch_idx, num_graphs)
    atomic_reference = lookup_atomic_reference(
        atomic_numbers, atomic_energy_keys, atomic_energy_values
    )
    total_energy = interaction_energy + graph_sum(atomic_reference, batch_idx, num_graphs)
    return (
        total_energy.detach(),
        interaction_energy.detach(),
        force.detach(),
        hooks.captured(),
        used_dst,
    )


def capture_phases_without_forces(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    hooks: PhaseHooks,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    pos, atomic_numbers, batch_idx, _, _, src, dst, shifts, cell, _ = batch
    used_dst = dst if bool(getattr(model, "preserve_edge_order", False)) else dst[torch.argsort(dst)]
    hooks.prepare("native", used_dst)
    with torch.no_grad():
        model(pos, atomic_numbers, batch_idx, src, dst, shifts, cell)
    return hooks.captured(), used_dst


def add_intervention_metrics(
    accumulators: dict[str, dict[str, ErrorAccumulator]],
    *,
    mode: str,
    energy: torch.Tensor,
    force: torch.Tensor,
    target_energy: torch.Tensor,
    target_force: torch.Tensor,
    batch_idx: torch.Tensor,
    interaction_energy: torch.Tensor,
    native_interaction_energy: torch.Tensor,
    native_force: torch.Tensor,
) -> None:
    num_graphs = target_energy.numel()
    atom_counts = torch.bincount(batch_idx, minlength=num_graphs).to(dtype=energy.dtype)
    accumulators[mode]["energy_error_per_atom"].add(
        (energy - target_energy.reshape(-1)) / atom_counts
    )
    accumulators[mode]["force_error"].add(force - target_force)
    accumulators[mode]["energy_change_per_atom"].add(
        (interaction_energy - native_interaction_energy) / atom_counts
    )
    accumulators[mode]["force_change"].add(force - native_force)


def plot_payload(payload: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional on clusters
        print(f"plotting skipped: {exc}", flush=True)
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    histogram = payload["phase_histograms"]
    edges = np.asarray(histogram["bin_edges_radians"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    channels = histogram["channels"]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, max(len(channels), 2)))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.75), constrained_layout=True)

    for color, (key, values) in zip(colors, sorted(channels.items())):
        axes[0].plot(
            centers,
            values["absolute_probability"],
            color=color,
            lw=1.35,
            label=key.replace("layer", "layer ").replace(".", " / "),
        )
        axes[1].plot(
            centers,
            values["relative_amplitude_weighted_probability"],
            color=color,
            lw=1.35,
        )

    for axis in axes[:2]:
        axis.set_xlim(-math.pi, math.pi)
        axis.set_xticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
        axis.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        axis.set_ylabel("Probability")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Absolute phase (descriptive)", loc="left", fontweight="semibold")
    axes[0].set_xlabel(r"$\theta$")
    axes[0].legend(frameon=False, ncol=2, handlelength=1.6, columnspacing=0.8)
    axes[1].axvspan(-math.pi, -math.pi / 2, color="#D55E00", alpha=0.08, lw=0)
    axes[1].axvspan(math.pi / 2, math.pi, color="#D55E00", alpha=0.08, lw=0)
    axes[1].set_title("Within-environment relative phase", loc="left", fontweight="semibold")
    axes[1].set_xlabel(r"$\Delta\theta_{jk}$")

    labels = []
    constructive = []
    destructive = []
    for key, values in sorted(channels.items()):
        labels.append(key.replace("layer", "Lyr ").replace(".", "/"))
        summary = values["summary"]
        constructive.append(100.0 * summary["constructive_kernel_fraction"])
        destructive.append(100.0 * summary["destructive_kernel_fraction"])
    y = np.arange(len(labels))
    axes[2].barh(y, constructive, color="#0072B2", height=0.68, label="constructive")
    axes[2].barh(
        y,
        destructive,
        left=constructive,
        color="#D55E00",
        height=0.68,
        label="destructive",
    )
    axes[2].set_yticks(y, labels)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel(r"$|a_j a_k\cos\Delta\theta|$ mass (%)")
    axes[2].set_title("Signed pair-kernel mass", loc="left", fontweight="semibold")
    axes[2].spines[["top", "right", "left"]].set_visible(False)
    axes[2].legend(frameon=False, ncol=2, loc="lower right")

    dataset = payload["dataset_label"]
    fig.suptitle(f"CHORUS phase interference — {dataset}", fontsize=11, fontweight="semibold")
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(output_dir / f"phase_interference.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    if args.intervention_batches < 0:
        raise ValueError("--intervention-batches must be non-negative")

    device = torch.device(args.device)
    elements = [item.strip() for item in args.elements.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    deployed = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=args.checkpoint,
        element_types=elements,
        device=str(device),
    )
    model = deployed.wrapper.model
    atomic_energy_keys = deployed.wrapper.atomic_energy_keys
    atomic_energy_values = deployed.wrapper.atomic_energy_values
    model.eval()
    model.skip_input_validation = False
    dtype = next(model.parameters()).dtype
    hooks = PhaseHooks(model, global_shift=args.global_shift)
    histograms = PhaseHistogramAccumulator(args.bins)

    dataset = H5Dataset(prefix=args.split, data_dir=args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_h5,
        num_workers=0,
    )
    metric_names = (
        "energy_error_per_atom",
        "force_error",
        "energy_change_per_atom",
        "force_change",
    )
    accumulators: dict[str, dict[str, ErrorAccumulator]] = {
        mode: {name: ErrorAccumulator() for name in metric_names} for mode in INTERVENTIONS
    }
    batches_processed = 0
    intervention_batches_processed = 0
    try:
        for batch_number, cpu_batch in enumerate(loader):
            if batch_number >= args.max_batches:
                break
            batch = move_batch(cpu_batch, device, dtype)
            pos, _, batch_idx, target_force, target_energy, _, dst, _, _, _ = batch
            if batch_number < args.intervention_batches:
                (
                    native_energy,
                    native_interaction_energy,
                    native_force,
                    captured,
                    used_dst,
                ) = model_energy_and_force(
                    model,
                    batch,
                    hooks,
                    "native",
                    atomic_energy_keys,
                    atomic_energy_values,
                )
            else:
                captured, used_dst = capture_phases_without_forces(model, batch, hooks)
            for layer, (theta, amplitude) in captured.items():
                histograms.add(
                    layer=layer,
                    interaction=hooks.interactions[layer],
                    theta=theta,
                    amplitude=amplitude,
                    edge_dst=used_dst,
                )
            if batch_number < args.intervention_batches:
                add_intervention_metrics(
                    accumulators,
                    mode="native",
                    energy=native_energy,
                    force=native_force,
                    target_energy=target_energy,
                    target_force=target_force,
                    batch_idx=batch_idx,
                    interaction_energy=native_interaction_energy,
                    native_interaction_energy=native_interaction_energy,
                    native_force=native_force,
                )
                for mode in INTERVENTIONS[1:]:
                    energy, interaction_energy, force, _, _ = model_energy_and_force(
                        model,
                        batch,
                        hooks,
                        mode,
                        atomic_energy_keys,
                        atomic_energy_values,
                    )
                    add_intervention_metrics(
                        accumulators,
                        mode=mode,
                        energy=energy,
                        force=force,
                        target_energy=target_energy,
                        target_force=target_force,
                        batch_idx=batch_idx,
                        interaction_energy=interaction_energy,
                        native_interaction_energy=native_interaction_energy,
                        native_force=native_force,
                    )
                intervention_batches_processed += 1
            batches_processed += 1
            print(
                f"batch={batch_number + 1}/{min(len(loader), args.max_batches)} "
                f"nodes={pos.shape[0]} edges={dst.numel()}",
                flush=True,
            )
    finally:
        hooks.close()

    if batches_processed == 0:
        raise RuntimeError("no batches were processed")
    interventions = {
        mode: {name: accumulator.result() for name, accumulator in metrics.items()}
        for mode, metrics in accumulators.items()
    }
    payload = {
        "schema_version": 1,
        "dataset_label": args.dataset_label or Path(args.data_dir).name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "split": args.split,
        "elements": elements,
        "device": str(device),
        "dtype": str(dtype),
        "num_batches": batches_processed,
        "num_intervention_batches": intervention_batches_processed,
        "phase_scope": str(getattr(model, "ictd_fix_phase_scope", "unknown")),
        "phase_density_pairs": str(
            getattr(model, "ictd_fix_phase_density_pairs", "unknown")
        ),
        "phase_density_rank": int(getattr(model, "ictd_fix_phase_density_rank", 0)),
        "active_phase_layers": sorted(hooks.interactions),
        "definitions": {
            "absolute_phase": "wrapped learned edge phase; descriptive and gauge dependent",
            "relative_phase": "wrapped theta_ij - theta_ik for unordered edge pairs sharing destination i",
            "pair_weight": "a_ij a_ik using the learned positive amplitudes",
            "constructive_kernel_mass": "sum a_ij a_ik max(cos Delta theta, 0)",
            "destructive_kernel_mass": "sum a_ij a_ik max(-cos Delta theta, 0)",
            "permute": "deterministic within-destination phase permutation; local phase marginals are unchanged",
            "global_shift": f"theta -> theta + {args.global_shift}; U(1) invariance sanity check",
        },
        "phase_histograms": histograms.payload(),
        "interventions": interventions,
    }
    output_json = output_dir / "phase_interference.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    plot_payload(payload, output_dir)
    print(f"wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
