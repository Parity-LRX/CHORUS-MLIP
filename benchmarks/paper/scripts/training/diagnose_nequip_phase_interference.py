#!/usr/bin/env python3
"""Measure learned CHORUS phase organization in a trained NequIP model.

The absolute edge phase is gauge dependent and is retained only as a
descriptive diagnostic.  Mechanism claims use the relative phase between two
edges entering the same atom and three same-checkpoint interventions:

* ``zero`` removes all learned relative phases while retaining amplitudes;
* ``permute`` preserves each atom's phase marginal but breaks phase--edge
  assignment;
* ``global-shift`` adds a common phase and checks U(1) invariance.

The script loads the validation-Force-MAE-selected ``best_model.pth`` from a
NequIP training directory.  Histogram frames are spread deterministically
over the full validation set; interventions use the leading sampled batches.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nequip.data import AtomicData, AtomicDataDict, Collater, dataset_from_config
from nequip.scripts.train import check_code_version, default_config
from nequip.train import Trainer
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options


INTERVENTIONS = ("native", "zero", "permute", "global-shift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--model-name", default="best_model.pth")
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--intervention-batches", type=int, default=8)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--global-shift", type=float, default=1.234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def deterministic_within_atom_permutation(
    edge_dst: torch.Tensor, layer: int
) -> torch.Tensor:
    """Cyclically permute edges inside each receiving atom."""
    permutation = torch.arange(edge_dst.numel(), device=edge_dst.device)
    for destination in torch.unique(edge_dst, sorted=True):
        indices = torch.nonzero(edge_dst == destination, as_tuple=False).reshape(-1)
        count = int(indices.numel())
        if count > 1:
            shift = 1 + int(layer) % (count - 1)
            permutation[indices] = torch.roll(indices, shifts=shift)
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


class NequIPPhaseHooks:
    """Capture and intervene on the polar output of each CHORUS phase MLP."""

    def __init__(self, model: torch.nn.Module, global_shift: float) -> None:
        self.global_shift = float(global_shift)
        self.mode = "native"
        self.edge_dst: torch.Tensor | None = None
        self.captured_raw: dict[int, torch.Tensor] = {}
        self.chorus_modules: dict[int, torch.nn.Module] = {}
        self.handles: list[Any] = []

        for name, module in model.named_modules():
            phase_network = getattr(module, "phase_network", None)
            if phase_network is None or not hasattr(module, "charged_edge_messages"):
                continue
            match = re.search(r"layer(\d+)_convnet", name)
            if match is None:
                raise RuntimeError(f"cannot infer CHORUS layer from module name {name!r}")
            layer = int(match.group(1))
            self.chorus_modules[layer] = module
            self.handles.append(
                phase_network.register_forward_hook(self._phase_network_hook(layer))
            )
        if not self.chorus_modules:
            raise RuntimeError("loaded NequIP model contains no active CHORUS module")

    def _phase_network_hook(self, layer: int):
        def hook(_module, _inputs, output: torch.Tensor) -> torch.Tensor:
            if output.ndim != 2 or output.shape[-1] != 2:
                raise RuntimeError(
                    f"layer {layer} phase-network output has shape {tuple(output.shape)}"
                )
            self.captured_raw[layer] = output.detach()
            if self.mode == "native":
                return output

            amplitude_raw = output[:, 0]
            phase_raw = output[:, 1]
            if self.mode == "zero":
                phase_raw = torch.zeros_like(phase_raw)
            elif self.mode == "permute":
                if self.edge_dst is None or self.edge_dst.numel() != output.shape[0]:
                    raise RuntimeError("within-atom permutation is missing edge destinations")
                permutation = deterministic_within_atom_permutation(
                    self.edge_dst, layer
                )
                phase_raw = phase_raw.index_select(0, permutation)
            elif self.mode == "global-shift":
                theta = math.pi * torch.tanh(phase_raw)
                shifted = wrap_angle(theta + self.global_shift)
                epsilon = 16.0 * torch.finfo(output.dtype).eps
                scaled = (shifted / math.pi).clamp(
                    min=-1.0 + epsilon, max=1.0 - epsilon
                )
                phase_raw = torch.atanh(scaled)
            else:
                raise RuntimeError(f"unknown phase intervention {self.mode!r}")
            return torch.stack((amplitude_raw, phase_raw), dim=-1)

        return hook

    def prepare(self, mode: str, edge_dst: torch.Tensor) -> None:
        if mode not in INTERVENTIONS:
            raise ValueError(f"unknown intervention {mode!r}")
        self.mode = mode
        self.edge_dst = edge_dst
        self.captured_raw.clear()

    def captured(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, raw in self.captured_raw.items():
            amplitude = torch.nn.functional.softplus(raw[:, 0]) / math.log(2.0)
            theta = math.pi * torch.tanh(raw[:, 1])
            result[layer] = (theta[:, None], amplitude[:, None])
        return result

    def phase_parameter_norms(self) -> dict[str, dict[str, float]]:
        result = {}
        for layer, module in sorted(self.chorus_modules.items()):
            final = module.phase_network[-1]
            result[f"layer{layer}"] = {
                "amplitude_weight_l2": float(final.weight[0].detach().norm().item()),
                "phase_weight_l2": float(final.weight[1].detach().norm().item()),
                "phase_weight_max_abs": float(
                    final.weight[1].detach().abs().max().item()
                ),
                "phase_bias": float(final.bias[1].detach().item()),
            }
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
        theta: torch.Tensor,
        amplitude: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> None:
        if theta.ndim != 2 or theta.shape[1] != 1:
            raise RuntimeError(
                f"layer {layer} NequIP phase shape must be [edges, 1], got {tuple(theta.shape)}"
            )
        destinations = edge_dst.detach().cpu().numpy()
        order = np.argsort(destinations, kind="stable")
        destinations = destinations[order]
        angles = wrap_angle(theta[:, 0]).detach().double().cpu().numpy()[order]
        weights = amplitude[:, 0].detach().double().cpu().numpy()[order]
        key = f"layer{layer}.shared.H0"
        self.absolute[key] += np.histogram(angles, bins=self.edges)[0]
        self.absolute_amplitude[key] += np.histogram(
            angles, bins=self.edges, weights=weights
        )[0]
        stats = self.stats[key]
        stats["edge_count"] += float(angles.size)
        stats["amplitude_sum"] += float(weights.sum())

        boundaries = np.concatenate(
            (
                [0],
                np.flatnonzero(destinations[1:] != destinations[:-1]) + 1,
                [destinations.size],
            )
        )
        pair_rows: list[np.ndarray] = []
        pair_cols: list[np.ndarray] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            count = int(end - start)
            if count < 2:
                continue
            row, column = np.triu_indices(count, k=1)
            pair_rows.append(row + int(start))
            pair_cols.append(column + int(start))
        if not pair_rows:
            return
        pair_row = np.concatenate(pair_rows)
        pair_column = np.concatenate(pair_cols)
        delta = np.angle(
            np.exp(1j * (angles[pair_row] - angles[pair_column]))
        )
        pair_amplitude = weights[pair_row] * weights[pair_column]
        cosine = np.cos(delta)
        sine = np.sin(delta)
        self.relative[key] += np.histogram(delta, bins=self.edges)[0]
        self.relative_amplitude[key] += np.histogram(
            delta, bins=self.edges, weights=pair_amplitude
        )[0]
        stats["pair_count"] += float(delta.size)
        stats["pair_amplitude_sum"] += float(pair_amplitude.sum())
        stats["constructive_kernel_mass"] += float(
            np.sum(pair_amplitude * np.clip(cosine, 0.0, None))
        )
        stats["destructive_kernel_mass"] += float(
            np.sum(pair_amplitude * np.clip(-cosine, 0.0, None))
        )
        stats["weighted_cosine_sum"] += float(np.sum(pair_amplitude * cosine))
        stats["weighted_sine_sum"] += float(np.sum(pair_amplitude * sine))

    @staticmethod
    def _normalize(values: np.ndarray) -> list[float]:
        total = float(values.sum())
        if total <= 0.0:
            return [0.0 for _ in values]
        return (values / total).tolist()

    def payload(self) -> dict[str, Any]:
        channels = {}
        for key in sorted(self.stats):
            raw = self.stats[key]
            pair_amplitude = max(raw["pair_amplitude_sum"], 1.0e-30)
            absolute_kernel = max(
                raw["constructive_kernel_mass"]
                + raw["destructive_kernel_mass"],
                1.0e-30,
            )
            summary = {
                **raw,
                "weighted_mean_cos_delta": raw["weighted_cosine_sum"]
                / pair_amplitude,
                "weighted_mean_sin_delta": raw["weighted_sine_sum"]
                / pair_amplitude,
                "destructive_kernel_fraction": raw["destructive_kernel_mass"]
                / absolute_kernel,
                "constructive_kernel_fraction": raw["constructive_kernel_mass"]
                / absolute_kernel,
            }
            channels[key] = {
                "absolute_probability": self._normalize(self.absolute[key]),
                "absolute_amplitude_weighted_probability": self._normalize(
                    self.absolute_amplitude[key]
                ),
                "relative_probability": self._normalize(self.relative[key]),
                "relative_amplitude_weighted_probability": self._normalize(
                    self.relative_amplitude[key]
                ),
                "summary": summary,
            }
        return {"bin_edges_radians": self.edges.tolist(), "channels": channels}


def load_model_and_validation_dataset(
    train_dir: Path,
    model_name: str,
    dataset_config_path: Path | None,
    device: torch.device,
):
    config_path = train_dir / "config.yaml"
    global_config = Config.from_file(str(config_path), defaults=default_config)
    _set_global_options(global_config)
    check_code_version(global_config)
    model, model_config = Trainer.load_model_from_training_session(
        traindir=train_dir, model_name=model_name
    )
    model = model.to(device).eval()

    source = dataset_config_path or config_path
    dataset_config = Config.from_file(
        str(source), defaults={"r_max": model_config["r_max"]}
    )
    try:
        dataset = dataset_from_config(dataset_config, prefix="validation_dataset")
        indexes = torch.arange(len(dataset), dtype=torch.long)
        validation_source = "validation_dataset"
    except KeyError:
        dataset = dataset_from_config(dataset_config)
        trainer_state = torch.load(
            train_dir / "trainer.pth", map_location="cpu", weights_only=False
        )
        indexes = torch.as_tensor(trainer_state["val_idcs"], dtype=torch.long)
        validation_source = "trainer.val_idcs"
    return model, dataset, indexes, validation_source


def clone_atomic_data_dict(batch) -> dict[str, Any]:
    data = AtomicData.to_AtomicDataDict(batch)
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def run_forward(
    model: torch.nn.Module,
    batch,
    hooks: NequIPPhaseHooks,
    mode: str,
) -> tuple[dict[str, Any], dict[int, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    data = clone_atomic_data_dict(batch)
    edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]
    hooks.prepare(mode, edge_dst)
    output = model(data)
    return output, hooks.captured(), edge_dst


def add_intervention_metrics(
    accumulators: dict[str, dict[str, ErrorAccumulator]],
    *,
    mode: str,
    output: dict[str, Any],
    target: dict[str, Any],
    native_output: dict[str, Any],
) -> None:
    predicted_energy = output[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)
    target_energy = target[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)
    batch_index = target[AtomicDataDict.BATCH_KEY]
    atom_counts = torch.bincount(
        batch_index, minlength=target_energy.numel()
    ).to(dtype=predicted_energy.dtype)
    predicted_force = output[AtomicDataDict.FORCE_KEY]
    target_force = target[AtomicDataDict.FORCE_KEY]
    native_energy = native_output[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)
    native_force = native_output[AtomicDataDict.FORCE_KEY]
    accumulators[mode]["energy_error_per_atom"].add(
        (predicted_energy - target_energy) / atom_counts
    )
    accumulators[mode]["force_error"].add(predicted_force - target_force)
    accumulators[mode]["energy_change_per_atom"].add(
        (predicted_energy - native_energy) / atom_counts
    )
    accumulators[mode]["force_change"].add(predicted_force - native_force)


def plot_payload(payload: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plotting skipped: {exc}", flush=True)
        return

    blue = "#205381"
    orange = "#DC7520"
    ink = "#252A30"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.3,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    histogram = payload["phase_histograms"]
    edges = np.asarray(histogram["bin_edges_radians"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    channels = histogram["channels"]
    colors = plt.get_cmap("Blues")(
        np.linspace(0.45, 0.9, max(len(channels), 2))
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=True)
    for color, (key, values) in zip(colors, sorted(channels.items())):
        label = key.split(".")[0].replace("layer", "layer ")
        axes[0].plot(
            centers,
            values["relative_amplitude_weighted_probability"],
            color=color,
            lw=1.45,
            label=label,
        )
    axes[0].axvspan(-math.pi, -math.pi / 2, color=orange, alpha=0.1, lw=0)
    axes[0].axvspan(math.pi / 2, math.pi, color=orange, alpha=0.1, lw=0)
    axes[0].set_xlim(-math.pi, math.pi)
    axes[0].set_xticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
    axes[0].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axes[0].set_xlabel(r"Relative phase $\Delta\theta_{jk}$")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Relative-phase distribution", loc="left")
    axes[0].legend(frameon=False)

    labels = []
    destructive = []
    for key, values in sorted(channels.items()):
        labels.append(key.split(".")[0].replace("layer", "L"))
        destructive.append(
            100.0 * values["summary"]["destructive_kernel_fraction"]
        )
    y = np.arange(len(labels))
    axes[1].barh(y, destructive, color=orange, height=0.58)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, max(1.0, 1.12 * max(destructive, default=0.0)))
    axes[1].set_xlabel("Destructive mass (%)")
    axes[1].set_title("Signed pair kernel", loc="left")

    ratios = []
    for mode in ("zero", "permute"):
        ratios.append(
            payload["interventions"][mode]["force_error"]["mae"]
            / payload["interventions"]["native"]["force_error"]["mae"]
        )
    axes[2].bar(
        np.arange(2), ratios, color=[blue, orange], width=0.58
    )
    axes[2].axhline(1.0, color=ink, lw=0.75)
    axes[2].set_xticks(
        np.arange(2), [r"$\theta\!\to\!0$", "permute"]
    )
    axes[2].set_ylabel("Force MAE / native")
    axes[2].set_title("Same-checkpoint intervention", loc="left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(payload["dataset_label"], x=0.01, ha="left", fontsize=9.8)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"nequip_phase_interference.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.intervention_batches < 0:
        raise ValueError("--intervention-batches must be non-negative")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(args.train_dir)

    model, dataset, validation_indexes, validation_source = (
        load_model_and_validation_dataset(
            train_dir=train_dir,
            model_name=args.model_name,
            dataset_config_path=(
                None if args.dataset_config is None else Path(args.dataset_config)
            ),
            device=device,
        )
    )
    sample_count = min(int(validation_indexes.numel()), int(args.max_frames))
    positions = np.linspace(
        0, int(validation_indexes.numel()) - 1, num=sample_count, dtype=np.int64
    )
    sampled_indexes = validation_indexes[torch.as_tensor(positions)].unique(sorted=True)
    collater = Collater.for_dataset(dataset, exclude_keys=[])
    hooks = NequIPPhaseHooks(model, global_shift=args.global_shift)
    histograms = PhaseHistogramAccumulator(args.bins)
    metric_names = (
        "energy_error_per_atom",
        "force_error",
        "energy_change_per_atom",
        "force_change",
    )
    accumulators = {
        mode: {name: ErrorAccumulator() for name in metric_names}
        for mode in INTERVENTIONS
    }

    batches_processed = 0
    intervention_batches_processed = 0
    try:
        for start in range(0, sampled_indexes.numel(), args.batch_size):
            indexes = sampled_indexes[start : start + args.batch_size]
            batch = collater.collate([dataset[int(index)] for index in indexes])
            batch = batch.to(device)
            target = AtomicData.to_AtomicDataDict(batch)
            native_output, captured, edge_dst = run_forward(
                model, batch, hooks, "native"
            )
            for layer, (theta, amplitude) in captured.items():
                histograms.add(
                    layer=layer,
                    theta=theta,
                    amplitude=amplitude,
                    edge_dst=edge_dst,
                )
            if batches_processed < args.intervention_batches:
                add_intervention_metrics(
                    accumulators,
                    mode="native",
                    output=native_output,
                    target=target,
                    native_output=native_output,
                )
                for mode in INTERVENTIONS[1:]:
                    output, _, _ = run_forward(model, batch, hooks, mode)
                    add_intervention_metrics(
                        accumulators,
                        mode=mode,
                        output=output,
                        target=target,
                        native_output=native_output,
                    )
                intervention_batches_processed += 1
            batches_processed += 1
            print(
                f"batch={batches_processed} frames={int(indexes.numel())} "
                f"edges={int(edge_dst.numel())}",
                flush=True,
            )
    finally:
        phase_parameter_norms = hooks.phase_parameter_norms()
        hooks.close()

    interventions = {
        mode: {name: accumulator.result() for name, accumulator in metrics.items()}
        for mode, metrics in accumulators.items()
    }
    payload = {
        "schema_version": 1,
        "base_model": "NequIP-SH",
        "dataset_label": args.dataset_label or train_dir.parent.name,
        "train_dir": str(train_dir.resolve()),
        "model_name": args.model_name,
        "checkpoint_selection": "validation Force MAE",
        "test_used_for_selection": False,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "validation_source": validation_source,
        "validation_size": int(validation_indexes.numel()),
        "sampled_frame_count": int(sampled_indexes.numel()),
        "sampling": "deterministic evenly spaced validation frames",
        "num_batches": batches_processed,
        "num_intervention_batches": intervention_batches_processed,
        "active_phase_layers": sorted(hooks.chorus_modules),
        "phase_parameter_norms": phase_parameter_norms,
        "definitions": {
            "absolute_phase": "wrapped learned edge phase; descriptive and gauge dependent",
            "relative_phase": "wrapped theta_ij - theta_ik for unordered edges sharing destination i",
            "pair_weight": "a_ij a_ik using learned positive amplitudes",
            "constructive_kernel_mass": "sum a_ij a_ik max(cos Delta theta, 0)",
            "destructive_kernel_mass": "sum a_ij a_ik max(-cos Delta theta, 0)",
            "permute": "deterministic within-destination phase permutation preserving local phase marginals",
            "global_shift": f"theta -> theta + {args.global_shift}; U(1) invariance check",
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
