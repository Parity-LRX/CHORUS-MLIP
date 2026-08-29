#!/usr/bin/env python3
"""Paired, same-checkpoint interventions on learned CHORUS edge phases.

The trained checkpoint, positive amplitudes, edge messages, Hermitian closure,
and all non-phase parameters are held fixed.  Every held-out frame is evaluated
under the factual learned phase and under controlled phase interventions.  The
wide Parquet outputs preserve pairing at the atom and frame levels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chorus.data import H5Dataset, collate_fn_h5
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


DOSES = (0.25, 0.50, 0.75)
GLOBAL_SHIFT = 1.234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--elements", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--test-label", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--permutation-repeats", type=int, default=4)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Debug/smoke limit only; omit for every formal held-out evaluation.",
    )
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


def graph_sum(values: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
    result = values.new_zeros(num_graphs)
    result.index_add_(0, batch_idx, values.reshape(-1))
    return result


def lookup_atomic_reference(
    atomic_numbers: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    matches = atomic_numbers[:, None] == keys.to(atomic_numbers.device)[None, :]
    if not bool(matches.any(dim=1).all().item()):
        missing = torch.unique(atomic_numbers[~matches.any(dim=1)]).tolist()
        raise ValueError(f"atomic reference energies are missing for Z={missing}")
    return matches.to(dtype=values.dtype) @ values.to(atomic_numbers.device)


def circular_centre(theta: torch.Tensor, edge_dst: torch.Tensor) -> torch.Tensor:
    """Return the per-centre circular mean for every phase channel."""
    if edge_dst.numel() == 0:
        return theta
    num_nodes = int(edge_dst.max().item()) + 1
    sin_sum = theta.new_zeros(num_nodes, theta.shape[1])
    cos_sum = theta.new_zeros(num_nodes, theta.shape[1])
    sin_sum.index_add_(0, edge_dst, torch.sin(theta))
    cos_sum.index_add_(0, edge_dst, torch.cos(theta))
    centre = torch.atan2(sin_sum, cos_sum)
    return centre.index_select(0, edge_dst)


def deterministic_local_permutation(
    edge_dst: torch.Tensor,
    *,
    layer: int,
    repeat: int,
    batch_number: int,
) -> torch.Tensor:
    """Random-looking but exactly reproducible permutations within centres."""
    permutation = torch.arange(edge_dst.numel(), device=edge_dst.device)
    if edge_dst.numel() == 0:
        return permutation
    dst_cpu = edge_dst.detach().cpu()
    _, counts = torch.unique_consecutive(dst_cpu, return_counts=True)
    start = 0
    for group_number, count_tensor in enumerate(counts):
        count = int(count_tensor.item())
        if count > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                20260825
                + 1_000_003 * int(repeat)
                + 10_007 * int(layer)
                + 101 * int(batch_number)
                + int(group_number)
            )
            order = torch.randperm(count, generator=generator)
            if bool(torch.equal(order, torch.arange(count))):
                order = torch.roll(order, shifts=1)
            permutation[start : start + count] = order.to(edge_dst.device) + start
        start += count
    return permutation


class PhaseInterventionHooks:
    def __init__(self, model: torch.nn.Module) -> None:
        self.mode = "native"
        self.edge_dst: torch.Tensor | None = None
        self.batch_number = 0
        self.handles: list[Any] = []
        self.layers: list[int] = []
        for layer, interaction in enumerate(model.interactions):
            head = getattr(interaction, "phase_head", None)
            if head is None:
                continue
            coefficient = str(getattr(interaction, "phase_coefficient", ""))
            if coefficient != "polar":
                raise RuntimeError(
                    f"phase interventions require polar coefficients, got {coefficient!r}"
                )
            self.layers.append(layer)
            self.handles.append(head.register_forward_hook(self._hook(layer)))
        if not self.layers:
            raise RuntimeError("checkpoint has no active learned phase head")

    def prepare(self, mode: str, edge_dst: torch.Tensor, batch_number: int) -> None:
        self.mode = mode
        self.edge_dst = edge_dst
        self.batch_number = int(batch_number)

    def _hook(self, layer: int):
        def hook(_module, _inputs, theta: torch.Tensor):
            if self.mode == "native":
                return theta
            if self.edge_dst is None or theta.shape[0] != self.edge_dst.numel():
                raise RuntimeError("phase intervention is missing aligned edge destinations")
            centre = circular_centre(theta, self.edge_dst)
            if self.mode == "aligned":
                return centre
            if self.mode == "common_shift":
                return theta + GLOBAL_SHIFT
            if self.mode.startswith("dose_"):
                dose = float(self.mode.split("_", 1)[1])
                relative = torch.atan2(
                    torch.sin(theta - centre), torch.cos(theta - centre)
                )
                return centre + dose * relative
            if self.mode.startswith("permute_"):
                repeat = int(self.mode.split("_", 1)[1])
                permutation = deterministic_local_permutation(
                    self.edge_dst,
                    layer=layer,
                    repeat=repeat,
                    batch_number=self.batch_number,
                )
                return theta.index_select(0, permutation)
            raise RuntimeError(f"unknown intervention {self.mode!r}")

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def predict(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    hooks: PhaseInterventionHooks,
    mode: str,
    batch_number: int,
    atomic_energy_keys: torch.Tensor,
    atomic_energy_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos, atomic_numbers, batch_idx, _, _, src, dst, shifts, cell, _ = batch
    used_dst = dst if bool(getattr(model, "preserve_edge_order", False)) else dst[torch.argsort(dst)]
    hooks.prepare(mode, used_dst, batch_number)
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
    force = -torch.autograd.grad(atomic_energy.sum(), differentiable_pos)[0]
    num_graphs = int(batch_idx.max().item()) + 1
    interaction_energy = graph_sum(atomic_energy, batch_idx, num_graphs)
    reference = lookup_atomic_reference(
        atomic_numbers, atomic_energy_keys, atomic_energy_values
    )
    total_energy = interaction_energy + graph_sum(reference, batch_idx, num_graphs)
    return total_energy.detach(), force.detach()


class MetricAccumulator:
    def __init__(self) -> None:
        self.force_abs = 0.0
        self.force_sq = 0.0
        self.force_count = 0
        self.energy_abs = 0.0
        self.energy_sq = 0.0
        self.energy_count = 0
        self.force_change_abs = 0.0
        self.force_change_max = 0.0

    def add(
        self,
        *,
        force: torch.Tensor,
        native_force: torch.Tensor,
        target_force: torch.Tensor,
        energy: torch.Tensor,
        target_energy: torch.Tensor,
        atom_counts: torch.Tensor,
    ) -> None:
        force_error = (force - target_force).detach().double().reshape(-1)
        energy_error = ((energy - target_energy.reshape(-1)) / atom_counts).detach().double()
        force_change = (force - native_force).detach().double().reshape(-1)
        self.force_abs += float(force_error.abs().sum())
        self.force_sq += float(force_error.square().sum())
        self.force_count += int(force_error.numel())
        self.energy_abs += float(energy_error.abs().sum())
        self.energy_sq += float(energy_error.square().sum())
        self.energy_count += int(energy_error.numel())
        self.force_change_abs += float(force_change.abs().sum())
        self.force_change_max = max(
            self.force_change_max,
            float(force_change.abs().max()) if force_change.numel() else 0.0,
        )

    def result(self) -> dict[str, float | int]:
        return {
            "force_mae_ev_per_angstrom": self.force_abs / self.force_count,
            "force_rmse_ev_per_angstrom": math.sqrt(self.force_sq / self.force_count),
            "energy_mae_ev_per_atom": self.energy_abs / self.energy_count,
            "energy_rmse_ev_per_atom": math.sqrt(self.energy_sq / self.energy_count),
            "mean_absolute_force_change_ev_per_angstrom": (
                self.force_change_abs / self.force_count
            ),
            "maximum_absolute_force_change_ev_per_angstrom": self.force_change_max,
            "force_component_count": self.force_count,
            "frame_count": self.energy_count,
        }


def write_parquet(path: Path, frames: list[pd.DataFrame]) -> None:
    if not frames:
        raise RuntimeError(f"no records collected for {path}")
    pd.concat(frames, ignore_index=True).to_parquet(
        path, index=False, compression="zstd"
    )


def main() -> None:
    args = parse_args()
    if args.permutation_repeats <= 0:
        raise ValueError("--permutation-repeats must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    elements = [item.strip() for item in args.elements.split(",") if item.strip()]
    device = torch.device(args.device)

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    deployed = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=str(checkpoint),
        element_types=elements,
        device=str(device),
    )
    model = deployed.wrapper.model
    model.eval()
    model.skip_input_validation = False
    dtype = next(model.parameters()).dtype
    hooks = PhaseInterventionHooks(model)

    dataset = H5Dataset(prefix=args.split, data_dir=args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_h5,
        num_workers=0,
    )
    modes = ["native", "aligned"]
    modes += [f"dose_{dose:.2f}" for dose in DOSES]
    modes += [f"permute_{repeat}" for repeat in range(args.permutation_repeats)]
    modes += ["common_shift"]
    metrics = {mode: MetricAccumulator() for mode in modes}
    atomic_frames: list[pd.DataFrame] = []
    frame_frames: list[pd.DataFrame] = []
    global_frame = 0

    try:
        for batch_number, cpu_batch in enumerate(loader):
            if args.max_batches is not None and batch_number >= args.max_batches:
                break
            batch = move_batch(cpu_batch, device, dtype)
            (
                _,
                atomic_numbers,
                batch_idx,
                target_force,
                target_energy,
                _,
                dst,
                _,
                _,
                _,
            ) = batch
            num_graphs = int(batch_idx.max().item()) + 1
            atom_counts = torch.bincount(batch_idx, minlength=num_graphs).to(dtype=dtype)
            predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for mode in modes:
                predictions[mode] = predict(
                    model,
                    batch,
                    hooks,
                    mode,
                    batch_number,
                    deployed.wrapper.atomic_energy_keys,
                    deployed.wrapper.atomic_energy_values,
                )
            native_energy, native_force = predictions["native"]
            for mode, (energy, force) in predictions.items():
                metrics[mode].add(
                    force=force,
                    native_force=native_force,
                    target_force=target_force,
                    energy=energy,
                    target_energy=target_energy,
                    atom_counts=atom_counts,
                )

            batch_idx_cpu = batch_idx.detach().cpu().numpy()
            z_cpu = atomic_numbers.detach().cpu().numpy()
            ref_force = target_force.detach().cpu().numpy()
            neighbour_count = torch.bincount(
                dst, minlength=atomic_numbers.shape[0]
            ).detach().cpu().numpy()
            atom_payload: dict[str, Any] = {
                "dataset": args.dataset,
                "seed": args.seed,
                "test": args.test_label,
                "frame_id": global_frame + batch_idx_cpu,
                "atom_index_in_batch": np.arange(len(batch_idx_cpu)),
                "atomic_number": z_cpu,
                "neighbour_count": neighbour_count,
                "reference_force_norm": np.linalg.norm(ref_force, axis=1),
            }
            for mode, (_, force) in predictions.items():
                error = (force.detach().cpu().numpy() - ref_force)
                atom_payload[f"force_mae_{mode}"] = np.mean(np.abs(error), axis=1)
            permute_columns = [
                atom_payload[f"force_mae_permute_{repeat}"]
                for repeat in range(args.permutation_repeats)
            ]
            atom_payload["force_mae_permute_mean"] = np.mean(
                np.stack(permute_columns, axis=0), axis=0
            )
            atomic_frames.append(pd.DataFrame(atom_payload))

            frame_payload: dict[str, Any] = {
                "dataset": [args.dataset] * num_graphs,
                "seed": [args.seed] * num_graphs,
                "test": [args.test_label] * num_graphs,
                "frame_id": np.arange(global_frame, global_frame + num_graphs),
                "atom_count": atom_counts.detach().cpu().numpy().astype(np.int64),
            }
            target_e_cpu = target_energy.reshape(-1).detach().cpu().numpy()
            for mode, (energy, _force) in predictions.items():
                frame_payload[f"energy_abs_error_per_atom_{mode}"] = np.abs(
                    energy.detach().cpu().numpy() - target_e_cpu
                ) / frame_payload["atom_count"]
            frame_frames.append(pd.DataFrame(frame_payload))
            global_frame += num_graphs
            print(
                f"batch={batch_number + 1}/{len(loader)} frames={global_frame} "
                f"atoms={atomic_numbers.shape[0]}",
                flush=True,
            )
    finally:
        hooks.close()

    write_parquet(output_dir / "paired_atomic_outcomes.parquet", atomic_frames)
    write_parquet(output_dir / "paired_frame_outcomes.parquet", frame_frames)
    aggregate = {mode: accumulator.result() for mode, accumulator in metrics.items()}
    permutation_force_mae = float(
        np.mean(
            [
                aggregate[f"permute_{repeat}"]["force_mae_ev_per_angstrom"]
                for repeat in range(args.permutation_repeats)
            ]
        )
    )
    aggregate["permute_mean"] = {
        "force_mae_ev_per_angstrom": permutation_force_mae,
        "force_mae_mev_per_angstrom": 1000.0 * permutation_force_mae,
    }
    for values in aggregate.values():
        if "force_mae_ev_per_angstrom" in values:
            values["force_mae_mev_per_angstrom"] = (
                1000.0 * values["force_mae_ev_per_angstrom"]
            )
        if "energy_mae_ev_per_atom" in values:
            values["energy_mae_mev_per_atom"] = 1000.0 * values["energy_mae_ev_per_atom"]
    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "seed": args.seed,
        "test": args.test_label,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "split": args.split,
        "frames": global_frame,
        "permutation_repeats": args.permutation_repeats,
        "max_batches": args.max_batches,
        "dose_definition": (
            "theta(lambda)=circular_mean_i(theta)+lambda*wrap(theta-circular_mean_i(theta))"
        ),
        "estimands": {
            "aligned": "controlled removal of relative phase with amplitudes, messages, weights, and full Hermitian j,k closure fixed",
            "permutation": "controlled breakage of within-centre phase-edge assignment while preserving each local phase multiset",
            "common_shift": "negative-control U(1) invariance intervention",
        },
        "metrics": aggregate,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "DONE").touch()
    print(f"wrote {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
