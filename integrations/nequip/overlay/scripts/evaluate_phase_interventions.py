#!/usr/bin/env python3
"""Frozen-checkpoint phase interventions for NequIP-CHORUS.

The validation-Force-MAE-selected checkpoint and every learned parameter are
held fixed.  The test split is evaluated under the factual phase, phase
alignment, partial phase doses, within-centre phase permutations, and a common
phase shift.  The output is one auditable JSON record per trained model; atoms
and frames are outcome observations, not causal replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from pathlib import Path
from typing import Any

import torch

from nequip.data import AtomicData, AtomicDataDict, Collater, dataset_from_config
from nequip.scripts.evaluate import _load_deployed_or_traindir
from nequip.scripts.train import default_config
from nequip.utils import Config


DOSES = (0.25, 0.50, 0.75)
PERMUTATION_REPEATS = 4
GLOBAL_SHIFT = 1.234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--test-config", type=Path, required=True)
    parser.add_argument("--scope", choices=("final", "persistent"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circular_centre(theta: torch.Tensor, edge_dst: torch.Tensor) -> torch.Tensor:
    if edge_dst.numel() == 0:
        return theta
    num_nodes = int(edge_dst.max().item()) + 1
    sin_sum = theta.new_zeros(num_nodes)
    cos_sum = theta.new_zeros(num_nodes)
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
    """Return a reproducible permutation that never crosses centre atoms."""
    dst_cpu = edge_dst.detach().cpu()
    permutation = torch.arange(edge_dst.numel(), dtype=torch.long)
    for group_number, centre in enumerate(torch.unique(dst_cpu, sorted=True)):
        indexes = torch.nonzero(dst_cpu == centre, as_tuple=False).reshape(-1)
        count = int(indexes.numel())
        if count <= 1:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            20260829
            + 1_000_003 * int(repeat)
            + 10_007 * int(layer)
            + 101 * int(batch_number)
            + int(group_number)
        )
        order = torch.randperm(count, generator=generator)
        if bool(torch.equal(order, torch.arange(count))):
            order = torch.roll(order, shifts=1)
        permutation[indexes] = indexes.index_select(0, order)
    return permutation.to(edge_dst.device)


class PhaseController:
    """Patch only the learned phase in every Hermitian-density module."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.mode = "native"
        self.batch_number = 0
        self.modules: list[torch.nn.Module] = []
        for layer, module in enumerate(model.modules()):
            if not all(
                hasattr(module, name)
                for name in (
                    "phase_network",
                    "node_norm",
                    "edge_projection",
                    "charged_edge_messages",
                    "density_pairs",
                )
            ):
                continue
            if str(module.density_pairs).lower() != "full":
                raise RuntimeError(
                    f"frozen phase probes require a Full checkpoint, got "
                    f"{module.density_pairs!r}"
                )
            self.modules.append(module)
            controller = self

            def patched(
                this,
                edge_messages: torch.Tensor,
                node_features: torch.Tensor,
                edge_invariants: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                *,
                _layer: int = layer,
                _controller: "PhaseController" = controller,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                node_invariants = this.node_norm(node_features)
                context = torch.cat(
                    (
                        edge_invariants,
                        node_invariants[edge_src],
                        node_invariants[edge_dst],
                    ),
                    dim=-1,
                )
                amplitude_raw, phase_raw = this.phase_network(context).unbind(-1)
                amplitude = torch.nn.functional.softplus(amplitude_raw) / math.log(2.0)
                phase = math.pi * torch.tanh(phase_raw)
                mode = _controller.mode
                if mode != "native":
                    centre = circular_centre(phase, edge_dst)
                    if mode == "aligned":
                        phase = centre
                    elif mode == "common_shift":
                        phase = phase + GLOBAL_SHIFT
                    elif mode.startswith("dose_"):
                        dose = float(mode.split("_", 1)[1])
                        relative = torch.atan2(
                            torch.sin(phase - centre), torch.cos(phase - centre)
                        )
                        phase = centre + dose * relative
                    elif mode.startswith("permute_"):
                        repeat = int(mode.split("_", 1)[1])
                        permutation = deterministic_local_permutation(
                            edge_dst,
                            layer=_layer,
                            repeat=repeat,
                            batch_number=_controller.batch_number,
                        )
                        phase = phase.index_select(0, permutation)
                    else:
                        raise RuntimeError(f"unknown phase intervention {mode!r}")
                charged = this.edge_projection(edge_messages)
                return (
                    charged * (amplitude * torch.cos(phase)).unsqueeze(-1),
                    charged * (amplitude * torch.sin(phase)).unsqueeze(-1),
                )

            object.__setattr__(
                module,
                "charged_edge_messages",
                types.MethodType(patched, module),
            )
        if not self.modules:
            raise RuntimeError("loaded checkpoint has no active CHORUS phase module")

    def set(self, mode: str, batch_number: int) -> None:
        self.mode = mode
        self.batch_number = int(batch_number)


class Accumulator:
    def __init__(self) -> None:
        self.force_abs = 0.0
        self.force_sq = 0.0
        self.force_count = 0
        self.energy_abs = 0.0
        self.energy_sq = 0.0
        self.frame_count = 0
        self.force_change_abs = 0.0
        self.force_change_max = 0.0
        self.energy_change_abs = 0.0
        self.energy_change_max = 0.0

    def add(
        self,
        *,
        predicted_force: torch.Tensor,
        predicted_energy: torch.Tensor,
        factual_force: torch.Tensor,
        factual_energy: torch.Tensor,
        target_force: torch.Tensor,
        target_energy: torch.Tensor,
        atom_counts: torch.Tensor,
    ) -> None:
        force_error = (predicted_force - target_force).detach().double().reshape(-1)
        energy_error = (
            (predicted_energy.reshape(-1) - target_energy.reshape(-1)) / atom_counts
        ).detach().double()
        force_change = (
            predicted_force - factual_force
        ).detach().double().reshape(-1)
        energy_change = (
            (predicted_energy.reshape(-1) - factual_energy.reshape(-1)) / atom_counts
        ).detach().double()
        self.force_abs += float(force_error.abs().sum())
        self.force_sq += float(force_error.square().sum())
        self.force_count += int(force_error.numel())
        self.energy_abs += float(energy_error.abs().sum())
        self.energy_sq += float(energy_error.square().sum())
        self.frame_count += int(energy_error.numel())
        self.force_change_abs += float(force_change.abs().sum())
        self.energy_change_abs += float(energy_change.abs().sum())
        if force_change.numel():
            self.force_change_max = max(
                self.force_change_max, float(force_change.abs().max())
            )
        if energy_change.numel():
            self.energy_change_max = max(
                self.energy_change_max, float(energy_change.abs().max())
            )

    def result(self) -> dict[str, float | int]:
        return {
            "force_mae_mev_per_angstrom": 1000.0 * self.force_abs / self.force_count,
            "force_rmse_mev_per_angstrom": 1000.0
            * math.sqrt(self.force_sq / self.force_count),
            "energy_mae_mev_per_atom_raw": 1000.0 * self.energy_abs / self.frame_count,
            "energy_rmse_mev_per_atom_raw": 1000.0
            * math.sqrt(self.energy_sq / self.frame_count),
            "mean_absolute_force_change_mev_per_angstrom": 1000.0
            * self.force_change_abs
            / self.force_count,
            "maximum_absolute_force_change_mev_per_angstrom": 1000.0
            * self.force_change_max,
            "mean_absolute_energy_change_mev_per_atom": 1000.0
            * self.energy_change_abs
            / self.frame_count,
            "maximum_absolute_energy_change_mev_per_atom": 1000.0
            * self.energy_change_max,
            "force_component_count": self.force_count,
            "frame_count": self.frame_count,
        }


def discover_selected_checkpoint(work: Path) -> tuple[Path, dict[str, Any]]:
    metrics_path = work / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
    checkpoint = Path(
        metrics.get("selected_checkpoint", work / "run" / "best_model.pth")
    )
    if checkpoint.name != "best_model.pth":
        raise RuntimeError(f"unexpected selected checkpoint: {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = Config.from_file(
        str(work / "run" / "config.yaml"), defaults=default_config
    )
    if config.get("metrics_key") != "validation_f_mae":
        raise RuntimeError(
            f"best_model.pth was not selected by validation Force MAE: "
            f"metrics_key={config.get('metrics_key')!r}"
        )
    return checkpoint, metrics


def modes() -> tuple[str, ...]:
    return (
        "native",
        "aligned",
        *(f"dose_{dose:.2f}" for dose in DOSES),
        *(f"permute_{repeat}" for repeat in range(PERMUTATION_REPEATS)),
        "common_shift",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, selection = discover_selected_checkpoint(args.work_dir)
    device = torch.device(args.device)
    model, _, model_r_max, _ = _load_deployed_or_traindir(
        checkpoint, device=device, freeze=False
    )
    model.eval()
    controller = PhaseController(model)

    dataset_config = Config.from_file(
        str(args.test_config), defaults={"r_max": model_r_max}
    )
    dataset = dataset_from_config(dataset_config)
    collater = Collater.for_dataset(dataset, exclude_keys=[])
    accumulators = {mode: Accumulator() for mode in modes()}
    processed_frames = 0

    for batch_number, start in enumerate(range(0, len(dataset), args.batch_size)):
        if args.max_batches is not None and batch_number >= args.max_batches:
            break
        indexes = range(start, min(start + args.batch_size, len(dataset)))
        batch = collater.collate([dataset[index] for index in indexes]).to(device)
        data = AtomicData.to_AtomicDataDict(batch)
        targets_force = data[AtomicDataDict.FORCE_KEY]
        targets_energy = data[AtomicDataDict.TOTAL_ENERGY_KEY]
        batch_index = data[AtomicDataDict.BATCH_KEY]
        num_graphs = int(batch.num_graphs)
        atom_counts = torch.bincount(
            batch_index, minlength=num_graphs
        ).to(dtype=targets_energy.dtype)

        controller.set("native", batch_number)
        factual = model(dict(data))
        factual_force = factual[AtomicDataDict.FORCE_KEY].detach()
        factual_energy = factual[AtomicDataDict.TOTAL_ENERGY_KEY].detach()

        for mode in modes():
            if mode == "native":
                predicted_force = factual_force
                predicted_energy = factual_energy
            else:
                controller.set(mode, batch_number)
                prediction = model(dict(data))
                predicted_force = prediction[AtomicDataDict.FORCE_KEY].detach()
                predicted_energy = prediction[AtomicDataDict.TOTAL_ENERGY_KEY].detach()
            accumulators[mode].add(
                predicted_force=predicted_force,
                predicted_energy=predicted_energy,
                factual_force=factual_force,
                factual_energy=factual_energy,
                target_force=targets_force,
                target_energy=targets_energy,
                atom_counts=atom_counts,
            )
        processed_frames += num_graphs

    results = {mode: accumulator.result() for mode, accumulator in accumulators.items()}
    common = results["common_shift"]
    record = {
        "schema_version": 1,
        "backbone": "NequIP-SH",
        "scope": args.scope,
        "dataset": args.dataset,
        "seed": args.seed,
        "selection_rule": "minimum complete-validation Force MAE",
        "test_used_for_selection": False,
        "work_dir": str(args.work_dir),
        "test_config": str(args.test_config),
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": sha256(checkpoint),
        "selected_epoch": selection.get("selected_epoch"),
        "selected_step": selection.get("selected_step"),
        "phase_modules": len(controller.modules),
        "phase_interventions": {
            "alignment": "per-centre circular mean; relative phase set to zero",
            "dose": "per-centre wrapped relative phase multiplied by lambda",
            "permutation": "learned phases permuted only within each centre atom",
            "common_shift_radians": GLOBAL_SHIFT,
        },
        "processed_frames": processed_frames,
        "complete_test_split": args.max_batches is None,
        "energy_note": (
            "raw checkpoint energy; ZrO train-only energy calibration is not "
            "refitted under frozen interventions"
        ),
        "modes": results,
        "negative_control_pass": (
            float(common["maximum_absolute_force_change_mev_per_angstrom"])
            <= 0.02
            and float(common["maximum_absolute_energy_change_mev_per_atom"])
            <= 1.0
        ),
        "negative_control_tolerances": {
            "force_mev_per_angstrom": 0.02,
            "energy_mev_per_atom": 1.0,
            "energy_rationale": (
                "float32 total energies include large element-reference terms; "
                "the force check is the stricter invariance diagnostic"
            ),
        },
        "causal_unit_note": (
            "the trained dataset-by-seed model is the causal replicate; atom "
            "and frame counts are nested outcome observations"
        ),
    }
    output = args.output_dir / "metrics.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    if args.max_batches is None:
        (args.output_dir / "DONE").write_text("complete\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
