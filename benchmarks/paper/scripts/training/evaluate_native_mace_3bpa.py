#!/usr/bin/env python3
"""Select a native-MACE checkpoint on 3BPA validation and test temperatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import iread
from mace.calculators import MACECalculator


def build_matching_model(state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Rebuild the fixed 3BPA native-MACE architecture without an exported model.

    mace-torch writes complete state-dict checkpoints before its final model
    deepcopy/export step.  Some e3nn/CuEq runtimes cannot pickle a scripted CG
    helper during that final deepcopy even though training and all checkpoints
    are complete.  Reconstructing the declared architecture here makes the
    validation-selected checkpoint independently evaluable.
    """
    from e3nn import o3
    from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes

    atomic_energies = (
        state_dict["atomic_energies_fn.atomic_energies"]
        .detach()
        .cpu()
        .numpy()
    )
    scale = float(state_dict["scale_shift.scale"].detach().cpu())
    shift = float(state_dict["scale_shift.shift"].detach().cpu())
    return ScaleShiftMACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=4,
        hidden_irreps=o3.Irreps("128x0e + 128x1o + 128x2e"),
        MLP_irreps=o3.Irreps("64x0e"),
        atomic_energies=atomic_energies,
        avg_num_neighbors=16.712427983539094,
        atomic_numbers=[1, 6, 7, 8],
        correlation=3,
        gate=gate_dict["silu"],
        radial_type="bessel",
        radial_MLP=[64, 64, 64],
        atomic_inter_scale=scale,
        atomic_inter_shift=shift,
        use_reduced_cg=True,
    )


def evaluate(calculator: MACECalculator, path: Path) -> dict[str, float]:
    energy_errors = []
    force_errors = []
    frames = 0
    for atoms in iread(path, index=":", format="extxyz"):
        reference_energy = float(atoms.get_potential_energy())
        reference_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        prediction = atoms.copy()
        prediction.calc = calculator
        energy_errors.append(
            (float(prediction.get_potential_energy()) - reference_energy) / len(atoms)
        )
        force_errors.append(
            np.asarray(prediction.get_forces(), dtype=np.float64) - reference_forces
        )
        frames += 1
    energy = np.asarray(energy_errors)
    force = np.concatenate([value.reshape(-1) for value in force_errors])
    return {
        "frames": frames,
        "energy_mae_ev_per_atom": float(np.mean(np.abs(energy))),
        "energy_rmse_ev_per_atom": float(np.sqrt(np.mean(energy**2))),
        "force_mae_ev_per_angstrom": float(np.mean(np.abs(force))),
        "force_rmse_ev_per_angstrom": float(np.sqrt(np.mean(force**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--name", default="native_mace_3bpa")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results_file = (
        args.run_dir
        / "results"
        / f"{args.name}_run-{args.seed}_train.txt"
    )
    rows = [json.loads(line) for line in results_file.read_text().splitlines()]
    validation = [
        row
        for row in rows
        if row.get("mode") == "eval" and row.get("epoch") is not None
    ]
    if not validation:
        raise RuntimeError(f"no per-epoch validation rows in {results_file}")
    selected = min(validation, key=lambda row: (float(row["mae_f"]), int(row["epoch"])))
    epoch = int(selected["epoch"])
    checkpoint = (
        args.run_dir
        / "checkpoints"
        / f"{args.name}_run-{args.seed}_epoch-{epoch}.pt"
    )
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model_candidates = sorted((args.run_dir / "models").glob("*.model"))
    if len(model_candidates) == 1:
        model = torch.load(
            model_candidates[0],
            map_location=args.device,
            weights_only=False,
        )
        model_source = str(model_candidates[0])
    elif not model_candidates:
        model = build_matching_model(state["model"])
        model_source = "reconstructed from declared 3BPA native-MACE architecture"
    else:
        raise RuntimeError(f"expected at most one exported model, got {model_candidates}")
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device)
    model.eval()
    calculator = MACECalculator(
        models=model,
        device=args.device,
        default_dtype="float32",
        enable_cueq=False,
    )

    result = {
        "selection_rule": "minimum 300K validation Force MAE; test never selects",
        "selected_epoch": epoch,
        "selected_checkpoint": str(checkpoint),
        "model_source": model_source,
        "validation_checkpoint_count": len(validation),
        "validation_log_metrics": selected,
        "tests": {},
    }
    for temperature in ("300K", "600K", "1200K"):
        result["tests"][temperature] = evaluate(
            calculator,
            args.data_dir / f"test_{temperature}.extxyz",
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "DONE").touch()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
