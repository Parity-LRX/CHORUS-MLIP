#!/usr/bin/env python3
"""Evaluate a validation-selected native-MACE checkpoint on a fixed test set.

The native CuEq training runs can finish all epochs and checkpoints but fail
while deep-copying the final scripted model.  This evaluator reconstructs the
declared architecture, loads the selected state dict, and evaluates the held-
out split.  For Transition1x it can additionally fit one constant correction
per element on the training split only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import iread
from mace.calculators import MACECalculator


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def build_matching_model(
    state_dict: dict[str, torch.Tensor],
    atomic_numbers: list[int],
    avg_num_neighbors: float,
) -> torch.nn.Module:
    from e3nn import o3
    from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes
    from mace.modules.wrapper_ops import CuEquivarianceConfig

    atomic_energies = (
        state_dict["atomic_energies_fn.atomic_energies"].detach().cpu().numpy()
    )
    scale = float(state_dict["scale_shift.scale"].detach().cpu())
    shift = float(state_dict["scale_shift.shift"].detach().cpu())
    return ScaleShiftMACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(atomic_numbers),
        hidden_irreps=o3.Irreps("128x0e + 128x1o + 128x2e"),
        MLP_irreps=o3.Irreps("64x0e"),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=3,
        gate=gate_dict["silu"],
        radial_type="bessel",
        radial_MLP=[64, 64, 64],
        atomic_inter_scale=scale,
        atomic_inter_shift=shift,
        use_reduced_cg=True,
        cueq_config=CuEquivarianceConfig(
            enabled=True,
            layout="ir_mul",
            group="O3_e3nn",
            optimize_all=True,
            conv_fusion=True,
        ),
    )


def select_validation_checkpoint(
    results_file: Path,
    checkpoint_dir: Path,
    name: str,
    seed: int,
) -> tuple[dict[str, object], Path, int]:
    rows = [json.loads(line) for line in results_file.read_text().splitlines()]
    validation = [
        row
        for row in rows
        if row.get("mode") == "eval" and row.get("epoch") is not None
    ]
    if not validation:
        raise RuntimeError(f"no validation rows in {results_file}")
    selected = min(
        validation,
        key=lambda row: (float(row["mae_f"]), int(row["epoch"])),
    )
    epoch = int(selected["epoch"])
    checkpoint = checkpoint_dir / f"{name}_run-{seed}_epoch-{epoch}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return selected, checkpoint, len(validation)


def predict_frame(
    calculator: MACECalculator,
    atoms,
) -> tuple[float, np.ndarray]:
    prediction = atoms.copy()
    prediction.calc = calculator
    return (
        float(prediction.get_potential_energy()),
        np.asarray(prediction.get_forces(), dtype=np.float64),
    )


def reference_energy(atoms) -> float:
    if atoms.calc is not None and "energy" in atoms.calc.results:
        return float(atoms.calc.results["energy"])
    for key in ("energy", "Energy"):
        if key in atoms.info:
            return float(atoms.info[key])
    raise KeyError("reference structure has no energy or Energy field")


def reference_forces(atoms) -> np.ndarray:
    if atoms.calc is not None and "forces" in atoms.calc.results:
        return np.asarray(atoms.calc.results["forces"], dtype=np.float64)
    for key in ("forces", "force"):
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=np.float64)
    raise KeyError("reference structure has no forces or force array")


def fit_train_only_element_correction(
    calculator: MACECalculator,
    path: Path,
    atomic_numbers: list[int],
) -> dict[str, object]:
    counts = []
    residuals = []
    frames = 0
    for atoms in iread(path, index=":", format="extxyz"):
        prediction_energy, _ = predict_frame(calculator, atoms)
        target_energy = reference_energy(atoms)
        numbers = np.asarray(atoms.numbers)
        counts.append([int(np.sum(numbers == z)) for z in atomic_numbers])
        residuals.append(target_energy - prediction_energy)
        frames += 1
    design = np.asarray(counts, dtype=np.float64)
    target = np.asarray(residuals, dtype=np.float64)
    correction, _, rank, singular_values = np.linalg.lstsq(
        design, target, rcond=None
    )
    fitted = design @ correction
    return {
        "frames": frames,
        "atomic_numbers": atomic_numbers,
        "correction_ev_per_element": correction.tolist(),
        "design_rank": int(rank),
        "singular_values": singular_values.tolist(),
        "train_residual_mae_ev_per_structure_before": float(
            np.mean(np.abs(target))
        ),
        "train_residual_mae_ev_per_structure_after": float(
            np.mean(np.abs(target - fitted))
        ),
    }


def evaluate(
    calculator: MACECalculator,
    path: Path,
    atomic_numbers: list[int],
    correction: np.ndarray | None,
) -> dict[str, float | int]:
    raw_energy_errors = []
    corrected_energy_errors = []
    force_errors = []
    frames = 0
    for atoms in iread(path, index=":", format="extxyz"):
        target_energy = reference_energy(atoms)
        target_forces = reference_forces(atoms)
        prediction_energy, prediction_forces = predict_frame(calculator, atoms)
        raw_error = (prediction_energy - target_energy) / len(atoms)
        raw_energy_errors.append(raw_error)
        if correction is not None:
            numbers = np.asarray(atoms.numbers)
            counts = np.asarray(
                [np.sum(numbers == z) for z in atomic_numbers],
                dtype=np.float64,
            )
            prediction_energy += float(counts @ correction)
        corrected_energy_errors.append(
            (prediction_energy - target_energy) / len(atoms)
        )
        force_errors.append(prediction_forces - target_forces)
        frames += 1

    raw_energy = np.asarray(raw_energy_errors)
    corrected_energy = np.asarray(corrected_energy_errors)
    force = np.concatenate([value.reshape(-1) for value in force_errors])
    return {
        "frames": frames,
        "raw_energy_mae_ev_per_atom": float(np.mean(np.abs(raw_energy))),
        "raw_energy_rmse_ev_per_atom": float(np.sqrt(np.mean(raw_energy**2))),
        "energy_mae_ev_per_atom": float(np.mean(np.abs(corrected_energy))),
        "energy_rmse_ev_per_atom": float(
            np.sqrt(np.mean(corrected_energy**2))
        ),
        "force_mae_ev_per_angstrom": float(np.mean(np.abs(force))),
        "force_rmse_ev_per_angstrom": float(np.sqrt(np.mean(force**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--atomic-numbers", required=True)
    parser.add_argument("--avg-num-neighbors", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-train-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atomic_numbers = csv_ints(args.atomic_numbers)
    results_file = (
        args.run_dir / "results" / f"{args.name}_run-{args.seed}_train.txt"
    )
    selected, checkpoint, validation_count = select_validation_checkpoint(
        results_file,
        args.run_dir / "checkpoints",
        args.name,
        args.seed,
    )
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model = build_matching_model(
        state["model"],
        atomic_numbers,
        args.avg_num_neighbors,
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device)
    model.eval()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    calculator = MACECalculator(
        models=model,
        device=args.device,
        default_dtype="float32",
        enable_cueq=False,
    )

    calibration = None
    correction = None
    if args.calibration_train_file is not None:
        calibration = fit_train_only_element_correction(
            calculator,
            args.calibration_train_file,
            atomic_numbers,
        )
        correction = np.asarray(
            calibration["correction_ev_per_element"],
            dtype=np.float64,
        )

    test = evaluate(
        calculator,
        args.test_file,
        atomic_numbers,
        correction,
    )
    result = {
        "selection_rule": "minimum full-validation Force MAE; test never selects",
        "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": str(checkpoint),
        "validation_checkpoint_count": validation_count,
        "validation_log_metrics": selected,
        "trainable_parameters": parameter_count,
        "energy_calibration": calibration,
        "energy_calibration_fit_split": (
            "train" if calibration is not None else None
        ),
        "test": test,
        "force_unchanged_by_energy_calibration": True,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "DONE").touch()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
