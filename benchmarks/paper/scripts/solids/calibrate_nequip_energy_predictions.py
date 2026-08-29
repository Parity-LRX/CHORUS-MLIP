#!/usr/bin/env python3
"""Fit a train-only elemental energy residual from NequIP XYZ predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import iread


def scalar(value: object) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"expected scalar energy, got shape {array.shape}")
    return float(array[0])


def energy_fields(atoms) -> tuple[float, float]:
    reference_candidates = [
        key for key in atoms.info
        if "energy" in key.lower() and key.startswith("original_dataset_")
    ]
    if len(reference_candidates) != 1 or atoms.calc is None:
        raise RuntimeError(
            "could not identify one reference energy field and one ASE prediction; "
            f"available info keys={sorted(atoms.info)}"
        )
    return (
        scalar(atoms.info[reference_candidates[0]]),
        float(atoms.get_potential_energy()),
    )


def load(path: Path, elements: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    reference: list[float] = []
    prediction: list[float] = []
    compositions: list[list[int]] = []
    for atoms in iread(path, index=":", format="extxyz"):
        ref, pred = energy_fields(atoms)
        symbols = atoms.get_chemical_symbols()
        reference.append(ref)
        prediction.append(pred)
        compositions.append([symbols.count(element) for element in elements])
    if not reference:
        raise RuntimeError(f"no predictions in {path}")
    return (
        np.asarray(reference, dtype=np.float64),
        np.asarray(prediction, dtype=np.float64),
        np.asarray(compositions, dtype=np.float64),
    )


def metrics(
    reference: np.ndarray, prediction: np.ndarray, counts: np.ndarray
) -> dict[str, float]:
    error = (prediction - reference) / counts.sum(axis=1)
    return {
        "energy_mae_ev_per_atom": float(np.mean(np.abs(error))),
        "energy_rmse_ev_per_atom": float(np.sqrt(np.mean(error**2))),
        "energy_mae_mev_per_atom": float(1000.0 * np.mean(np.abs(error))),
        "energy_rmse_mev_per_atom": float(1000.0 * np.sqrt(np.mean(error**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--elements", default="O,Zr")
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    elements = tuple(value.strip() for value in args.elements.split(",") if value.strip())
    loaded = {
        split: load(path, elements)
        for split, path in (("train", args.train), ("val", args.val), ("test", args.test))
    }
    train_reference, train_prediction, train_counts = loaded["train"]
    train_natoms = train_counts.sum(axis=1)
    train_fractions = train_counts / train_natoms[:, None]
    train_residual_per_atom = (
        train_reference - train_prediction
    ) / train_natoms
    delta, _, rank, singular_values = np.linalg.lstsq(
        train_fractions, train_residual_per_atom, rcond=None
    )
    if int(rank) != len(elements):
        raise RuntimeError(f"rank-deficient composition matrix: {rank} < {len(elements)}")

    result: dict[str, object] = {
        "selected_checkpoint": str(args.selected_checkpoint),
        "selection_rule": "minimum full-validation Force MAE; test never selects",
        "calibration_fit_split": "train only",
        "fit_objective": "least-squares residual energy per atom",
        "elements": list(elements),
        "delta_ev_per_element": {
            element: float(value) for element, value in zip(elements, delta)
        },
        "design_rank": int(rank),
        "singular_values": singular_values.tolist(),
        "forces_changed": False,
        "splits": {},
    }
    for split, (reference, prediction, counts) in loaded.items():
        result["splits"][split] = {
            "num_structures": int(reference.size),
            "raw": metrics(reference, prediction, counts),
            "calibrated": metrics(reference, prediction + counts @ delta, counts),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
