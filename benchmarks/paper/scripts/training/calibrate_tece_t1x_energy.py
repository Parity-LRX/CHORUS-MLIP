#!/usr/bin/env python3
"""Fit a train-only per-element energy correction for a selected TECE model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from ase.io import iread


def run_predictions(
    tace_eval: Path,
    checkpoint: Path,
    source: Path,
    output: Path,
    log: Path,
    batch_size: int,
) -> None:
    if output.is_file():
        return
    command = [
        str(tace_eval),
        "-i",
        str(source),
        "-o",
        str(output),
        "-m",
        str(checkpoint),
        "-t",
        "0",
        "-e",
        "1",
        "-b",
        str(batch_size),
        "--device",
        "cuda",
        "--dtype",
        "float32",
        "--nl_backend",
        "matscipy",
        "--energy_key",
        "energy",
        "--forces_key",
        "forces",
    ]
    env = os.environ.copy()
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def reference_energy(atoms) -> float:
    if atoms.calc is not None and "energy" in atoms.calc.results:
        return float(atoms.calc.results["energy"])
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    raise KeyError("reference structure has no energy")


def read_pair(
    source: Path, predicted: Path, elements: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    references: list[float] = []
    predictions: list[float] = []
    compositions: list[np.ndarray] = []
    count = 0
    for reference, prediction in zip(
        iread(source, index=":"),
        iread(predicted, index=":"),
        strict=True,
    ):
        references.append(reference_energy(reference))
        predictions.append(float(prediction.info["TACE_energy"]))
        symbols = reference.get_chemical_symbols()
        compositions.append(
            np.asarray([symbols.count(element) for element in elements], dtype=np.float64)
        )
        count += 1
    if count == 0:
        raise RuntimeError(f"no structures in {source}")
    return (
        np.asarray(references, dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
        np.stack(compositions),
    )


def energy_metrics(
    reference: np.ndarray, prediction: np.ndarray, counts: np.ndarray
) -> dict[str, float]:
    error = (prediction - reference) / counts.sum(axis=1)
    return {
        "energy_mae_mev_per_atom": float(1000.0 * np.abs(error).mean()),
        "energy_rmse_mev_per_atom": float(
            1000.0 * np.sqrt(np.square(error).mean())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tace-eval",
        type=Path,
        default=Path("/home/ylzhang/tace_chorus_venv/bin/tace-eval"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--elements", default="H,C,N,O")
    args = parser.parse_args()

    elements = tuple(item.strip() for item in args.elements.split(",") if item.strip())
    args.out.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for split in ("train", "val", "test"):
        source = args.data / f"{split}.extxyz"
        output = args.out / f"{split}_predicted.extxyz"
        run_predictions(
            args.tace_eval,
            args.checkpoint,
            source,
            output,
            args.out / f"{split}_predict.log",
            args.batch_size,
        )
        loaded[split] = read_pair(source, output, elements)

    train_reference, train_prediction, train_counts = loaded["train"]
    delta, _, matrix_rank, singular_values = np.linalg.lstsq(
        train_counts,
        train_reference - train_prediction,
        rcond=None,
    )
    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "selection_rule": "checkpoint selected by minimum validation Force MAE",
        "calibration_split": "train only",
        "elements": list(elements),
        "delta_ev_per_element": {
            element: float(value) for element, value in zip(elements, delta)
        },
        "normal_matrix_rank": int(matrix_rank),
        "singular_values": singular_values.tolist(),
        "forces_changed": False,
        "splits": {},
    }
    for split, (reference, prediction, counts) in loaded.items():
        corrected = prediction + counts @ delta
        result["splits"][split] = {
            "num_structures": int(reference.shape[0]),
            "raw": energy_metrics(reference, prediction, counts),
            "calibrated": energy_metrics(reference, corrected, counts),
        }
    (args.out / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "DONE").touch()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
