#!/usr/bin/env python3
"""Fit a train-only per-element energy correction for a selected DPA-4 model."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np


HEADER = re.compile(r"^#\s+(.+):\s+data_e\s+pred_e\s*$")


def run_detail(dp: Path, checkpoint: Path, split: Path, prefix: Path) -> None:
    energy_detail = prefix.with_suffix(".e.out")
    if energy_detail.is_file():
        return
    command = [
        str(dp),
        "--pt",
        "test",
        "-m",
        str(checkpoint),
        "-s",
        str(split),
        "-n",
        "0",
        "-d",
        str(prefix),
    ]
    env = os.environ.copy()
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    env.setdefault("DP_INFER_BATCH_SIZE", "8192")
    log = prefix.with_suffix(".log")
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def composition(system: Path, elements: tuple[str, ...]) -> np.ndarray:
    type_map = (system / "type_map.raw").read_text().split()
    type_indices = np.loadtxt(system / "type.raw", dtype=np.int64, ndmin=1)
    counts = {symbol: 0 for symbol in elements}
    for index in type_indices:
        symbol = type_map[int(index)]
        if symbol not in counts:
            raise ValueError(f"unexpected element {symbol!r} in {system}")
        counts[symbol] += 1
    return np.asarray([counts[symbol] for symbol in elements], dtype=np.float64)


def read_energy_detail(
    path: Path, elements: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    references: list[float] = []
    predictions: list[float] = []
    compositions: list[np.ndarray] = []
    current: np.ndarray | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = HEADER.match(line)
        if match is not None:
            current = composition(Path(match.group(1)), elements)
            continue
        if not line or line.startswith("#"):
            continue
        if current is None:
            raise RuntimeError(f"energy row before system header in {path}")
        fields = line.split()
        if len(fields) < 2:
            raise RuntimeError(f"malformed energy row in {path}: {line!r}")
        references.append(float(fields[0]))
        predictions.append(float(fields[1]))
        compositions.append(current)
    if not references:
        raise RuntimeError(f"no energy predictions found in {path}")
    return (
        np.asarray(references, dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
        np.stack(compositions),
    )


def energy_metrics(
    reference: np.ndarray, prediction: np.ndarray, counts: np.ndarray
) -> dict[str, float]:
    per_atom_error = (prediction - reference) / counts.sum(axis=1)
    return {
        "energy_mae_ev_per_atom": float(np.abs(per_atom_error).mean()),
        "energy_rmse_ev_per_atom": float(np.sqrt(np.square(per_atom_error).mean())),
        "energy_mae_mev_per_atom": float(1000.0 * np.abs(per_atom_error).mean()),
        "energy_rmse_mev_per_atom": float(
            1000.0 * np.sqrt(np.square(per_atom_error).mean())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dp", type=Path, default=Path("/home/ylzhang/venvs/dpa4-master/bin/dp")
    )
    parser.add_argument("--elements", default="H,C,N,O")
    args = parser.parse_args()

    elements = tuple(item.strip() for item in args.elements.split(",") if item.strip())
    args.out.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for split in ("train", "val", "test"):
        prefix = args.out / split
        run_detail(args.dp, args.checkpoint, args.data / split, prefix)
        loaded[split] = read_energy_detail(prefix.with_suffix(".e.out"), elements)

    train_reference, train_prediction, train_counts = loaded["train"]
    delta, _, matrix_rank, singular_values = np.linalg.lstsq(
        train_counts,
        train_reference - train_prediction,
        rcond=None,
    )

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "selection_rule": "checkpoint selected by minimum full-validation Force MAE",
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
