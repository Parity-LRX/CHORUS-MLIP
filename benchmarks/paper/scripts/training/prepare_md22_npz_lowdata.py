#!/usr/bin/env python3
"""Prepare one MD22 npz into a fixed 600/600/1000 low-data split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols


KCAL_MOL_TO_EV = 0.0433641153087705


def stratified_indices(values: np.ndarray, count: int, bins: int, seed: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    exact = np.asarray([count * len(group) / len(order) for group in groups])
    allocation = np.floor(exact).astype(int)
    remainder = count - int(allocation.sum())
    if remainder:
        allocation[np.argsort(-(exact - allocation), kind="stable")[:remainder]] += 1
    rng = np.random.default_rng(seed)
    selected = [
        rng.choice(group, size=n, replace=False)
        for group, n in zip(groups, allocation)
        if n
    ]
    result = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
    if len(result) != count or len(np.unique(result)) != count:
        raise RuntimeError("failed to construct the requested unique stratified subset")
    return result


def write_extxyz(
    path: Path,
    indices: np.ndarray,
    positions: np.ndarray,
    numbers: np.ndarray,
    forces: np.ndarray,
    energies: np.ndarray,
    config_type: str,
) -> None:
    symbols = [chemical_symbols[int(z)] for z in numbers]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index in indices:
            handle.write(f"{len(numbers)}\n")
            handle.write(
                'Lattice="100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0" '
                'Properties=species:S:1:pos:R:3:force:R:3 '
                f'Energy={float(energies[index]):.16g} config_type={config_type} '
                'pbc="F F F"\n'
            )
            for symbol, xyz, force in zip(symbols, positions[index], forces[index]):
                handle.write(
                    f"{symbol:2s} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} "
                    f"{force[0]:.10f} {force[1]:.10f} {force[2]:.10f}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-size", type=int, default=1200)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--energy-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    with np.load(args.input) as data:
        positions = np.asarray(data["R"], dtype=np.float64)
        numbers = np.asarray(data["z"], dtype=np.int64).reshape(-1)
        forces = np.asarray(data["F"], dtype=np.float64) * KCAL_MOL_TO_EV
        energies = np.asarray(data["E"], dtype=np.float64).reshape(-1) * KCAL_MOL_TO_EV

    if len(energies) < args.candidate_size + args.test_size:
        raise ValueError("source is too small for the requested disjoint subsets")
    candidate = stratified_indices(energies, args.candidate_size, args.energy_bins, args.seed)
    remaining = np.setdiff1d(np.arange(len(energies), dtype=np.int64), candidate)
    test_local = stratified_indices(
        energies[remaining], args.test_size, args.energy_bins, args.seed + 1
    )
    test = np.sort(remaining[test_local])
    if np.intersect1d(candidate, test).size:
        raise RuntimeError("candidate/test overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.input.stem.removeprefix("md22_").replace("-", "_")
    write_extxyz(
        args.output_dir / "candidate_1200.extxyz",
        candidate,
        positions,
        numbers,
        forces,
        energies,
        tag,
    )
    write_extxyz(
        args.output_dir / "heldout_test.extxyz",
        test,
        positions,
        numbers,
        forces,
        energies,
        tag,
    )
    np.save(args.output_dir / "candidate_source_indices.npy", candidate)
    np.save(args.output_dir / "test_source_indices.npy", test)
    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(
            {
                "dataset": args.input.stem,
                "source": str(args.input),
                "source_frames": int(len(energies)),
                "atoms": int(len(numbers)),
                "atomic_numbers": sorted(map(int, np.unique(numbers))),
                "candidate_frames": int(len(candidate)),
                "train_frames_after_preprocess": 600,
                "validation_frames_after_preprocess": 600,
                "heldout_test_frames": int(len(test)),
                "unused_source_frames": int(len(energies) - len(candidate) - len(test)),
                "selection": "energy-stratified disjoint candidate and test subsets",
                "energy_bins": int(args.energy_bins),
                "seed": int(args.seed),
                "units": {"energy": "eV", "force": "eV/Angstrom"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"{args.input.stem}: {len(candidate)} candidate, {len(test)} test")


if __name__ == "__main__":
    main()
