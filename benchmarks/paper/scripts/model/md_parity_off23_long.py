#!/usr/bin/env python3
"""Run native-MACE vs MACE-ICTC MD parity checks for MACE-OFF23.

The script has engine-specific entry points because the historical OFF23 pickle
needs an old e3nn runtime for native MACE, while MACE-ICTC runs in the current
environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms, units
from ase.build import molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet


DEFAULT_CASES = {
    "ethanol": {"kind": "molecule", "name": "CH3CH2OH", "steps": 2000},
    "acetic_acid": {"kind": "molecule", "name": "CH3COOH", "steps": 2000},
    "acetamide": {"kind": "molecule", "name": "CH3CONH2", "steps": 2000},
    "benzene": {"kind": "molecule", "name": "C6H6", "steps": 2000},
    "ethanol_64_grid": {"kind": "grid", "name": "CH3CH2OH", "nmol": 64, "steps": 200},
    "acetic_acid_64_grid": {"kind": "grid", "name": "CH3COOH", "nmol": 64, "steps": 200},
    "acetamide_64_grid": {"kind": "grid", "name": "CH3CONH2", "nmol": 64, "steps": 200},
    "benzene_64_grid": {"kind": "grid", "name": "C6H6", "nmol": 64, "steps": 200},
    "ethanol_128_grid": {"kind": "grid", "name": "CH3CH2OH", "nmol": 128, "steps": 200},
    "acetic_acid_128_grid": {"kind": "grid", "name": "CH3COOH", "nmol": 128, "steps": 200},
    "acetamide_128_grid": {"kind": "grid", "name": "CH3CONH2", "nmol": 128, "steps": 200},
    "benzene_128_grid": {"kind": "grid", "name": "C6H6", "nmol": 128, "steps": 200},
}


def make_atoms(case: str, spacing: float) -> Atoms:
    spec = DEFAULT_CASES[case]
    if spec["kind"] == "molecule":
        atoms = molecule(str(spec["name"]))
        atoms.center(vacuum=8.0)
        return atoms

    base = molecule(str(spec["name"]))
    base_pos = base.get_positions() - base.get_center_of_mass()
    base_symbols = base.get_chemical_symbols()
    nmol = int(spec["nmol"])
    nside = int(np.ceil(nmol ** (1.0 / 3.0)))
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    for idx in range(nmol):
        ix = idx % nside
        iy = (idx // nside) % nside
        iz = idx // (nside * nside)
        shift = np.array([ix, iy, iz], dtype=float) * spacing
        for sym, pos in zip(base_symbols, base_pos):
            symbols.append(sym)
            positions.append(pos + shift)
    atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float), pbc=False)
    atoms.center(vacuum=8.0)
    return atoms


def init_velocities(atoms: Atoms, *, case: str, seed: int, temperature_k: float) -> None:
    stable = sum((idx + 1) * ord(ch) for idx, ch in enumerate(case))
    rng = np.random.default_rng(seed + stable)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, force_temp=True, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)


def run_md(
    *,
    atoms: Atoms,
    calc,
    steps: int,
    dt_fs: float,
    record_stride: int = 1,
) -> dict[str, np.ndarray | float]:
    atoms.calc = calc
    dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    natoms = len(atoms)
    sample_steps = list(range(0, steps + 1, max(1, int(record_stride))))
    if sample_steps[-1] != steps:
        sample_steps.append(steps)
    positions = np.empty((len(sample_steps), natoms, 3), dtype=np.float64)
    velocities = np.empty((len(sample_steps), natoms, 3), dtype=np.float64)
    energies = np.empty(len(sample_steps), dtype=np.float64)
    forces = np.empty((len(sample_steps), natoms, 3), dtype=np.float64)
    temperatures = np.empty(len(sample_steps), dtype=np.float64)

    t0 = time.perf_counter()
    sample_idx = 0
    for step in range(steps + 1):
        if step == sample_steps[sample_idx]:
            positions[sample_idx] = atoms.get_positions()
            velocities[sample_idx] = atoms.get_velocities()
            energies[sample_idx] = atoms.get_potential_energy()
            forces[sample_idx] = atoms.get_forces()
            temperatures[sample_idx] = atoms.get_temperature()
            sample_idx += 1
        if step < steps:
            dyn.run(1)
    elapsed_s = time.perf_counter() - t0
    return {
        "sample_steps": np.asarray(sample_steps, dtype=np.int64),
        "positions": positions,
        "velocities": velocities,
        "energies": energies,
        "forces": forces,
        "temperatures": temperatures,
        "elapsed_s": elapsed_s,
    }


def save_npz(path: Path, atoms: Atoms, data: dict[str, np.ndarray | float]) -> None:
    np.savez_compressed(
        path,
        z=atoms.get_atomic_numbers().astype(np.int16),
        cell=atoms.get_cell().array.astype(np.float64),
        positions=data["positions"],
        velocities=data["velocities"],
        energies=data["energies"],
        forces=data["forces"],
        temperatures=data["temperatures"],
        sample_steps=data["sample_steps"],
        elapsed_s=float(data["elapsed_s"]),
    )


def load_reference(path: Path) -> dict[str, np.ndarray]:
    blob = np.load(path)
    return {
        "z": blob["z"].astype(int),
        "cell": blob["cell"],
        "positions": blob["positions"],
        "velocities": blob["velocities"],
        "energies": blob["energies"],
        "forces": blob["forces"],
        "temperatures": blob["temperatures"],
        "sample_steps": blob["sample_steps"],
    }


def evaluate_on_frames(calc, ref: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    atoms = Atoms(numbers=ref["z"], positions=ref["positions"][0], cell=ref["cell"], pbc=False)
    atoms.calc = calc
    nsamples = ref["positions"].shape[0]
    energies = np.empty(nsamples, dtype=np.float64)
    forces = np.empty_like(ref["forces"])
    t0 = time.perf_counter()
    for idx in range(nsamples):
        atoms.set_positions(ref["positions"][idx])
        energies[idx] = atoms.get_potential_energy()
        forces[idx] = atoms.get_forces()
    return energies, forces, time.perf_counter() - t0


def summarize_pair(
    *,
    case: str,
    native: dict[str, np.ndarray],
    ictd: dict[str, np.ndarray | float],
    same_frame_e: np.ndarray,
    same_frame_f: np.ndarray,
    same_frame_elapsed_s: float,
    dt_fs: float,
) -> dict[str, float | int | str]:
    de = same_frame_e - native["energies"]
    df = same_frame_f - native["forces"]
    dpos = np.asarray(ictd["positions"]) - native["positions"]
    dvel = np.asarray(ictd["velocities"]) - native["velocities"]
    natoms = int(native["z"].shape[0])
    steps = int(native["sample_steps"][-1])
    nsamples = int(native["positions"].shape[0])
    return {
        "case": case,
        "atoms": natoms,
        "steps": steps,
        "recorded_frames": nsamples,
        "record_stride": int(native["sample_steps"][1] - native["sample_steps"][0]) if nsamples > 1 else steps,
        "dt_fs": float(dt_fs),
        "same_frame_energy_abs_max_eV": float(np.max(np.abs(de))),
        "same_frame_energy_abs_rms_eV": float(np.sqrt(np.mean(de**2))),
        "same_frame_energy_abs_max_meV_per_atom": float(1000.0 * np.max(np.abs(de)) / natoms),
        "same_frame_energy_rel_max": float(np.max(np.abs(de)) / max(np.max(np.abs(native["energies"])), 1e-30)),
        "same_frame_force_abs_max_eV_A": float(np.max(np.abs(df))),
        "same_frame_force_rms_eV_A": float(np.sqrt(np.mean(df**2))),
        "same_frame_force_rel_rms": float(
            np.sqrt(np.mean(df**2)) / max(np.sqrt(np.mean(native["forces"] ** 2)), 1e-30)
        ),
        "independent_traj_position_rms_max_A": float(np.max(np.sqrt(np.mean(dpos**2, axis=(1, 2))))),
        "independent_traj_position_abs_max_A": float(np.max(np.abs(dpos))),
        "independent_traj_velocity_rms_max_A_fs_units": float(np.max(np.sqrt(np.mean(dvel**2, axis=(1, 2))))),
        "native_elapsed_s": float(native.get("elapsed_s", np.nan)),
        "ictd_elapsed_s": float(ictd["elapsed_s"]),
        "same_frame_eval_elapsed_s": float(same_frame_elapsed_s),
        "native_ms_per_step_including_eval": float(1000.0 * native.get("elapsed_s", np.nan) / max(steps, 1)),
        "ictd_ms_per_step_including_eval": float(1000.0 * float(ictd["elapsed_s"]) / max(steps, 1)),
        "same_frame_ms_per_frame": float(1000.0 * same_frame_elapsed_s / max(nsamples, 1)),
        "native_energy0_eV": float(native["energies"][0]),
        "native_energy_final_eV": float(native["energies"][-1]),
        "ictd_energy0_eV": float(np.asarray(ictd["energies"])[0]),
        "ictd_energy_final_eV": float(np.asarray(ictd["energies"])[-1]),
        "native_temp_mean_K": float(np.mean(native["temperatures"])),
        "ictd_temp_mean_K": float(np.mean(np.asarray(ictd["temperatures"]))),
    }


def selected_cases(raw: str | None) -> Iterable[str]:
    if not raw:
        return DEFAULT_CASES.keys()
    cases = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(cases) - set(DEFAULT_CASES))
    if unknown:
        raise ValueError(f"unknown case(s): {unknown}; available={sorted(DEFAULT_CASES)}")
    return cases


def case_steps(args: argparse.Namespace, case: str) -> int:
    return int(args.steps) if args.steps is not None else int(DEFAULT_CASES[case]["steps"])


def run_native(args: argparse.Namespace) -> None:
    from mace.calculators import MACECalculator

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    calc = MACECalculator(model_paths=args.mace_model, device=args.device, default_dtype=args.dtype)
    summaries = []
    for case in selected_cases(args.cases):
        atoms = make_atoms(case, args.spacing)
        init_velocities(atoms, case=case, seed=args.seed, temperature_k=args.temperature_k)
        steps = case_steps(args, case)
        data = run_md(atoms=atoms, calc=calc, steps=steps, dt_fs=args.dt_fs, record_stride=args.record_stride)
        path = outdir / f"native_{args.dtype}_{case}.npz"
        save_npz(path, atoms, data)
        summary = {
            "engine": f"native_mace_{args.dtype}",
            "case": case,
            "atoms": len(atoms),
            "steps": steps,
            "recorded_frames": int(np.asarray(data["sample_steps"]).shape[0]),
            "record_stride": int(args.record_stride),
            "elapsed_s": float(data["elapsed_s"]),
            "ms_per_step_including_eval": float(1000.0 * float(data["elapsed_s"]) / max(steps, 1)),
            "energy0_eV": float(np.asarray(data["energies"])[0]),
            "energy_final_eV": float(np.asarray(data["energies"])[-1]),
            "force_rms0_eV_A": float(np.sqrt(np.mean(np.asarray(data["forces"])[0] ** 2))),
            "temp_mean_K": float(np.mean(np.asarray(data["temperatures"]))),
            "output": str(path),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    (outdir / f"native_{args.dtype}_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))


def run_ictd(args: argparse.Namespace) -> None:
    import torch
    from mace_ictc.evaluation.calculator import MyE3NNCalculator
    from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = []

    obj = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.ictd_checkpoint,
        element_types=[item.strip() for item in args.elements.split(",") if item.strip()],
        device=args.device,
    )
    aek = obj.wrapper.atomic_energy_keys.detach().cpu().tolist()
    aev = obj.wrapper.atomic_energy_values.detach().cpu().tolist()
    e0 = {int(k): float(v) for k, v in zip(aek, aev)}

    for case in selected_cases(args.cases):
        native_path = outdir / f"native_{args.dtype}_{case}.npz"
        native_ref = load_reference(native_path)
        calc = MyE3NNCalculator(obj.wrapper.model, e0, torch.device(args.device), float(obj.rcutfac))

        atoms = Atoms(
            numbers=native_ref["z"],
            positions=native_ref["positions"][0],
            cell=native_ref["cell"],
            pbc=False,
        )
        atoms.set_velocities(native_ref["velocities"][0])
        data = run_md(
            atoms=atoms,
            calc=calc,
            steps=int(native_ref["sample_steps"][-1]),
            dt_fs=args.dt_fs,
            record_stride=args.record_stride,
        )
        save_npz(outdir / f"ictd_{args.dtype}_{case}.npz", atoms, data)

        same_e, same_f, same_elapsed = evaluate_on_frames(calc, native_ref)
        np.savez_compressed(outdir / f"ictd_on_native_{args.dtype}_{case}.npz", energies=same_e, forces=same_f)
        native_with_elapsed = dict(native_ref)
        native_blob = np.load(native_path)
        native_with_elapsed["elapsed_s"] = float(native_blob["elapsed_s"])
        summary = summarize_pair(
            case=case,
            native=native_with_elapsed,
            ictd=data,
            same_frame_e=same_e,
            same_frame_f=same_f,
            same_frame_elapsed_s=same_elapsed,
            dt_fs=args.dt_fs,
        )
        summaries.append(summary)
        (outdir / f"compare_{args.dtype}_{case}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, sort_keys=True), flush=True)
    (outdir / f"compare_{args.dtype}_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["native", "ictd"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dt-fs", type=float, default=0.25)
    parser.add_argument("--temperature-k", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--spacing", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=None, help="override the default step count for all selected cases")
    parser.add_argument("--record-stride", type=int, default=1, help="save/evaluate every N integration steps")
    parser.add_argument("--mace-model", default="/home/ylzhang/.cache/mace/MACE-OFF23_small.model")
    parser.add_argument("--ictd-checkpoint", default="/tmp/mace_ictc_pretrained/off23_small_ictd_bridge_u_float64.pth")
    parser.add_argument("--elements", default="H,C,N,O,F,P,S,Cl,Br,I")
    args = parser.parse_args()
    if args.engine == "native":
        run_native(args)
    else:
        run_ictd(args)


if __name__ == "__main__":
    main()
