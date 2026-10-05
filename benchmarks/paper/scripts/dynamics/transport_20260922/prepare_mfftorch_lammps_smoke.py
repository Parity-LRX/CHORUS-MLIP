#!/usr/bin/env python3
"""Prepare and audit a same-frame MFF-Torch/LAMMPS deployment smoke test.

The ``prepare`` command writes a repeated periodic structure as LAMMPS data,
evaluates the original checkpoint on exactly that structure, and emits the
LAMMPS input used for both a run-0 parity check and a short NVT timing run.
The ``compare`` command reads the LAMMPS outputs and writes a machine-readable
parity/timing report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from ase.data import atomic_numbers
from ase.io import read, write

from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from run_carnet_md import ChorusASECalculator, parse_repeat


def _prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atoms = read(args.initial_structure, index=args.frame_index).repeat(args.repeat)
    if not bool(np.all(atoms.pbc)):
        raise RuntimeError("the LAMMPS smoke structure must be fully periodic")
    if abs(float(np.linalg.det(atoms.cell.array))) < 1.0e-8:
        raise RuntimeError("the LAMMPS smoke structure has a singular cell")

    data_path = args.output_dir / "system.data"
    reference_path = args.output_dir / "eager_reference.npz"
    input_path = args.output_dir / "in.mfftorch_smoke"
    write(
        data_path,
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=list(args.elements),
    )

    # ``write_lammps_data`` rotates a general triclinic ASE cell into the
    # restricted LAMMPS prism convention.  Evaluate the reference after a
    # read-back so that both force arrays use that same Cartesian frame.
    lammps_atoms = read(
        data_path,
        format="lammps-data",
        style="atomic",
        Z_of_type={
            index: atomic_numbers[element]
            for index, element in enumerate(args.elements, start=1)
        },
    )
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    deployed = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        element_types=list(args.elements),
        device=str(device),
        avg_num_neighbors=args.avg_num_neighbors,
    )
    lammps_atoms.calc = ChorusASECalculator(deployed, device)
    energy = float(lammps_atoms.get_potential_energy())
    forces = np.asarray(lammps_atoms.get_forces(), dtype=np.float64)
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError("checkpoint produced a non-finite eager reference")

    np.savez(
        reference_path,
        energy_ev=np.asarray(energy, dtype=np.float64),
        forces_ev_per_angstrom=forces,
        atomic_numbers=np.asarray(lammps_atoms.numbers, dtype=np.int64),
        positions_angstrom=np.asarray(lammps_atoms.positions, dtype=np.float64),
    )

    masses = {
        "H": 1.008,
        "Li": 6.94,
        "O": 15.999,
        "P": 30.973761998,
        "S": 32.06,
    }
    missing_masses = sorted(set(args.elements) - masses.keys())
    if missing_masses:
        raise ValueError(f"missing LAMMPS masses for elements: {missing_masses}")
    mass_lines = "\n".join(
        f"mass {index} {masses[element]:.6f}"
        for index, element in enumerate(args.elements, start=1)
    )
    element_map = " ".join(args.elements)
    input_path.write_text(
        f"""units metal
atom_style atomic
boundary p p p
atom_modify map yes

read_data {data_path}
{mass_lines}

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style {args.pair_style} {args.cutoff_angstrom:.8g} cuda
pair_coeff * * {args.model} {element_map}

thermo_style custom step atoms temp pe ke etotal press fmax
thermo 10
run 0
variable parity_pe equal pe
print \"MFF_PARITY_PE ${{parity_pe}}\"
write_dump all custom {args.output_dir / 'parity.dump'} id type x y z fx fy fz modify sort id

reset_timestep 0
velocity all create {args.temperature_k:.8g} {args.seed} mom yes rot yes dist gaussian
fix thermostat all nvt temp {args.temperature_k:.8g} {args.temperature_k:.8g} 0.02 tchain 1
run {args.md_steps}
variable final_pe equal pe
print \"MFF_FINAL_PE ${{final_pe}}\"
"""
    )
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": str(args.model.resolve()),
        "initial_structure": str(args.initial_structure.resolve()),
        "repeat": list(args.repeat),
        "natoms": len(atoms),
        "elements": list(args.elements),
        "avg_num_neighbors": args.avg_num_neighbors,
        "cutoff_angstrom": args.cutoff_angstrom,
        "pair_style": args.pair_style,
        "temperature_k": args.temperature_k,
        "md_steps": args.md_steps,
        "strict_fp32": True,
        "tf32": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": True, **manifest}, indent=2), flush=True)


def _read_dump(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    try:
        marker = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    except StopIteration as exc:
        raise RuntimeError(f"no atom block in {path}") from exc
    header = lines[marker].split()[2:]
    columns = {name: index for index, name in enumerate(header)}
    required = ("id", "fx", "fy", "fz")
    missing = [name for name in required if name not in columns]
    if missing:
        raise RuntimeError(f"missing dump columns: {missing}")
    rows: list[tuple[int, float, float, float]] = []
    for line in lines[marker + 1 :]:
        if line.startswith("ITEM:"):
            break
        fields = line.split()
        rows.append(
            (
                int(fields[columns["id"]]),
                float(fields[columns["fx"]]),
                float(fields[columns["fy"]]),
                float(fields[columns["fz"]]),
            )
        )
    rows.sort(key=lambda row: row[0])
    return np.asarray([row[1:] for row in rows], dtype=np.float64)


def _match_float(pattern: str, text: str, label: str) -> float:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError(f"could not find {label} in LAMMPS log")
    return float(matches[-1])


def _compare(args: argparse.Namespace) -> None:
    reference = np.load(args.output_dir / "eager_reference.npz")
    eager_energy = float(reference["energy_ev"])
    eager_forces = np.asarray(reference["forces_ev_per_angstrom"], dtype=np.float64)
    lammps_forces = _read_dump(args.output_dir / "parity.dump")
    if eager_forces.shape != lammps_forces.shape:
        raise RuntimeError(
            f"force shape mismatch: eager={eager_forces.shape}, LAMMPS={lammps_forces.shape}"
        )
    log_text = args.log.read_text()
    lammps_energy = _match_float(
        r"^MFF_PARITY_PE\s+([-+0-9.eE]+)\s*$", log_text, "run-0 potential energy"
    )
    loop_seconds = _match_float(
        r"Loop time of\s+([-+0-9.eE]+)\s+on", log_text, "LAMMPS loop time"
    )
    force_delta = lammps_forces - eager_forces
    natoms = eager_forces.shape[0]
    report = {
        "natoms": natoms,
        "eager_energy_ev": eager_energy,
        "lammps_energy_ev": lammps_energy,
        "energy_abs_error_ev": abs(lammps_energy - eager_energy),
        "energy_abs_error_mev_per_atom": abs(lammps_energy - eager_energy) * 1000.0 / natoms,
        "force_mae_mev_per_angstrom": float(np.mean(np.abs(force_delta)) * 1000.0),
        "force_rmse_mev_per_angstrom": float(np.sqrt(np.mean(force_delta**2)) * 1000.0),
        "force_max_abs_mev_per_angstrom": float(np.max(np.abs(force_delta)) * 1000.0),
        "md_steps": args.md_steps,
        "loop_seconds": loop_seconds,
        "steps_per_second": args.md_steps / loop_seconds,
        "atom_steps_per_second": natoms * args.md_steps / loop_seconds,
    }
    passed = (
        report["energy_abs_error_mev_per_atom"] <= args.energy_tol_mev_per_atom
        and report["force_mae_mev_per_angstrom"] <= args.force_mae_tol_mev_per_angstrom
        and report["force_max_abs_mev_per_angstrom"] <= args.force_max_tol_mev_per_angstrom
    )
    report["tolerances"] = {
        "energy_mev_per_atom": args.energy_tol_mev_per_atom,
        "force_mae_mev_per_angstrom": args.force_mae_tol_mev_per_angstrom,
        "force_max_abs_mev_per_angstrom": args.force_max_tol_mev_per_angstrom,
    }
    report["passed"] = passed
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit("MFF-Torch/LAMMPS parity gate failed")
    (args.output_dir / "DONE").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--initial-structure", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--elements", nargs="+", default=["H", "O"])
    prepare.add_argument("--repeat", type=parse_repeat, default=(2, 2, 2))
    prepare.add_argument("--frame-index", type=int, default=0)
    prepare.add_argument("--avg-num-neighbors", type=float, required=True)
    prepare.add_argument("--cutoff-angstrom", type=float, default=5.0)
    prepare.add_argument("--pair-style", choices=["mff/torch", "mff/torch/kk"], default="mff/torch")
    prepare.add_argument("--temperature-k", type=float, default=300.0)
    prepare.add_argument("--seed", type=int, default=20260616)
    prepare.add_argument("--md-steps", type=int, default=1000)
    prepare.add_argument("--device", default="cuda")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--log", type=Path, required=True)
    compare.add_argument("--md-steps", type=int, default=1000)
    compare.add_argument("--energy-tol-mev-per-atom", type=float, default=2.0)
    compare.add_argument("--force-mae-tol-mev-per-angstrom", type=float, default=2.0)
    compare.add_argument("--force-max-tol-mev-per-angstrom", type=float, default=10.0)

    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
