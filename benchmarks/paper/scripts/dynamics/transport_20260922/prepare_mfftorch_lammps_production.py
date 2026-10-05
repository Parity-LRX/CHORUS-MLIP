#!/usr/bin/env python3
"""Prepare a restartable LAMMPS production trajectory for an MFF-Torch model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation


MASSES = {
    "H": 1.008,
    "Li": 6.94,
    "O": 15.999,
    "P": 30.973761998,
    "S": 32.06,
}


def parse_repeat(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 3 or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("repeat must contain three positive integers")
    return values


def restore_ase_state(path: Path) -> tuple[Atoms, int]:
    payload = json.loads(path.read_text())
    atoms = Atoms(
        numbers=payload["atomic_numbers"],
        positions=payload["positions_angstrom"],
        cell=payload["cell_angstrom"],
        pbc=payload["pbc"],
    )
    atoms.set_velocities(np.asarray(payload["velocities_angstrom_per_ase_time"]))
    return atoms, int(payload["completed_steps"])


def latest_binary_restart(output_dir: Path) -> tuple[Path | None, int | None]:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("restart.*.bin"):
        try:
            step = int(path.name.split(".")[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, path))
    if not candidates:
        return None, None
    step, path = max(candidates)
    return path, step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-structure", type=Path, required=True)
    parser.add_argument("--ase-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument("--temperature-k", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--repeat", type=parse_repeat, default=(2, 2, 2))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0)
    parser.add_argument(
        "--pair-style", choices=("mff/torch", "mff/torch/kk"), default="mff/torch"
    )
    parser.add_argument("--dump-interval", type=int, default=100)
    parser.add_argument("--restart-interval", type=int, default=5_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lammps_restart, lammps_step = latest_binary_restart(args.output_dir)
    source = "lammps-binary-restart" if lammps_restart else "fresh"

    if lammps_restart is not None:
        completed = int(lammps_step)
        read_command = f"read_restart {lammps_restart}"
    else:
        if args.ase_state.exists():
            atoms, completed = restore_ase_state(args.ase_state)
            source = "ase-json-restart"
        else:
            atoms = read(args.initial_structure, index=args.frame_index).repeat(args.repeat)
            MaxwellBoltzmannDistribution(
                atoms,
                temperature_K=args.temperature_k,
                rng=np.random.RandomState(args.seed),
            )
            Stationary(atoms)
            ZeroRotation(atoms)
            completed = 0
        if not bool(np.all(atoms.pbc)):
            raise RuntimeError("production MD requires full periodic boundaries")
        if abs(float(np.linalg.det(atoms.cell.array))) < 1.0e-8:
            raise RuntimeError("initial cell is singular")
        data_path = args.output_dir / "start_state.data"
        write(
            data_path,
            atoms,
            format="lammps-data",
            atom_style="atomic",
            specorder=list(args.elements),
            units="metal",
            velocities=True,
        )
        read_command = f"read_data {data_path}"

    if completed >= args.steps:
        (args.output_dir / "DONE").touch()
        print(json.dumps({"already_complete": True, "completed_steps": completed}))
        return

    missing_masses = sorted(set(args.elements) - MASSES.keys())
    if missing_masses:
        raise ValueError(f"missing LAMMPS masses for elements: {missing_masses}")
    mass_lines = "\n".join(
        f"mass {index} {MASSES[element]:.12g}"
        for index, element in enumerate(args.elements, start=1)
    )
    element_map = " ".join(args.elements)
    remaining = args.steps - completed
    input_path = args.output_dir / "in.mfftorch_production"
    input_path.write_text(
        f"""units metal
atom_style atomic
boundary p p p
atom_modify map yes

{read_command}
{mass_lines}
reset_timestep {completed}
timestep 0.001

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
pair_style {args.pair_style} {args.cutoff_angstrom:.8g} cuda
pair_coeff * * {args.model} {element_map}

thermo_style custom step atoms temp pe ke etotal press fmax
thermo 10
thermo_modify flush yes
dump dtraj all custom {args.dump_interval} {args.output_dir / 'trajectory_lammps.lammpstrj'} id type xu yu zu vx vy vz
dump_modify dtraj sort id append yes element {element_map}
restart {args.restart_interval} {args.output_dir / 'restart.*.bin'}

fix thermostat all nvt temp {args.temperature_k:.8g} {args.temperature_k:.8g} 0.02 tchain 1
run {remaining}
write_data {args.output_dir / 'final_state.data'} nocoeff
print "MFF_PRODUCTION_DONE step=$(step)"
"""
    )
    manifest = {
        "engine": "LAMMPS MFF-Torch",
        "pair_style": args.pair_style,
        "model": str(args.model.resolve()),
        "initial_structure": str(args.initial_structure.resolve()),
        "source": source,
        "source_step": completed,
        "target_steps": args.steps,
        "remaining_steps": remaining,
        "temperature_k": args.temperature_k,
        "seed": args.seed,
        "time_step_fs": 1.0,
        "thermostat": "LAMMPS fix nvt",
        "thermostat_damping_fs": 20.0,
        "thermostat_chain_length": 1,
        "elements": list(args.elements),
        "strict_fp32": True,
        "tf32": False,
    }
    (args.output_dir / "manifest_lammps.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
