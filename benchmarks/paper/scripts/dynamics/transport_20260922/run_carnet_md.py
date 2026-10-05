#!/usr/bin/env python3
"""Run a restartable CarNet-style NVT trajectory for a CHORUS checkpoint.

The protocol follows the public CarNet examples: a 1 fs time step, a
Nosé--Hoover-chain thermostat with a 20 fs damping time and one thermostat
chain.  The driver deliberately keeps model selection outside MD: the input
checkpoint must already have been selected on the full validation split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from chorus.utils.graph_utils import radius_graph_pbc_gpu
from chorus.utils.tensor_utils import map_tensor_values


class ChorusASECalculator(Calculator):
    """Minimal FP32 ASE calculator around a validation-selected CHORUS model."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, deployed: LAMMPS_MLIAP_MFF, device: torch.device):
        super().__init__()
        self.model = deployed.wrapper.model.to(device).eval()
        self.device = device
        self.dtype = next(self.model.parameters()).dtype
        self.max_radius = float(deployed.rcutfac)
        self.keys = deployed.wrapper.atomic_energy_keys.to(device=device, dtype=torch.long)
        self.values = deployed.wrapper.atomic_energy_values.to(device=device, dtype=self.dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = torch.as_tensor(
            self.atoms.get_positions(), dtype=self.dtype, device=self.device
        ).clone().requires_grad_(True)
        numbers = torch.as_tensor(
            self.atoms.get_atomic_numbers(), dtype=self.dtype, device=self.device
        )
        cell = torch.as_tensor(
            self.atoms.cell.array, dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        pbc = tuple(bool(value) for value in self.atoms.pbc)
        edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(
            positions, self.max_radius, cell, pbc=pbc
        )
        batch = torch.zeros(len(self.atoms), dtype=torch.long, device=self.device)
        atomic_energies = self.model(
            positions,
            numbers,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
        )
        offset = map_tensor_values(numbers, self.keys, self.values).sum()
        energy = atomic_energies.sum() + offset
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
        self.results["energy"] = float(energy.detach().cpu())
        self.results["forces"] = forces.detach().cpu().numpy().astype(np.float64, copy=False)


def parse_repeat(text: str) -> tuple[int, int, int]:
    parts = tuple(int(value) for value in text.split(","))
    if len(parts) != 3 or any(value < 1 for value in parts):
        raise argparse.ArgumentTypeError("repeat must contain three positive integers")
    return parts


def write_state(path: Path, atoms: Atoms, completed_steps: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "completed_steps": int(completed_steps),
        "positions_angstrom": atoms.get_positions().tolist(),
        "velocities_angstrom_per_ase_time": atoms.get_velocities().tolist(),
        "cell_angstrom": atoms.cell.array.tolist(),
        "atomic_numbers": atoms.numbers.tolist(),
        "pbc": atoms.pbc.tolist(),
    }
    temporary.write_text(json.dumps(payload) + "\n")
    os.replace(temporary, path)


def restore_state(path: Path) -> tuple[Atoms, int]:
    payload = json.loads(path.read_text())
    atoms = Atoms(
        numbers=payload["atomic_numbers"],
        positions=payload["positions_angstrom"],
        cell=payload["cell_angstrom"],
        pbc=payload["pbc"],
    )
    atoms.set_velocities(np.asarray(payload["velocities_angstrom_per_ase_time"]))
    return atoms, int(payload["completed_steps"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-structure", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature-k", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--repeat", type=parse_repeat, default=(1, 1, 1))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--trajectory-interval", type=int, default=100)
    parser.add_argument("--restart-interval", type=int, default=5_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("steps must be positive")
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "restart.json"
    done_path = args.output_dir / "DONE"
    if done_path.exists():
        print(f"already complete: {done_path}", flush=True)
        return

    if state_path.exists():
        atoms, completed = restore_state(state_path)
        resumed = True
    else:
        atoms = read(args.initial_structure, index=args.frame_index)
        atoms = atoms.repeat(args.repeat)
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=args.temperature_k, rng=np.random.RandomState(args.seed)
        )
        Stationary(atoms)
        ZeroRotation(atoms)
        completed = 0
        resumed = False

    if not bool(np.all(atoms.pbc)):
        raise RuntimeError("CarNet-style production MD requires full periodic boundaries")
    if abs(float(np.linalg.det(atoms.cell.array))) < 1.0e-8:
        raise RuntimeError("initial cell is singular")

    device = torch.device(args.device)
    deployed = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        element_types=sorted(set(atoms.get_chemical_symbols())),
        device=str(device),
    )
    atoms.calc = ChorusASECalculator(deployed, device)

    remaining = args.steps - completed
    if remaining <= 0:
        done_path.touch()
        return

    dynamics = NoseHooverChainNVT(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=args.temperature_k,
        tdamp=20.0 * units.fs,
        tchain=1,
    )
    log_mode = "a" if resumed else "w"
    logger = MDLogger(
        dynamics,
        atoms,
        str(args.output_dir / "thermo.log"),
        header=not resumed,
        stress=False,
        peratom=False,
        mode=log_mode,
    )
    dynamics.attach(logger, interval=args.log_interval)
    trajectory = Trajectory(
        str(args.output_dir / "trajectory.traj"),
        mode="a" if resumed else "w",
        atoms=atoms,
    )
    dynamics.attach(trajectory.write, interval=args.trajectory_interval)

    progress_path = args.output_dir / "progress.json"
    started = time.time()

    def checkpoint() -> None:
        current = completed + dynamics.nsteps
        write_state(state_path, atoms, current)
        elapsed = time.time() - started
        progress = {
            "completed_steps": current,
            "target_steps": args.steps,
            "temperature_k": args.temperature_k,
            "seed": args.seed,
            "natoms": len(atoms),
            "elapsed_seconds_this_run": elapsed,
            "steps_per_second_this_run": dynamics.nsteps / elapsed if elapsed > 0 else None,
        }
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(progress), flush=True)

    dynamics.attach(checkpoint, interval=args.restart_interval)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "initial_structure": str(args.initial_structure.resolve()),
        "frame_index": args.frame_index,
        "repeat": list(args.repeat),
        "natoms": len(atoms),
        "temperature_k": args.temperature_k,
        "seed": args.seed,
        "time_step_fs": 1.0,
        "thermostat": "NoseHooverChainNVT",
        "thermostat_damping_fs": 20.0,
        "thermostat_chain_length": 1,
        "target_steps": args.steps,
        "target_time_ps": args.steps / 1000.0,
        "resumed": resumed,
        "strict_fp32": True,
        "tf32": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Force one evaluation before the integrator starts so load/graph errors fail early.
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces())
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError("non-finite initial energy or forces")
    dynamics.run(remaining)
    checkpoint()
    trajectory.close()
    done_path.touch()
    print(f"MD_DONE steps={args.steps} natoms={len(atoms)}", flush=True)


if __name__ == "__main__":
    main()
