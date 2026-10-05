#!/usr/bin/env python3
"""Run a restartable CarNet-style NVT trajectory with a DPA-4 checkpoint.

The integration protocol matches ``run_carnet_md.py``: 1 fs time steps, a
Nosé--Hoover-chain thermostat with 20 fs damping, and one thermostat chain.
The checkpoint is selected before MD from the complete validation scan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from deepmd.calculator import DP


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
    parser.add_argument("--nlist-backend", default="auto")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("steps must be positive")
    np.random.seed(args.seed)

    # DeepMD/DPA-4 was trained and evaluated without AMP.  Keep all optional
    # TensorFloat-32 paths disabled for the production trajectory as well.
    try:
        import torch

        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(args.seed)
    except ImportError:
        pass

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

    atoms.calc = DP(model=args.checkpoint, nlist_backend=args.nlist_backend)
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
    logger = MDLogger(
        dynamics,
        atoms,
        str(args.output_dir / "thermo.log"),
        header=not resumed,
        stress=False,
        peratom=False,
        mode="a" if resumed else "w",
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
        "engine": "DeepMD/DPA-4",
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
        "amp": False,
        "tf32": False,
        "nlist_backend": args.nlist_backend,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

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
