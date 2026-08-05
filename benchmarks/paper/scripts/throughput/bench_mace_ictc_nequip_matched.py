#!/usr/bin/env python3
"""Match the NequIP full-model throughput protocol with MACE-ICTC.

The workload is a batch of 16 disconnected nine-atom graphs.  Every graph has
all directed non-self edges, giving 144 atoms and 1152 edges in total, exactly
matching ``NequIP-CHORUS/scripts/benchmark_full_model_step.py`` on xxMD-MAL.
Inference evaluates energy and conservative forces.  Training evaluates the
same quantities and performs the force double backward, without an optimizer
update, matching the NequIP timing boundary.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from chorus.bench.synthetic_workloads import make_graph
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from chorus.training.makefx_compile import trace_and_compile_force
from chorus.training.train_loop import disable_tf32


MODE_SPECS = {
    "off": {
        "phase_mode": "none",
        "phase_amplitude": "unit",
        "phase_placement": "post-product",
        "phase_scope": "final",
    },
    "final": {
        "phase_mode": "final-full-l-residual",
        "phase_amplitude": "softplus",
        "phase_placement": "pre-product-full-l",
        "phase_scope": "final",
    },
    "persistent": {
        "phase_mode": "final-full-l-residual",
        "phase_amplitude": "softplus",
        "phase_placement": "pre-product-full-l",
        "phase_scope": "persistent",
    },
}


def sync() -> None:
    torch.cuda.synchronize()


def matched_graph(*, device: torch.device, nequip_config: Path) -> object:
    from ase.data import atomic_numbers as ase_atomic_numbers
    from nequip.data import AtomicData, AtomicDataDict, Collater, dataset_from_config
    from nequip.scripts.train import default_config
    from nequip.utils import Config
    from nequip.utils._global_options import _set_global_options

    config = Config.from_file(str(nequip_config), defaults=default_config)
    config["allow_tf32"] = False
    config["default_dtype"] = "float32"
    config["model_dtype"] = "float32"
    _set_global_options(config)
    dataset = dataset_from_config(config, prefix="dataset")
    collater = Collater.for_dataset(dataset, exclude_keys=[])
    batch_data = collater.collate([dataset[index] for index in range(16)]).to(device)
    data = AtomicData.to_AtomicDataDict(batch_data)
    positions = data[AtomicDataDict.POSITIONS_KEY].to(dtype=torch.float32)
    edge_index = data[AtomicDataDict.EDGE_INDEX_KEY].to(dtype=torch.long)
    batch = data[AtomicDataDict.BATCH_KEY].to(dtype=torch.long)
    graphs = int(batch_data.ptr.numel() - 1)
    cell = data.get(AtomicDataDict.CELL_KEY)
    if cell is None:
        cell = (
            torch.eye(3, dtype=torch.float32, device=device)
            .mul(100.0)
            .repeat(graphs, 1, 1)
        )
    else:
        cell = cell.to(dtype=torch.float32)
    unit_shifts = data.get(AtomicDataDict.EDGE_CELL_SHIFT_KEY)
    if unit_shifts is None:
        unit_shifts = torch.zeros(
            edge_index.shape[1], 3, dtype=torch.float32, device=device
        )
    else:
        unit_shifts = unit_shifts.to(dtype=torch.float32)
    atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1).to(dtype=torch.long)
    symbols = list(config["chemical_symbols"])
    type_to_z = torch.tensor(
        [ase_atomic_numbers[symbol] for symbol in symbols],
        dtype=torch.long,
        device=device,
    )
    atomic_numbers = type_to_z.index_select(0, atom_types)
    atoms = int(positions.shape[0])
    graphs = int(cell.shape[0])
    if atoms != 144 or int(edge_index.shape[1]) != 1152 or graphs != 16:
        raise RuntimeError(
            "NequIP reference batch changed: expected 144 atoms, 1152 edges, "
            f"16 graphs; got {atoms}, {int(edge_index.shape[1])}, {graphs}"
        )
    base = make_graph(
        atoms=atoms,
        avg_degree=8,
        dtype=torch.float32,
        device=device,
        seed=20260616,
    )
    edge_src, edge_dst = edge_index[0], edge_index[1]
    ptr = batch_data.ptr.to(device=device, dtype=torch.long)
    return dataclasses.replace(
        base,
        pos=positions,
        atomic_numbers=atomic_numbers,
        batch=batch,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_index=edge_index,
        unit_shifts=unit_shifts,
        shifts=unit_shifts,
        cell=cell,
        ptr=ptr,
        energy_ref=torch.zeros(graphs, device=device),
        stress_ref=torch.zeros(graphs, 3, 3, device=device),
    )


def force_compute_fn(model: torch.nn.Module, *, training: bool):
    def compute(pos, atomic_numbers, batch, edge_src, edge_dst, shifts, cell):
        positions = pos.detach().requires_grad_(True)
        energy_atom = model(
            positions,
            atomic_numbers,
            batch,
            edge_src,
            edge_dst,
            shifts,
            cell,
        )
        if isinstance(energy_atom, tuple):
            energy_atom = energy_atom[0]
        gradient = torch.autograd.grad(
            energy_atom.sum(),
            positions,
            create_graph=training,
        )[0]
        return energy_atom.reshape(-1), -gradient

    return compute


def benchmark_once(
    *,
    scope: str,
    task: str,
    graph,
    checkpoint: Path,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    torch.manual_seed(20260616)
    torch.cuda.manual_seed_all(20260616)
    loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint,
        element_types=["H", "C", "N", "O"],
        device="cuda",
    )
    model = loaded.wrapper.model.to(device="cuda", dtype=torch.float32)
    if task == "training":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    model.train(task == "training")
    inputs = (
        graph.pos,
        graph.atomic_numbers,
        graph.batch,
        graph.edge_src,
        graph.edge_dst,
        graph.unit_shifts,
        graph.cell,
    )
    started = time.perf_counter()
    compiled = trace_and_compile_force(
        model,
        inputs,
        training=task == "training",
        compute_fn=force_compute_fn(model, training=task == "training"),
        compile_dynamic_shapes=False,
    )
    energy_atom, forces = compiled(*inputs)
    sync()
    compile_seconds = time.perf_counter() - started
    num_graphs = int(graph.cell.shape[0])

    def step():
        model.zero_grad(set_to_none=True)
        energy_atom, forces = compiled(*inputs)
        if task == "training":
            total_energy = torch.zeros(
                num_graphs,
                dtype=energy_atom.dtype,
                device=energy_atom.device,
            ).index_add(0, graph.batch, energy_atom)
            loss = forces.square().mean() + total_energy.square().mean()
            loss.backward()
        return energy_atom, forces

    for _ in range(warmup):
        energy_atom, forces = step()
    sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(iterations):
        energy_atom, forces = step()
    sync()
    elapsed = time.perf_counter() - started
    milliseconds = 1.0e3 * elapsed / iterations
    result = {
        "scope": scope,
        "mode": task,
        "atoms": int(graph.pos.shape[0]),
        "edges": int(graph.edge_src.numel()),
        "batch_size": num_graphs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "iterations": iterations,
        "milliseconds_per_iteration": milliseconds,
        "iterations_per_second": iterations / elapsed,
        "atoms_per_second": int(graph.pos.shape[0]) * iterations / elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024.0**2),
        "compile_seconds": compile_seconds,
        "energy_sum": energy_atom.detach().double().sum().item(),
        "force_square_sum": forces.detach().double().square().sum().item(),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": "float32",
        "tf32": False,
        "backend": "MACE-ICTC MakeFX/Inductor",
        "checkpoint": str(checkpoint),
    }
    del compiled, model, energy_atom, forces
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-off", type=Path, required=True)
    parser.add_argument("--checkpoint-final", type=Path, required=True)
    parser.add_argument("--checkpoint-persistent", type=Path, required=True)
    parser.add_argument("--nequip-config", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    disable_tf32()
    torch.set_float32_matmul_precision("highest")
    graph = matched_graph(
        device=torch.device("cuda"),
        nequip_config=args.nequip_config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    checkpoints = {
        "off": args.checkpoint_off,
        "final": args.checkpoint_final,
        "persistent": args.checkpoint_persistent,
    }
    for repeat in range(1, args.repeats + 1):
        for scope in MODE_SPECS:
            for task in ("inference", "training"):
                result = benchmark_once(
                    scope=scope,
                    task=task,
                    graph=graph,
                    checkpoint=checkpoints[scope],
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                record = {"repeat": repeat, **result}
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    jsonl = args.output_dir / "results.jsonl"
    jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    summary: dict[str, object] = {}
    for scope in MODE_SPECS:
        for task in ("inference", "training"):
            selected = [
                row for row in records
                if row["scope"] == scope and row["mode"] == task
            ]
            summary[f"{scope}/{task}"] = {
                "median_ms": statistics.median(
                    float(row["milliseconds_per_iteration"]) for row in selected
                ),
                "median_iterations_per_second": statistics.median(
                    float(row["iterations_per_second"]) for row in selected
                ),
                "median_atoms_per_second": statistics.median(
                    float(row["atoms_per_second"]) for row in selected
                ),
                "parameters": selected[0]["parameters"],
                "peak_memory_mib": statistics.median(
                    float(row["peak_memory_mib"]) for row in selected
                ),
            }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
