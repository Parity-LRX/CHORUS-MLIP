#!/usr/bin/env python3
"""Benchmark NequIP-ICTC through the CHORUS make_fx/Inductor force path."""

from __future__ import annotations

import argparse
import json
import time

import torch

from chorus.training.makefx_compile import trace_and_compile_force
from nequip.data import AtomicData, AtomicDataDict, Collater, dataset_from_config
from nequip.model import model_from_config
from nequip.scripts.train import default_config
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", choices=("inference", "training"), required=True)
    return parser.parse_args()


def sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260616)
    torch.cuda.manual_seed_all(20260616)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    config = Config.from_file(args.config, defaults=default_config)
    interaction_backend = str(config.get("interaction_backend", "e3nn")).lower()
    config["allow_tf32"] = False
    config["default_dtype"] = "float32"
    config["model_dtype"] = "float32"
    # Flatten energy -> force ourselves so make_fx can expose the first
    # derivative as ordinary FX nodes before Inductor compiles it.
    config["model_builders"] = [
        builder
        for builder in config["model_builders"]
        if str(builder) != "ForceOutput"
    ]
    _set_global_options(config)
    dataset = dataset_from_config(config, prefix="dataset")
    collater = Collater.for_dataset(dataset, exclude_keys=[])
    batch = collater.collate(
        [dataset[index] for index in range(args.batch_size)]
    ).to(device)
    data = AtomicData.to_AtomicDataDict(batch)

    torch.manual_seed(20260616)
    torch.cuda.manual_seed_all(20260616)
    model = model_from_config(config, initialize=True, dataset=dataset).to(device)
    model.train(args.mode == "training")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    position_key = AtomicDataDict.POSITIONS_KEY
    other_tensor_keys = sorted(
        (
            key
            for key, value in data.items()
            if key != position_key and torch.is_tensor(value)
        ),
        key=str,
    )
    static_values = {
        key: value for key, value in data.items() if not torch.is_tensor(value)
    }
    inputs = (data[position_key],) + tuple(data[key] for key in other_tensor_keys)
    training = args.mode == "training"

    def compute(pos: torch.Tensor, *rest: torch.Tensor):
        positions = pos.detach().requires_grad_(True)
        local_data = dict(static_values)
        local_data[position_key] = positions
        local_data.update(zip(other_tensor_keys, rest))
        output = model(local_data)
        total_energy = output[AtomicDataDict.TOTAL_ENERGY_KEY]
        gradient = torch.autograd.grad(
            total_energy.sum(), positions, create_graph=training
        )[0]
        return total_energy, -gradient

    eager_energy, eager_forces = compute(*inputs)
    sync()
    started = time.perf_counter()
    compiled = trace_and_compile_force(
        model,
        inputs,
        training=training,
        compute_fn=compute,
        compile_dynamic_shapes=False,
    )
    compiled_energy, compiled_forces = compiled(*inputs)
    sync()
    compile_seconds = time.perf_counter() - started
    energy_error = float(
        (compiled_energy.detach() - eager_energy.detach()).abs().max().item()
    )
    force_error = float(
        (compiled_forces.detach() - eager_forces.detach()).abs().max().item()
    )
    energy_scale = max(float(eager_energy.detach().abs().max().item()), 1.0)
    force_scale = max(float(eager_forces.detach().abs().max().item()), 1.0)
    if energy_error / energy_scale > 2.0e-5 or force_error / force_scale > 2.0e-4:
        raise RuntimeError(
            "make_fx numerical mismatch: "
            f"energy={energy_error / energy_scale:.3e}, "
            f"force={force_error / force_scale:.3e}"
        )

    def step():
        model.zero_grad(set_to_none=True)
        energy, forces = compiled(*inputs)
        if training:
            loss = forces.square().mean() + energy.square().mean()
            loss.backward()
        return energy, forces

    for _ in range(args.warmup):
        compiled_energy, compiled_forces = step()
    sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.iterations):
        compiled_energy, compiled_forces = step()
    sync()
    elapsed = time.perf_counter() - started
    milliseconds = 1.0e3 * elapsed / args.iterations
    result = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "atoms": int(batch.pos.shape[0]),
        "edges": int(batch.edge_index.shape[1]),
        "parameters": parameter_count,
        "iterations": args.iterations,
        "milliseconds_per_iteration": milliseconds,
        "iterations_per_second": args.iterations / elapsed,
        "atoms_per_second": int(batch.pos.shape[0]) * args.iterations / elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024.0**2),
        "compile_seconds": compile_seconds,
        "energy_sum": compiled_energy.detach().double().sum().item(),
        "force_square_sum": compiled_forces.detach().double().square().sum().item(),
        "energy_relative_error": energy_error / energy_scale,
        "force_relative_error": force_error / force_scale,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": "float32",
        "tf32": False,
        "backend": f"NequIP-{interaction_backend.upper()} MakeFX/Inductor",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
