"""Matching changing-batch differentiable DPA baseline, with no CUDA Graph."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

os.environ["DP_INTERFACE_PREC"] = "low"
os.environ["DP_COMPILE_INFER"] = "1"
os.environ["DP_TRITON_INFER"] = "2"
os.environ["DP_TF32_INFER"] = "0"
os.environ["DP_CUTE_INFER"] = "0"

from no_cuda_graph import forbid_cuda_graph
import torch
from ase.neighborlist import neighbor_list
from deepmd.pt.model.model import get_model
from deepmd.utils.argcheck import normalize


def main(graph_policy):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--atoms", type=int, required=True)
    p.add_argument("--neighbors", type=int, choices=(32, 50), default=50)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--parameter-targets-only", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    config = normalize(json.loads(args.config.read_text()))
    config["model"]["descriptor"]["use_amp"] = False
    config["model"].update(use_compile=True, enable_tf32=False)
    model = get_model(copy.deepcopy(config["model"]))
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({k.removeprefix("model.Default."): v for k, v in raw.items()
                           if k.startswith("model.Default.")}, strict=True)
    model = model.cuda().train()
    if not model.should_use_compile():
        raise RuntimeError("DPA training compile disabled")
    source = Path(os.environ["SOURCE"])
    script = source / "benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
    spec = importlib.util.spec_from_file_location("training_geometry", script)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    types = {symbol: i for i, symbol in enumerate(config["model"]["type_map"])}
    batches = []
    for index in range(3):
        atoms = common.make_atoms(args.atoms, model.get_rcut(), args.neighbors, 20260728 + index)
        src, dst = neighbor_list("ij", atoms, model.get_rcut(), self_interaction=False)
        if len(src) != args.atoms * args.neighbors:
            raise RuntimeError("Cutoff edge count mismatch")
        symbols = atoms.get_chemical_symbols()
        if index:
            symbols = symbols[-index:] + symbols[:-index]
        coord = torch.as_tensor(atoms.positions, device="cuda", dtype=torch.float32)[None]
        atype = torch.tensor([[types[s] for s in symbols]], device="cuda")
        box = torch.as_tensor(atoms.cell.array, device="cuda", dtype=torch.float32).reshape(1, 9)
        schema = model.build_neighbor_list(coord, atype, box)
        if int(schema.edge_mask.sum()) != len(src):
            raise RuntimeError("Compact DPA edge count mismatch")
        batches.append(schema)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                 lr=1e-5, weight_decay=0.)

    def step(schema):
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_lower(schema.coord, schema.atype, schema.edge_index,
            schema.edge_vec, schema.edge_scatter_index, schema.edge_mask)
        loss = (out["energy"] / args.atoms).square().mean() + 100. * out["extended_force"].square().mean()
        if args.parameter_targets_only:
            parameters = tuple(v for v in model.parameters() if v.requires_grad)
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            for parameter, gradient in zip(parameters, gradients):
                if gradient is not None:
                    if parameter.grad is None:
                        parameter.grad = gradient
                    else:
                        parameter.grad.add_(gradient)
        else:
            loss.backward()
        optimizer.step()
        return loss.detach()

    started = time.perf_counter()
    for _ in range(args.warmup):
        for schema in batches:
            loss = step(schema)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    if not model.compiled_core_compute_cache:
        raise RuntimeError("DPA training did not create a compiled core")
    if not bool(loss.isfinite()):
        raise RuntimeError("Nonfinite DPA warmup loss")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for index in range(args.repeats):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        loss = step(batches[index % len(batches)])
        end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000
        if not bool(loss.isfinite()):
            raise RuntimeError("Nonfinite DPA timed loss")
        samples.append({"batch": index % 3, "wall_ms": wall_ms,
                        "gpu_ms": begin.elapsed_time(end), "loss": float(loss)})
    if not all(bool(v.isfinite().all()) and (v.grad is None or bool(v.grad.isfinite().all()))
               for v in model.parameters() if v.requires_grad):
        raise RuntimeError("Nonfinite DPA parameter or gradient after full training steps")
    result = {"status": "PASS", "atoms": args.atoms, "neighbors": args.neighbors,
        "backward_targets": "all_trainable_parameters" if args.parameter_targets_only else "all_leaves",
        "gradient_execution": "autograd.grad+accumulation" if args.parameter_targets_only else "Tensor.backward",
        "post_timing_parameters_and_gradients_finite": True,
        "graph_execution_policy": graph_policy, "changing_batches": 3, "boundary": "lower",
        "graph_build_in_timing": False, "samples": samples,
        "median_ms": statistics.median(v["wall_ms"] for v in samples),
        "median_gpu_ms": statistics.median(v["gpu_ms"] for v in samples),
        "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
        "warmup_seconds": warmup_seconds, "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
        "compiled_core_slots": len(model.compiled_core_compute_cache),
        "triton_modules": [{"name": name, "configured_inference_level": module.triton_infer_level,
                            "training": module.training, "value_path_active_in_training": False}
                           for name, module in model.named_modules() if hasattr(module, "triton_infer_level")],
        "arithmetic": "strict FP32; no AMP/TF32", "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "source": str(source),
        "checkpoint": str(args.checkpoint), "model_config": str(args.config),
        "descriptor": config['model']['descriptor'],
        "timed_work": "energy+conservative force+E/F loss+mixed double backward+AdamW update"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    with forbid_cuda_graph() as graph_policy:
        main(graph_policy)
