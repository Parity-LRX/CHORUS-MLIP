"""Remote DPA4-mini training baseline, with the same full E/F update workload."""
import argparse
import copy
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

os.environ["DP_INTERFACE_PREC"] = "low"
os.environ["DP_COMPILE_INFER"] = "1"
os.environ["DP_TRITON_INFER"] = "2"
os.environ["DP_TF32_INFER"] = "0"
os.environ["DP_CUTE_INFER"] = "0"
import torch
from deepmd.pt.model.model import get_model
from deepmd.utils.argcheck import normalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--atoms", type=int, required=True)
    p.add_argument("--neighbors", type=int, choices=(32, 50), default=50)
    p.add_argument("--boundary", choices=("full", "lower"), default="lower")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    script = Path(__file__).parents[1]/"paper/scripts/throughput/bench_cross_model_scaling.py"
    spec = importlib.util.spec_from_file_location("common_geometry", script)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    atoms = common.make_atoms(args.atoms, model.get_rcut(), args.neighbors, 20260728)
    from ase.neighborlist import neighbor_list
    src, dst = neighbor_list("ij", atoms, model.get_rcut(), self_interaction=False)
    if len(src) != args.atoms*args.neighbors:
        raise RuntimeError("Cutoff edge count mismatch")
    coord = torch.as_tensor(atoms.positions, device="cuda", dtype=torch.float32)[None]
    types = {symbol: i for i, symbol in enumerate(config["model"]["type_map"])}
    atype = torch.tensor([[types[s] for s in atoms.get_chemical_symbols()]], device="cuda")
    box = torch.as_tensor(atoms.cell.array, device="cuda", dtype=torch.float32).reshape(1, 9)
    schema = model.build_neighbor_list(coord, atype, box)
    if int(schema.edge_mask.sum()) != len(src):
        raise RuntimeError("Compact DPA edge count mismatch")
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-5, weight_decay=0.)

    def step():
        optimizer.zero_grad(set_to_none=True)
        if args.boundary == "full":
            out = model(coord, atype, box)
            force = out["force"]
        else:
            out = model.forward_lower(schema.coord, schema.atype, schema.edge_index,
                schema.edge_vec, schema.edge_scatter_index, schema.edge_mask)
            force = out["extended_force"]
        loss = (out["energy"]/args.atoms).square().mean()+100.*force.square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    started = time.perf_counter()
    for _ in range(3):
        loss = step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-started
    if not model.compiled_core_compute_cache:
        raise RuntimeError("DPA training did not create a compiled core")
    if not bool(loss.isfinite()):
        raise RuntimeError("Nonfinite DPA training loss")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(8):
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-started)*1000)
    if not bool(loss.isfinite()):
        raise RuntimeError("Nonfinite DPA final loss")
    result = {"atoms": args.atoms, "edges": len(src), "neighbors": args.neighbors,
        "boundary": args.boundary, "graph_build_in_timing": args.boundary == "full",
        "median_ms": statistics.median(samples), "samples_ms": samples,
        "atoms_per_second": args.atoms*1000/statistics.median(samples),
        "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
        "warmup_seconds": warmup_seconds, "last_loss": float(loss),
        "optimizer": {"type": "AdamW", "lr": 1e-5, "weight_decay": 0.},
        "loss": "(energy/N)^2+100*mean(force^2); fixed zero synthetic targets",
        "timed_work": "energy+conservative force+energy/force MSE+double backward+AdamW update",
        "parameters": sum(v.numel() for v in model.parameters()),
        "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
        "should_use_compile": model.should_use_compile(), "compiled_core_slots": len(model.compiled_core_compute_cache),
        "triton_modules": [{"name": name, "configured_inference_level": module.triton_infer_level,
                            "training": module.training, "value_path_active_in_training": False}
                           for name, module in model.named_modules() if hasattr(module, "triton_infer_level")],
        "triton_note": "Upstream level2 fused value path is eval-only; training uses upstream compiled differentiable path",
        "arithmetic": "strict FP32; no AMP/TF32", "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "source_snapshot": str(Path(__file__).resolve().parents[2])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
