"""Remote-only DPA4-mini boundary audit on the common periodic geometry."""
import argparse
import copy
import importlib.util
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

# Select DeepMD's supported FP32 interface; upstream geometry construction and
# energy/virial reductions may still retain FP64 (record the actual outputs).
os.environ["DP_INTERFACE_PREC"] = "low"
os.environ.setdefault("DP_COMPILE_INFER", "1")
os.environ.setdefault("DP_TRITON_INFER", "2")
if os.environ["DP_TRITON_INFER"] not in ("1", "2"):
    raise ValueError("This FP32 benchmark permits DPA Triton levels 1/2 only; formal baseline is level 2")
os.environ["DP_TF32_INFER"] = "0"
os.environ["DP_CUTE_INFER"] = "0"
import torch
from deepmd.pt.model.model import get_model
from deepmd.utils.argcheck import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--neighbors", type=int, choices=(32, 50), default=32)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--boundary", choices=("full", "lower"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--capture", action="store_true",
                        help="Also qualify/replay the compiled lower path with a CUDA graph")
    parser.add_argument("--reference-atoms", type=int,
                        help="Explicit original-formula audit size; smaller than --atoms is recorded as partial coverage")
    args = parser.parse_args()
    reference_atoms = args.atoms if args.reference_atoms is None else args.reference_atoms
    if reference_atoms < 1 or reference_atoms > args.atoms:
        raise ValueError('Original reference size must be positive and no larger than the measured geometry')
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = normalize(json.loads(args.config.read_text()))
    config["model"]["descriptor"]["use_amp"] = False
    config["model"].update(use_compile=True, enable_tf32=False)
    model = get_model(copy.deepcopy(config["model"]))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({k.removeprefix("model.Default."): v for k, v in checkpoint.items()
                           if k.startswith("model.Default.")}, strict=True)
    model = model.cuda().eval().requires_grad_(False)
    if not model.should_use_compile():
        raise RuntimeError("DPA inference compile is not enabled")
    path = Path(__file__).parents[1]/"paper/scripts/throughput/bench_cross_model_scaling.py"
    spec = importlib.util.spec_from_file_location("common_geometry", path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    atoms = common.make_atoms(args.atoms, model.get_rcut(), args.neighbors, 20260728)
    from ase.neighborlist import neighbor_list
    import numpy as np
    source, destination = neighbor_list("ij", atoms, model.get_rcut(), self_interaction=False)
    degrees = np.bincount(source, minlength=args.atoms)
    neighbors = {"target": args.neighbors, "min": int(degrees.min()),
                 "max": int(degrees.max()), "mean": float(degrees.mean())}
    if neighbors["min"] != args.neighbors or neighbors["max"] != args.neighbors:
        raise ValueError(f"Unexpected cutoff graph: {neighbors}")
    coord = torch.as_tensor(atoms.positions, device="cuda", dtype=torch.float32)[None]
    types = {symbol: i for i, symbol in enumerate(config["model"]["type_map"])}
    atype = torch.tensor([[types[s] for s in atoms.get_chemical_symbols()]], device="cuda")
    box = torch.as_tensor(atoms.cell.array, device="cuda", dtype=torch.float32).reshape(1, 9)
    schema = model.build_neighbor_list(coord, atype, box)
    if int(schema.edge_mask.sum()) != len(source):
        raise ValueError("DPA compact graph edge count differs from ASE cutoff graph")

    def full():
        return model(coord, atype, box)

    def lower():
        out = model.forward_lower(schema.coord, schema.atype, schema.edge_index,
                                  schema.edge_vec, schema.edge_scatter_index, schema.edge_mask)
        return {"energy": out["energy"], "force": out["extended_force"], "virial": out["virial"]}

    print(json.dumps(dict(stage='full_size_accelerated_apis', atoms=args.atoms)), flush=True)
    full_value, lower_value = full(), lower()
    parity = {}
    for name in ("energy", "force", "virial"):
        torch.testing.assert_close(lower_value[name], full_value[name], atol=2e-4, rtol=1e-3)
        parity[name] = float((lower_value[name]-full_value[name]).abs().max())
    output_dtypes = {k: str(v.dtype) for k, v in full_value.items() if isinstance(v, torch.Tensor)}
    if reference_atoms == args.atoms:
        reference_coord, reference_atype, reference_box = coord, atype, box
        accelerated_reference = full_value
    else:
        reference_geometry = common.make_atoms(reference_atoms, model.get_rcut(), args.neighbors, 20260728)
        reference_coord = torch.as_tensor(reference_geometry.positions, device='cuda', dtype=torch.float32)[None]
        reference_atype = torch.tensor([[types[s] for s in reference_geometry.get_chemical_symbols()]], device='cuda')
        reference_box = torch.as_tensor(reference_geometry.cell.array, device='cuda', dtype=torch.float32).reshape(1, 9)
        accelerated_reference = model(reference_coord, reference_atype, reference_box)
    # Qualify the accelerated model against the same original checkpoint with
    # both opt-in inference paths disabled, not only against another fast API.
    active_env = {key: os.environ[key] for key in ("DP_COMPILE_INFER", "DP_TRITON_INFER")}
    print(json.dumps(dict(stage='unaccelerated_original_audit', atoms=reference_atoms,
                          full_stress_geometry=reference_atoms == args.atoms)), flush=True)
    os.environ.update(DP_COMPILE_INFER="0", DP_TRITON_INFER="0")
    try:
        reference = get_model(copy.deepcopy(config["model"]))
        reference.load_state_dict({k.removeprefix("model.Default."): v for k, v in checkpoint.items()
                                   if k.startswith("model.Default.")}, strict=True)
        reference = reference.cuda().eval().requires_grad_(False)
        ref_value = reference(reference_coord, reference_atype, reference_box)
    finally:
        os.environ.update(active_env)
    fast_parity = {}
    for name in ("energy", "force", "virial"):
        torch.testing.assert_close(accelerated_reference[name], ref_value[name], atol=2e-4, rtol=1e-3)
        fast_parity[name] = {"max_abs": float((accelerated_reference[name]-ref_value[name]).abs().max()),
            "relative_l2": float((accelerated_reference[name].double()-ref_value[name].double()).norm()/
                                  ref_value[name].double().norm().clamp_min(1e-20))}
    del reference, ref_value, accelerated_reference, reference_coord, reference_atype, reference_box
    del full_value, lower_value
    run = full if args.boundary == "full" else lower
    for _ in range(5):
        value = run()
    if any(not bool(v.isfinite().all()) for v in value.values()):
        raise RuntimeError("Nonfinite DPA output")
    del value
    if not model.compiled_core_compute_cache:
        raise RuntimeError("No compiled DPA core was actually created")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(15):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    result = {"atoms": args.atoms, "boundary": args.boundary,
        "neighbors": neighbors, "cutoff": model.get_rcut(),
        "source_snapshot": str(Path(__file__).resolve().parents[2]),
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF")),
        "graph_build_in_timing": args.boundary == "full", "outputs": "energy+force+virial",
        "median_ms": statistics.median(samples), "samples_ms": samples,
        "atoms_per_second": args.atoms*1000/statistics.median(samples),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "parameters": sum(p.numel() for p in model.parameters()), "output_dtypes": output_dtypes,
        "lower_full_max_abs": parity, "gpu": torch.cuda.get_device_name(),
        "accelerated_reference_parity": fast_parity,
        "accelerated_reference_atoms": reference_atoms,
        "accelerated_reference_full_stress_geometry": reference_atoms == args.atoms,
        "DP_COMPILE_INFER": os.environ["DP_COMPILE_INFER"],
        "DP_TRITON_INFER": os.environ["DP_TRITON_INFER"],
        "should_use_compile": model.should_use_compile(),
        "compiled_core_slots": len(model.compiled_core_compute_cache),
        "triton_modules": [{"name": name, "level": module.triton_infer_level,
                            "fused_value_path": type(getattr(module, "_triton_value_path", None)).__name__}
                           for name, module in model.named_modules() if hasattr(module, "triton_infer_level")],
        "torch": torch.__version__, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "edge_capacity": schema.edge_mask.numel(),
        "neighbors_within_cutoff": int(schema.edge_mask.sum()),
        "model_fp32_tf32_off": True,
        "output_reduction": "upstream FP64 energy/virial accumulation"}
    result["DP_INTERFACE_PREC"] = os.environ["DP_INTERFACE_PREC"]
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True).splitlines()
    result["process_resident_mib"] = [int(line.split(",")[1].strip()) for line in processes
                                      if line.split(",")[0].strip() == str(os.getpid())]
    if args.capture:
        if args.boundary != "lower":
            raise ValueError("CUDA graph audit requires prebuilt compact edges")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                lower()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            outputs = lower()
        capture_peak = torch.cuda.max_memory_allocated()
        delta = torch.arange(schema.edge_vec.numel(), device="cuda").reshape_as(schema.edge_vec) % 7*1e-5
        with torch.no_grad():
            schema.edge_vec.add_(delta)
        cg.replay()
        changed = lower()
        for name in outputs:
            torch.testing.assert_close(outputs[name], changed[name], atol=2e-4, rtol=1e-3)
        with torch.no_grad():
            schema.edge_vec.sub_(delta)
        del changed, delta
        cg.replay()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        capture_samples = []
        for _ in range(15):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cg.replay()
            torch.cuda.synchronize()
            capture_samples.append((time.perf_counter()-start)*1000)
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True).splitlines()
        result["cuda_graph"] = {"status": "PASS", "changed_edge_vectors": True,
            "median_ms": statistics.median(capture_samples), "samples_ms": capture_samples,
            "capture_peak_allocated": capture_peak,
            "reserved_including_graph_pool": torch.cuda.memory_reserved(),
            "process_resident_mib": [int(line.split(",")[1].strip()) for line in processes
                                     if line.split(",")[0].strip() == str(os.getpid())]}
    if args.profile:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            run()
            torch.cuda.synchronize()
        args.out.with_suffix(".profile.txt").write_text(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=60))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
