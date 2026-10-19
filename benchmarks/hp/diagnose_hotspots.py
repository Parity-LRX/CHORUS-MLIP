"""Remote-only module/shape and adjoint-liveness diagnosis (not a benchmark).

Use the production E/F/virial boundary. No labels or model terms are removed.
The eager profile retains forward scopes; the graph replay measures the same
kernel sequence without Python launch gaps. All files belong to the task root.
"""
import argparse
import json
from pathlib import Path

import torch

from profile_scaling import geometry, neighbor_statistics
from chorus.hp import configure_model, energy_forces_virial
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atoms", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.checkpoint, element_types=["H", "C", "N", "O"], device="cuda",
        ictd_fix_product_backend="ictd-bridge-u", ictd_tp_backend="pytorch",
        reuse_checkpoint_u=True)
    model = loaded.wrapper.model.eval().requires_grad_(False)
    graph = geometry(args.atoms, model.max_radius, 20260728, 50)
    neighbors = neighbor_statistics(graph, 50)
    model.skip_input_validation = True
    model.preserve_edge_order = True
    plan = configure_model(model, policy="generated", symmetric_contraction=True,
                           checkpoint_contraction=True, topology_backend="device")

    def run():
        return energy_forces_virial(model, *graph)

    for _ in range(3):
        outputs = run()
    if any(not bool(value.isfinite().all()) for value in outputs.values()):
        raise RuntimeError("nonfinite diagnostic output")
    del outputs
    torch.cuda.synchronize()

    # Query live autograd targets, not only construction-time requires_grad.
    import mace_ictc.hp.coherent as coherent
    from mace_ictc.models.adjoint_requirements import required_gradients
    original = coherent._Coherent.backward
    adjoints = []

    def backward(ctx, *grads):
        x, sh, _, g, real, imag, *_ = ctx.saved_tensors
        inputs = (x, sh, g, real, imag)
        declared = tuple(ctx.needs_input_grad[i] for i in (0, 1, 3, 4, 5))
        adjoints.append({"input_shapes": [list(v.shape) for v in inputs],
                         "declared": declared,
                         "required": required_gradients(inputs, declared),
                         "missing_cotangents": [v is None for v in grads],
                         "path_specs": ctx.p["path_specs"]})
        return original(ctx, *grads)

    coherent._Coherent.backward = staticmethod(backward)
    run()
    coherent._Coherent.backward = staticmethod(original)
    torch.cuda.synchronize()

    scopes, handles = {}, []
    for name, module in model.named_modules():
        if not name or isinstance(module, torch.jit.ScriptModule):
            continue
        def before(mod, inputs, name=name):
            scope = torch.profiler.record_function("module::" + name)
            scope.__enter__()
            scopes.setdefault(id(mod), []).append(scope)
        def after(mod, inputs, output):
            scopes[id(mod)].pop().__exit__(None, None, None)
        handles.extend((module.register_forward_pre_hook(before),
                        module.register_forward_hook(after)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True, with_stack=False) as prof:
        run()
        torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    averages = prof.key_averages(group_by_input_shape=True)
    args.out.with_suffix(".profile.txt").write_text(
        averages.table(sort_by="self_cuda_time_total", row_limit=200))
    prof.export_chrome_trace(str(args.out.with_suffix(".trace.json")))
    rows = [{"name": event.key, "shapes": event.input_shapes, "calls": event.count,
             "self_cuda_us": event.self_device_time_total,
             "total_cuda_us": event.device_time_total}
            for event in averages if event.device_time_total]
    rows.sort(key=lambda row: row["self_cuda_us"], reverse=True)
    metadata = {"atoms": args.atoms, "neighbors": neighbors, "plan": plan,
                "source_snapshot": str(Path(__file__).resolve().parents[2]),
                "adjoints": adjoints, "operators": rows,
                "interaction_dimensions": [
                    {"input_lmax": layer.input_lmax, "target_lmax": layer.target_lmax,
                     "radial_linears": [[m.in_features, m.out_features]
                                        for m in layer.fc if isinstance(m, torch.nn.Linear)],
                     "phase_linears": [[m.in_features, m.out_features]
                                       for m in (layer.phase_trunk or ()) if isinstance(m, torch.nn.Linear)]}
                    for layer in model.interactions]}
    args.out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"atoms": args.atoms, "adjoints": adjoints,
                      "interaction_dimensions": metadata["interaction_dimensions"],
                      "top_operators": rows[:35]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
