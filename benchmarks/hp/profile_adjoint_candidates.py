"""Remote-only paired screening of equivalent coherent adjoint schedules.

The qualified joint schedule supplies same-input E/F/virial references. A
winning candidate still needs the separate original-model halo qualification.
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from profile_scaling import geometry, neighbor_statistics, assert_outputs
from chorus.hp import configure_model, energy_forces_virial
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atoms", type=int, default=10000)
    parser.add_argument("--schedule", choices=("segmented", "edge_dx", "phase_context", "split", "csr1", "csr2", "csr4", "csr8"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.checkpoint, element_types=["H", "C", "N", "O"], device="cuda",
        ictd_fix_product_backend="ictd-bridge-u", ictd_tp_backend="pytorch", reuse_checkpoint_u=True)
    model = loaded.wrapper.model.eval().requires_grad_(False)
    graph = geometry(args.atoms, model.max_radius, 20260728, 50)
    neighbors = neighbor_statistics(graph, 50)
    model.skip_input_validation, model.preserve_edge_order = True, True
    plan = configure_model(model, policy="generated", symmetric_contraction=True,
                           checkpoint_contraction=True, topology_backend="device", large_graph_fusion=False)
    for layer in model.interactions:
        layer._hp_message.coherent.adjoint_splits = 1

    def run():
        value = energy_forces_virial(model, *graph)
        return tuple(value[name] for name in ("energy", "forces", "atomic_energy", "virial"))

    for _ in range(3):
        expected = run()
    expected = tuple(v.clone() for v in expected)
    import mace_ictc.hp.coherent as coherent
    from mace_ictc.hp._coherent_segmented import adjoint
    original = coherent._Coherent.backward
    current = ["joint"]

    def backward(ctx, *grads):
        if current[0] in ("joint", "phase_context", "split") or torch.is_grad_enabled() or not ctx.needs_input_grad[0]:
            return original(ctx, *grads)
        x, sh, w, g, real, imag, src, dst, mask = ctx.saved_tensors
        needed = tuple(ctx.needs_input_grad[i] for i in (0, 1, 3, 4, 5))
        if current[0].startswith("csr"):
            from mace_ictc.hp._coherent_csr_split import adjoint as split_adjoint
            dx, dh, dg, dr, di = split_adjoint(x, sh, w, g, real, imag, src, dst, mask,
                ctx.p, ctx.projection, needed, grads, splits=int(current[0][3:]))
        else:
            dx, dh, dg, dr, di = adjoint(x, sh, w, g, real, imag, src, dst, mask,
                ctx.p, ctx.projection, needed, grads, edge_dx=current[0] == "edge_dx")
        return dx, dh, None, dg, dr, di, None, None, None, None, None

    coherent._Coherent.backward = staticmethod(backward)
    from chorus.models.pure_cartesian_ictd_layers import _split_irreps as chorus_split
    split_modules = [(module, module._split_irreps) for name, module in tuple(sys.modules.items())
                     if name.startswith("chorus.") and module is not None
                     and getattr(module, "_split_irreps", None) is chorus_split]

    def sliced_irreps(x, channels, lmax):
        out, offset = {}, 0
        for l in range(lmax+1):
            width = channels*(2*l+1)
            out[l] = x[..., offset:offset+width].view(*x.shape[:-1], channels, 2*l+1)
            offset += width
        return out

    results = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for schedule in ("joint", args.schedule, "joint"):
        current[0] = schedule
        for layer in model.interactions:
            layer._hp_phase_trunk.fused_radial = schedule == "phase_context"
        if args.schedule == "split":
            for module, original_split in split_modules:
                module._split_irreps = original_split if schedule == "split" else sliced_irreps
        print(json.dumps({"stage": "warmup", "schedule": schedule, "atoms": args.atoms}), flush=True)
        for _ in range(3):
            actual = run()
        assert_outputs(actual, expected, args.atoms, virial=True)
        parity = {name: {"max_abs": float((got-ref).abs().max()),
                         "relative_l2": float((got.double()-ref.double()).norm()/ref.double().norm().clamp_min(1e-20))}
                  for name, got, ref in zip(("energy", "forces", "atomic_energy", "virial"), actual, expected)}
        del actual
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        cg = torch.cuda.CUDAGraph()
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.graph(cg):
            captured = run()
        cg.replay()
        assert_outputs(captured, expected, args.atoms, virial=True)
        samples = []
        for _ in range(15):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cg.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        row = {"median_ms": statistics.median(samples), "samples_ms": samples,
               "peak_reserved": torch.cuda.max_memory_reserved(),
               "peak_allocated": torch.cuda.max_memory_allocated(), "parity": parity}
        results.setdefault(schedule, []).append(row)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            cg.replay()
            torch.cuda.synchronize()
        args.out.with_name(args.out.stem+"_"+schedule+".profile.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
        print(json.dumps({"schedule": schedule, **row}, indent=2), flush=True)
        del captured, cg
        gc.collect()
        torch.cuda.empty_cache()
    coherent._Coherent.backward = staticmethod(original)
    for module, original_split in split_modules:
        module._split_irreps = original_split
    args.out.write_text(json.dumps({"atoms": args.atoms, "neighbors": neighbors, "plan": plan,
        "source_snapshot": str(Path(__file__).resolve().parents[2]),
        "reference": "qualified HP joint schedule, same inputs; screening only", "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
