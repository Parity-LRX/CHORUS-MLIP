"""Remote full-E/F/virial screening, default -> candidate -> default.

Same-input outputs of the qualified current HP path are the screening oracle.
Formal acceptance uses the independent original-model/halo verifier and DPA
on the same GPU, not these candidate-only timings.
"""
import argparse
import gc
import json
import re
import statistics
import time
from pathlib import Path

import torch

from profile_scaling import assert_outputs, geometry, neighbor_statistics
from chorus.hp import configure_model, energy_forces_virial
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atoms", type=int, default=10000)
    parser.add_argument("--candidate", choices=("fw4", "fw8", "streams16", "streams8", "owned2", "owned4", "owned8", "dest1", "dest2", "dest4", "phase", "dest1phase", "rank", "dest1rank", "dest1rankphase", "geom", "dest1rankphasegeom", "dest1atomicphasegeom", "readonly", "readonlyphasegeom", "readonlyfactorphasegeom", "dest1atomicfactorphasegeom", "neighbor2phasegeom", "neighbor4phasegeom", "network", "dest1atomicfactornetworkgeom", "dest1atomicfactorcoalescenetworkgeom", "dest1atomicfactorcoalescephasegeom", "dest1atomicfactorcoalescenetworkpreprojectgeom", "dest1atomicfactorcoalescenetworknativeharmgeom"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.checkpoint, element_types=["H", "C", "N", "O"], device="cuda",
        ictd_fix_product_backend="ictd-bridge-u", ictd_tp_backend="pytorch",
        reuse_checkpoint_u=True).wrapper.model.eval().requires_grad_(False)
    graph = geometry(args.atoms, model.max_radius, 20260728, 50)
    neighbors = neighbor_statistics(graph, 50)
    model.skip_input_validation, model.preserve_edge_order = True, True
    plan = configure_model(model, policy="generated", symmetric_contraction=True,
                           checkpoint_contraction=True, topology_backend="device", large_graph_fusion=False)
    # The prior atomic CSR placement produced rare near-zero force outliers.
    # Hold these independently screened stable reductions fixed in every arm.
    import mace_ictc.models.message_topology as topology_module
    from chorus.hp.density import DensityPlan
    topology_module.topology = topology_module.stable_topology
    DensityPlan.vjp = DensityPlan.gather_vjp
    import mace_ictc.hp._coherent_csr_split as split_module
    from mace_ictc.hp._coherent_csr_owned import adjoint as owned_adjoint
    from mace_ictc.hp._coherent_destination import adjoint as destination_adjoint
    original_adjoint = split_module.adjoint

    boundary = [False]
    def run():
        value = energy_forces_virial(model, *graph, fused_geometry=boundary[0])
        return tuple(value[name] for name in ("energy", "forces", "atomic_energy", "virial"))

    for _ in range(3):
        expected = run()
    expected = tuple(v.clone() for v in expected)
    import mace_ictc.hp.coherent as coherent
    from mace_ictc.hp._coherent_forward_tuned import forward as tuned_forward
    original = coherent._Coherent.forward
    current = ["default"]

    def forward(ctx, x, sh, w, g, real, imag, src, dst, mask, p, projection):
        if current[0] in ("default", "phase", "rank", "geom", "network") or current[0].startswith(("owned", "dest")):
            return original(ctx, x, sh, w, g, real, imag, src, dst, mask, p, projection)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, sh, w, g, real, imag, src, dst, mask)
        ctx.p, ctx.projection = p, projection
        out = tuple((x.new_empty if src.numel() else x.new_zeros)(
            x.shape[0], 3, 2*l+1, width//(2*l+1)).transpose(-1, -2)
            for l, _, width in coherent._outputs(p))
        separate = current[0].startswith("streams")
        neighbor_warps = int(re.search(r'neighbor([0-9]+)', current[0]).group(1)) if current[0].startswith('neighbor') else 1
        readonly = current[0].startswith(('readonly', 'neighbor'))
        width = 16 if readonly else int(current[0][7:] if separate else current[0][2:])
        tuned_forward(x, sh, w, g, real, imag, src, dst, mask, p, projection, out,
                      accumulator_limit=width, separate_streams=separate, readonly=readonly, neighbor_warps=neighbor_warps)
        return out

    coherent._Coherent.forward = staticmethod(forward)
    hp_harmonics = model._hp_harmonics
    results = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for policy in ("default", args.candidate, "default"):
        current[0] = policy
        boundary[0] = 'geom' in policy
        for layer in model.interactions:
            if getattr(layer, '_hp_phase_trunk', None) is not None:
                layer._hp_phase_trunk.fused_tail = 'phase' in policy
                layer._hp_phase_trunk.fused_network = 'network' in policy
                layer._hp_phase_trunk.network_preprojected = 'preproject' in policy
        object.__setattr__(model, '_hp_harmonics', None if 'nativeharm' in policy else hp_harmonics)
        terminal = model.interactions[-1]
        if 'rank' in policy:
            from chorus.hp.terminal import TerminalRankMessage
            if model.ictd_fix_phase_scope != 'final':
                raise ValueError('This candidate is the terminal Final branch, not Persistent')
            adapter = model.phase_adapters[str(len(model.interactions)-1)]
            rank_message = TerminalRankMessage(terminal, adapter)
            if not rank_message.eligible():
                raise ValueError('Original Final checkpoint is ineligible for exact terminal rank folding')
            object.__setattr__(terminal, '_hp_terminal_rank_message', rank_message)
        elif '_hp_terminal_rank_message' in vars(terminal):
            object.__delattr__(terminal, '_hp_terminal_rank_message')
        if policy.startswith(("owned", "dest")):
            splits = int(re.match(r'(?:owned|dest)([0-9]+)', policy).group(1))
            implementation = owned_adjoint if policy.startswith('owned') else destination_adjoint
            def adjoint(*a, **kw):
                kw['splits'] = splits
                if implementation is destination_adjoint:kw['atomic_x'] = 'atomic' in policy
                if implementation is destination_adjoint:kw['factored'] = 'factor' in policy
                if implementation is destination_adjoint:kw['coalesced'] = 'coalesce' in policy
                return implementation(*a, **kw)
            split_module.adjoint = adjoint
        elif policy.startswith(('readonly', 'neighbor')):
            def adjoint(*a, **kw):
                kw['readonly'] = True
                kw['factored'] = 'factor' in policy
                return original_adjoint(*a, **kw)
            split_module.adjoint = adjoint
        else:
            split_module.adjoint = original_adjoint
        print(json.dumps({"stage": "warmup", "policy": policy, "atoms": args.atoms}), flush=True)
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
        torch.cuda.reset_peak_memory_stats()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            captured = run()
        cg.replay()
        assert_outputs(captured, expected, args.atoms, virial=True)
        original_positions = graph[0].clone()
        delta = (torch.arange(graph[0].numel(), device='cuda').reshape_as(graph[0]) % 7)*1e-5
        graph[0].add_(delta)
        cg.replay()
        changed_reference = run()
        assert_outputs(captured, changed_reference, args.atoms, virial=True)
        capture_error = float((captured[1]-changed_reference[1]).abs().max())
        graph[0].copy_(original_positions)
        del changed_reference, original_positions, delta
        samples = []
        for _ in range(15):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cg.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        result = {"median_ms": statistics.median(samples), "samples_ms": samples,
                  "peak_reserved": torch.cuda.max_memory_reserved(),
                  "peak_allocated": torch.cuda.max_memory_allocated(), "parity": parity,
                  "changed_positions_force_max_abs": capture_error}
        results.setdefault(policy, []).append(result)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            cg.replay()
            torch.cuda.synchronize()
        args.out.with_name(args.out.stem+"_"+policy+".profile.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
        print(json.dumps({"policy": policy, **result}, indent=2), flush=True)
        del cg, captured
        gc.collect()
        torch.cuda.empty_cache()
    coherent._Coherent.forward = staticmethod(original)
    split_module.adjoint = original_adjoint
    args.out.write_text(json.dumps({"atoms": args.atoms, "neighbors": neighbors,
        "checkpoint": args.checkpoint, "plan": plan, "results": results,
        "source_snapshot": str(Path(__file__).resolve().parents[2]),
        "reference": "stable CSR/input-owned density HP path; screening only"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
