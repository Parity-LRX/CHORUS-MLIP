"""Remote-only same-geometry E/F profiler for the native CHORUS/HP path."""
import argparse
import importlib.util
import importlib.abc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from ase.neighborlist import neighbor_list


class _NoExternalTP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("cuequivariance", "openequivariance")):
            raise ModuleNotFoundError(f"External TP library prohibited: {fullname}", name=fullname)
        return None


sys.meta_path.insert(0, _NoExternalTP())
from chorus.hp import configure_model
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from chorus.synthetic import compute_energy_forces


def geometry(n, cutoff, seed, neighbors=32):
    script = (Path(os.environ['SOURCE']) / 'benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py'
              if os.environ.get('SOURCE') else
              Path(__file__).parents[1] / 'paper/scripts/throughput/bench_cross_model_scaling.py')
    spec = importlib.util.spec_from_file_location("cross_scaling", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    atoms = module.make_atoms(n, cutoff, neighbors, seed)
    src, dst, shift = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
    graph = (
        torch.as_tensor(atoms.positions, dtype=torch.float32, device="cuda"),
        torch.as_tensor(atoms.numbers, dtype=torch.long, device="cuda"),
        torch.zeros(n, dtype=torch.long, device="cuda"),
        torch.as_tensor(src, dtype=torch.long, device="cuda"),
        torch.as_tensor(dst, dtype=torch.long, device="cuda"),
        torch.as_tensor(shift, dtype=torch.float32, device="cuda"),
        torch.as_tensor(atoms.cell.array, dtype=torch.float32, device="cuda")[None],
    )
    return graph


def neighbor_statistics(graph, target):
    degree = torch.bincount(graph[3], minlength=graph[0].shape[0])
    result = {"target": target, "min": int(degree.min()), "max": int(degree.max()),
              "mean": float(degree.float().mean())}
    if result["min"] != target or result["max"] != target:
        raise ValueError(f"Geometry does not have the requested cutoff neighbors: {result}")
    return result


def reference_values(model, graph, virial):
    expected = tuple(v.detach() for v in compute_energy_forces(model, graph, create_graph=False))
    if virial:
        strain = torch.zeros_like(graph[6], dtype=torch.float64, requires_grad=True)
        deformation = torch.eye(3, device="cuda", dtype=torch.float64)[None]+strain
        vec = graph[0][graph[4]]-graph[0][graph[3]]+torch.einsum(
            "ei,eij->ej", graph[5], graph[6][graph[2][graph[3]]])
        deformed = torch.einsum("ei,eij->ej", vec.double(), deformation[graph[2][graph[3]]]).float()
        atom = model(*graph, precomputed_edge_vec=deformed)
        if isinstance(atom, tuple):
            atom = atom[0]
        expected += (-torch.autograd.grad(atom.sum(), strain)[0].detach(),)
    return expected


def assert_outputs(actual, expected, atoms, *, virial):
    """Intensive FP32 checks and a rotation-invariant virial tensor norm.

    A nearly cancelled shear component is not a relative-error denominator.
    Qualify ||delta V||_F <= 1e-5 ||V||_F + N*1e-7 eV. Keep every raw component
    and per-atom maximum error in reports, including the old component failures.
    """
    for index, (got, ref) in enumerate(zip(actual, expected)):
        if virial and index == 3:
            delta = (got.double()-ref.double()).norm()
            bound = 1e-5*ref.double().norm()+atoms*1e-7
            if float(delta) > float(bound):
                raise AssertionError("Virial Frobenius-norm error exceeds the FP32 bound")
        else:
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--atoms", type=int, required=True)
    p.add_argument("--neighbors", type=int, choices=(32, 50), default=32)
    p.add_argument("--allow-reference-oom", action="store_true",
                   help="Record monolithic-reference OOM and verify the full graph with exact dependency halos")
    p.add_argument("--policy", choices=("reference", "generated"), default="generated")
    p.add_argument("--sc", choices=("off", "on"), default="off")
    p.add_argument("--checkpoint-sc", action="store_true")
    p.add_argument("--stream-fusion", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherent-fusion", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--terminal-fusion", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--radial-fusion", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--radial-tile-edges", type=int, default=65536)
    p.add_argument("--radial-gate-pipeline", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--radial-jvp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--radial-destination", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--radial-cache-gates", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--topology-backend", choices=("native", "device"), default="native")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--virial", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument('--reference-cache', type=Path,
                   help='Verification-only original-model oracle; never used by the measured execution')
    p.add_argument('--execution-experiment', default='none',
                   choices=('none', 'none_forcecsr', 'persistent_distance_destination_forcecsr',
                            'persistent_distance_inline_forcecsr',
                            'persistent_distance_destination', 'persistent_distance_inline_forward',
                            'persistent_distance_cached', 'persistent_distance_cached_deep',
                            'persistent_distance_inline', 'persistent_dest', 'persistent_dest_trunk',
                            'persistent_dest_forward2', 'persistent_dest_forward4',
                            'persistent_dest_trunk_forward2', 'persistent_dest_trunk_forward4',
                            'persistent_dest_split2', 'persistent_dest_split4',
                            'persistent_dest_trunk_forward4_split2',
                            'persistent_dest_trunk32', 'persistent_dest_trunk64', 'persistent_dest_trunk128',
                            'persistent_dest_groups1', 'persistent_dest_groups2',
                            'stable', 'rankphasegeom', 'dest1rankphasegeom',
                            'dest1atomicphasegeom', 'dest1atomicphase', 'dest1atomic',
                            'dest1atomicfactorphasegeom', 'dest1atomicfactornetworkgeom',
                            'dest1atomicfactorcoalescenetworkgeom', 'dest1atomicfactorcoalescephasegeom',
                            'dest1atomicfactorcoalescenetworkpreprojectgeom',
                            'dest1atomicfactorcoalescenetworknativeharmgeom',
                            'dest1atomicfactorcoalescenetworkgeomnativeforces',
                            'dest1atomicfactorcoalescenetworkgeomfirst4fullcsr',
                            'dest1atomicfactorcoalescenetworkgeomforward4',
                            'dest1atomicfactorcoalescenetworkgeomforward4first4fullcsr'))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.checkpoint, element_types=["H", "C", "N", "O"], device="cuda",
        ictd_fix_product_backend="ictd-bridge-u", ictd_tp_backend="pytorch",
        reuse_checkpoint_u=True)
    model = loaded.wrapper.model.eval().requires_grad_(False)
    graph = geometry(args.atoms, model.max_radius, 20260728, args.neighbors)
    neighbors = neighbor_statistics(graph, args.neighbors)
    reference_status = {"status": "PASS", "atoms": args.atoms, "full_stress_geometry": True}
    # Validate once before removing the model's two host-synchronizing guards.
    print(json.dumps({"stage": "reference", "atoms": args.atoms, "neighbors": neighbors}), flush=True)
    qualification_graph = graph
    reference_oom = False
    checkpoint_stat = Path(args.checkpoint).stat()
    reference_identity = {'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_size': checkpoint_stat.st_size, 'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
        'atoms': args.atoms, 'neighbors': args.neighbors, 'virial': args.virial,
        'parameters': sum(v.numel() for v in model.parameters()), 'reference_version': 1}
    expected = None
    if args.reference_cache is not None and args.reference_cache.exists():
        cached = torch.load(args.reference_cache, map_location='cpu', weights_only=False)
        if cached['identity'] != reference_identity or any(
                not torch.equal(g.cpu(), saved) for g, saved in zip(graph, cached['geometry'])):
            raise ValueError('Original-model verification cache does not match checkpoint/protocol/live geometry')
        expected = tuple(value.to(device='cuda') for value in cached['expected'])
        reference_status = {**cached['reference_status'], 'cache_reused': True,
                            'verification_cache': str(args.reference_cache)}
        del cached
    try:
        if expected is None:expected = reference_values(model, graph, args.virial)
    except torch.cuda.OutOfMemoryError:
        if not args.allow_reference_oom:
            raise
        reference_oom = True
    if reference_oom:
        # Leave the except block first: its live traceback otherwise retains
        # the large failed forward's tensors and defeats the bounded verifier.
        torch.cuda.empty_cache()
        from blocked_reference import blocked_reference
        # Cross-check the halo assembly against the literal monolithic model
        # on a fitting case before using it for the complete large structure.
        small = geometry(512, model.max_radius, 20260728, args.neighbors)
        small_reference = reference_values(model, small, args.virial)
        small_blocked = blocked_reference(model, small, virial=args.virial, centres=32)
        assert_outputs(small_blocked, small_reference, 512, virial=args.virial)
        del small, small_reference, small_blocked
        expected = blocked_reference(model, graph, virial=args.virial)
        reference_status = {"status": "MONOLITHIC_OOM_FULL_HALOS_PASS", "atoms": args.atoms,
                            "full_stress_geometry": True, "halo_vs_monolithic_atoms": 512,
                            "centres_per_block": 128, "timing_includes_reference": False}
    if any(not bool(v.isfinite().all()) for v in expected):
        raise RuntimeError("Original reference is nonfinite on the benchmark geometry")
    if args.reference_cache is not None and not args.reference_cache.exists():
        args.reference_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.reference_cache.with_name(args.reference_cache.name+f'.{os.getpid()}.tmp')
        torch.save({'identity': reference_identity, 'geometry': tuple(v.cpu() for v in graph),
            'expected': tuple(v.cpu() for v in expected), 'reference_status': reference_status}, temporary)
        os.replace(temporary, args.reference_cache)
    print(json.dumps({"stage": "reference_complete", **reference_status}), flush=True)
    model.skip_input_validation = True
    model.preserve_edge_order = True
    info = configure_model(model, policy=args.policy, symmetric_contraction=args.sc == "on",
                           checkpoint_contraction=args.checkpoint_sc,
                           coherent_fusion=args.coherent_fusion,
                           terminal_fusion=args.terminal_fusion,
                           radial_fusion=args.radial_fusion,
                           radial_tile_edges=args.radial_tile_edges,
                           radial_gate_pipeline=args.radial_gate_pipeline,
                           radial_jvp=args.radial_jvp,
                           radial_destination=args.radial_destination,
                           radial_cache_gates=args.radial_cache_gates,
                           stream_fusion=args.stream_fusion, topology_backend=args.topology_backend)
    from experimental_execution import apply_experiment
    execution_options = apply_experiment(model, args.execution_experiment)

    def evaluate(inputs):
        if args.virial:
            from chorus.hp import energy_forces_virial
            value = energy_forces_virial(model, *inputs, **execution_options)
            return (value["energy"].sum(), value["forces"], value["atomic_energy"][:, None], value["virial"])
        return compute_energy_forces(model, inputs, create_graph=False)

    def run():
        return evaluate(graph)

    for _ in range(3):
        actual = evaluate(qualification_graph)
    parity = {name: {"max_abs": float((got-ref).abs().max()),
                     "relative_l2": float((got.double()-ref.double()).norm()/ref.double().norm().clamp_min(1e-20))}
              for name, got, ref in zip(("energy", "forces", "atomic_energy", "virial"), actual, expected)}
    if args.virial:
        parity["virial"]["max_abs_per_atom"] = parity["virial"]["max_abs"]/args.atoms
    for index, name in enumerate(('energy', 'forces', 'atomic_energy')):
        delta = (actual[index]-expected[index]).abs()
        bound = 1e-4+1e-3*expected[index].abs()
        parity[name]['failed_components'] = int((delta > bound).sum())
        parity[name]['max_bound_ratio'] = float((delta/bound).max())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        assert_outputs(actual, expected, args.atoms, virial=args.virial)
        qualification = {'status': 'PASS', 'parity': parity, 'reference_status': reference_status}
    except AssertionError as error:
        qualification = {'status': 'FAIL', 'parity': parity, 'reference_status': reference_status,
                         'error': str(error), 'execution_experiment': args.execution_experiment}
        args.out.with_suffix('.qualification.json').write_text(json.dumps(qualification, indent=2)+'\n')
        torch.save({'actual': tuple(v.cpu() for v in actual), 'expected': tuple(v.cpu() for v in expected)},
                   args.out.with_suffix('.qualification.pt'))
        raise
    args.out.with_suffix('.qualification.json').write_text(json.dumps(qualification, indent=2)+'\n')
    torch.save({'actual': tuple(v.cpu() for v in actual), 'expected': tuple(v.cpu() for v in expected)},
               args.out.with_suffix('.qualification.pt'))
    del actual, expected
    del qualification_graph
    actual = run()
    if any(not bool(v.isfinite().all()) for v in actual):
        raise RuntimeError("Nonfinite HP output on full stress geometry")
    baseline_cpu = tuple(value.detach().cpu() for value in actual) if args.capture else None
    del actual
    torch.cuda.synchronize()
    capture_status = "not_requested"
    capture_validation = None
    capture_memory = None
    if args.capture:
        try:
            eager = run
            # Prepare live-coordinate validation outputs before the graph's
            # private pool exists. A second full eager graph beside that pool
            # can exceed device capacity even when capture/replay itself fits.
            delta = (torch.arange(graph[0].numel(), device="cuda").reshape_as(graph[0]) % 7)*1e-5
            original_positions = graph[0].clone()
            graph[0].add_(delta)
            changed_cpu = tuple(value.detach().cpu() for value in eager())
            graph[0].copy_(original_positions)
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
                captured = eager()
            capture_memory = {"capture_peak_allocated": torch.cuda.max_memory_allocated(),
                              "capture_peak_reserved": torch.cuda.max_memory_reserved()}
            cg.replay()
            assert_outputs(tuple(value.detach().cpu() for value in captured), baseline_cpu,
                           args.atoms, virial=args.virial)
            # Read live positions on replay, rather than benchmarking a stale
            # cached E/F result. Keep the input shape and neighbor graph fixed.
            graph[0].add_(delta)
            cg.replay()
            captured_cpu = tuple(value.detach().cpu() for value in captured)
            assert_outputs(captured_cpu, changed_cpu, args.atoms, virial=args.virial)
            capture_validation = {"changed_positions": True,
                "force_max_abs": float((captured_cpu[1]-changed_cpu[1]).abs().max()),
                "strategy": "live_eager_oracles_prepared_before_capture_compared_on_CPU"}
            graph[0].copy_(original_positions)
            cg.replay()
            del captured_cpu, changed_cpu, baseline_cpu, delta, original_positions
            run = cg.replay
            capture_status = "captured"
        except Exception as error:
            capture_status = f"{type(error).__name__}: {error}"
            raise RuntimeError(capture_status) from error
    torch.cuda.empty_cache()
    steady_memory = {"allocated": torch.cuda.memory_allocated(),
                     "reserved_including_graph_pool": torch.cuda.memory_reserved()}
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(15):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    median = statistics.median(samples)
    result = {"checkpoint": args.checkpoint, "atoms": args.atoms,
              'execution_experiment': args.execution_experiment,
              "source_snapshot": str(Path(__file__).resolve().parents[2]),
              "edges": graph[3].numel(), "neighbors": neighbors, "cutoff": model.max_radius,
              "reference_status": reference_status, "plan": info, "capture": capture_status,
              "virial_check": "||delta V||_F <= 1e-5*||V||_F + N*1e-7 eV; raw component errors retained" if args.virial else None,
              "outputs": "energy+forces+virial" if args.virial else "energy+forces",
              "virial_reduction": "FP64" if args.virial else None,
              "capture_validation": capture_validation,
              "capture_memory": capture_memory, "steady_memory": steady_memory,
              "parity": parity,
              "median_ms": median, "atoms_per_second": args.atoms*1000/median,
              "peak_allocated": torch.cuda.max_memory_allocated(),
              "peak_reserved": torch.cuda.max_memory_reserved(), "samples_ms": samples,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF")),
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "product": model.ictd_fix_product_backend,
              "reduced_cg": model.ictd_fix_use_reduced_cg,
              "parameters": sum(v.numel() for v in model.parameters()),
              'execution_dispatch': {
                  'large_neutral_messages': sum(int(getattr(getattr(layer, '_hp_message', None), 'neutral_large', None) is not None
                                                   and graph[3].numel() >= 327680) for layer in model.interactions),
                  'phase_networks': sum(int(getattr(getattr(getattr(layer, '_hp_phase_trunk', None), 'network', None), 'signature', None)
                                             is not None) for layer in model.interactions),
                  'destination_extensions': [value.__name__ for value in getattr(
                      sys.modules.get('mace_ictc.hp._coherent_destination'), '_EXTENSIONS', {}).values()],
                  'radial_destination_extensions': [value.__name__ for value in getattr(
                      sys.modules.get('mace_ictc.hp._radial_destination'), '_EXTENSIONS', {}).values()],
              },
              "imported_external_tp": sorted(name for name in sys.modules
                    if name.startswith(("cuequivariance", "openequivariance")))}
    try:
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True).splitlines()
        result["process_resident_mib"] = [int(line.split(",")[1].strip()) for line in processes
                                           if line.split(",")[0].strip() == str(os.getpid())]
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        result["process_resident_mib_error"] = str(error)
    if args.profile:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            run()
            torch.cuda.synchronize()
        table = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=45)
        args.out.with_suffix(".profile.txt").write_text(table)
        prof.export_chrome_trace(str(args.out.with_suffix(".trace.json")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
