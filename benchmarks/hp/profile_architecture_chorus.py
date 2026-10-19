"""Full E/F/virial scaling of explicit untrained CHORUS architectures.

Original-formula qualification, rematerialization/halo construction and JIT
are excluded from timings. A measured-graph oracle OOM is explicitly distinct
from a passed small-graph oracle; it is never relabelled a full-size PASS.
"""
import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
import torch

from architecture_cases import CASES, MODEL_SEED, GEOMETRY_SEED, chorus_spec
from profile_scaling import geometry, neighbor_statistics, reference_values, assert_outputs
from chorus.synthetic import build_model
from chorus.hp import configure_model, energy_forces_virial


@contextmanager
def rematerialized_original(model):
    """Checkpoint only original modules; no HP kernel or changed model term."""
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    handles = []
    for modules in (model.interactions, model.products, model.phase_adapters, model.charged_updates):
        for module in modules.values() if isinstance(modules, torch.nn.ModuleDict) else modules:
            original = module.forward
            def forward(*a, _original=original, **kw):
                # TorchScript wraps the internal early-stop exception in a
                # RuntimeError. Full recomputation avoids that control-flow
                # conflict while keeping the literal original operations.
                with set_checkpoint_early_stop(False):
                    return checkpoint(_original, *a, use_reentrant=False, preserve_rng_state=False, **kw)
            module.forward = forward
            handles.append((module, original))
    try:
        yield
    finally:
        for module, original in handles:module.forward = original


def errors(actual, expected, n):
    result = {}
    for idx, (a, b) in enumerate(zip(actual, expected)):
        field = ('energy', 'forces', 'atomic_energy', 'virial')[idx]
        delta = (a.double()-b.double()).abs()
        result[field] = {'max_abs': float(delta.max()),
            'relative_l2': float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-20))}
        if idx < 3:
            bound = 1e-4+1e-3*b.abs()
            result[field].update(failed_components=int(((a-b).abs() > bound).sum()),
                                max_bound_ratio=float(((a-b).abs()/bound).max()))
        else:result[field]['max_abs_per_atom'] = float(delta.max())/n
    assert_outputs(actual, expected, n, virial=True)
    return result


def run(args, progress):
    def mark(stage):
        progress['stage'] = stage
        args.out.write_text(json.dumps(progress, indent=2)+'\n')
        print(json.dumps({'stage': stage, 'case': args.case, 'scope': args.scope,
                          'atoms': args.atoms}), flush=True)
    torch.manual_seed(MODEL_SEED)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = chorus_spec(args.case, args.scope)
    progress.update(spec=spec, source_snapshot=str(Path(__file__).resolve().parents[2]),
        model_seed=MODEL_SEED, geometry_seed=GEOMETRY_SEED, atoms=args.atoms, neighbors=50,
        outputs='energy+forces+virial', neural_precision='FP32', TF32=False, virial_reduction='FP64',
        gpu=torch.cuda.get_device_name(), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_job=os.environ.get('SLURM_JOB_ID'))
    mark('model_construction')
    model = build_model(channels=spec['channels'], lmax=spec['lmax'], num_interaction=spec['interactions'],
        route='baseline', product_backend='ictd-bridge-u', dtype=torch.float32, device=torch.device('cuda'),
        correlation=3, avg_num_neighbors=50., ictd_fix_use_reduced_cg=True,
        ictd_fix_phase_mode='final-full-l-residual', ictd_fix_phase_amplitude='softplus',
        ictd_fix_phase_placement='pre-product-full-l', ictd_fix_phase_density_pairs='full-nonlinear',
        ictd_fix_phase_density_rank=16, ictd_fix_phase_scope=args.scope,
        ictd_fix_freeze_conv_tp_weight=True).eval().requires_grad_(False)
    model.skip_input_validation, model.preserve_edge_order = True, True
    progress['parameters'] = sum(v.numel() for v in model.parameters())
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_dir/f'chorus_{args.scope}_{args.case}_initialization.pt'
    if state_file.exists():
        saved = torch.load(state_file, map_location='cpu', weights_only=False)
        if saved['spec'] != spec:raise ValueError('Existing architecture initialization has a different specification')
        model.load_state_dict(saved['state'], strict=True)
    else:
        torch.save({'spec': spec, 'state': {k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in model.state_dict().items()}}, state_file)
    progress['initialization_file'] = str(state_file)
    graph = geometry(args.atoms, 5., GEOMETRY_SEED, 50)
    progress['neighbor_statistics'] = neighbor_statistics(graph, 50)
    # The independent small original expression is always checked, even when
    # a large point cannot fit. The numerical reference is not an HP output.
    small = graph if args.atoms == 128 else geometry(128, 5., GEOMETRY_SEED, 50)
    mark('original_small_reference')
    expected_small = reference_values(model, small, True)
    if any(not bool(v.isfinite().all()) for v in expected_small):
        raise RuntimeError('Original formula produced nonfinite small-reference outputs')
    progress['reference_small_atoms'] = 128
    options = dict(policy='generated', symmetric_contraction=True, checkpoint_contraction=True,
                   topology_backend='device', radial_jvp=args.radial_jvp,
                   radial_fusion=args.radial_fusion, radial_gate_pipeline=args.radial_gate_pipeline,
                   radial_tile_edges=args.radial_tile_edges)
    info = configure_model(model, **options)
    progress['plan'] = info
    def evaluate(inputs):
        out = energy_forces_virial(model, *inputs)
        return out['energy'].sum(), out['forces'], out['atomic_energy'][:, None], out['virial']
    actual_small = evaluate(small)
    progress['small_parity'] = errors(actual_small, expected_small, 128)
    del actual_small, expected_small, small
    gc.collect()
    torch.cuda.empty_cache()
    mark('full_shape_hp_warmup')
    actual = evaluate(graph)
    if any(not bool(v.isfinite().all()) for v in actual):
        raise RuntimeError('HP model produced nonfinite full-shape outputs')
    progress['force_statistics'] = {'max_abs': float(actual[1].abs().max()),
                                    'norm': float(actual[1].double().norm())}
    actual = tuple(v.detach() for v in actual)
    # Use the complete original graph when it fits with exact checkpointing.
    # Larger graphs fall back to the same dependency-complete halo verifier as
    # the trained benchmark, not a cropped-force or changed-checkpoint oracle.
    configure_model(model, policy='reference')
    reference = None
    mark('measured_shape_original_reference')
    oracle_oom = False
    try:
        try:
            with rematerialized_original(model):reference = reference_values(model, graph, True)
        except RuntimeError as error:
            if isinstance(error, torch.cuda.OutOfMemoryError) or 'TorchScript interpreter' not in str(error):
                raise
            progress['reference_rematerialization_error'] = str(error)
            reference = reference_values(model, graph, True)
    except torch.cuda.OutOfMemoryError:
        oracle_oom = True
    if oracle_oom:
        gc.collect()
        torch.cuda.empty_cache()
        if spec['interactions'] <= 2:
            from blocked_reference import blocked_reference
            try:
                reference = blocked_reference(model, graph, virial=True, centres=16)
            except torch.cuda.OutOfMemoryError:
                oracle_oom = True
            if reference is not None:progress['reference_execution'] = 'complete_energy_halos_centres16'
        else:
            # A four-hop halo can span most of the whole graph. Keep the
            # large-size qualification limitation explicit instead of
            # treating the 128-node oracle as if it covered all large forces.
            progress['reference_execution'] = 'measured_original_OOM_small128_reference_only'
    elif reference is not None:
        progress['reference_execution'] = 'full_original_with_exact_module_checkpointing'
    if reference is not None:
        progress['parity'] = errors(actual, reference, args.atoms)
        progress['measured_original_qualified'] = True
    else:
        progress['measured_original_qualified'] = False
        progress.setdefault('reference_execution', 'measured_original_and_halo_OOM_small128_reference_only')
    del actual, reference
    gc.collect()
    torch.cuda.empty_cache()
    info = configure_model(model, **options)
    progress['plan'] = info
    mark('capture_and_live_coordinate_verification')
    def eager():return evaluate(graph)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):eager()
    torch.cuda.current_stream().wait_stream(stream)
    baseline_cpu = tuple(value.detach().cpu() for value in eager())
    original = graph[0].clone()
    delta = (torch.arange(graph[0].numel(), device='cuda').reshape_as(graph[0]) % 7)*1e-5
    graph[0].add_(delta)
    changed_cpu = tuple(value.detach().cpu() for value in eager())
    graph[0].copy_(original)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):captured = eager()
    capture_allocated, capture_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    cg.replay()
    errors(tuple(value.detach().cpu() for value in captured), baseline_cpu, args.atoms)
    graph[0].add_(delta)
    cg.replay()
    progress['capture_validation'] = errors(tuple(value.detach().cpu() for value in captured), changed_cpu, args.atoms)
    progress['capture_validation_strategy'] = 'live_eager_oracles_prepared_before_capture_compared_on_CPU'
    graph[0].copy_(original)
    del original, delta, baseline_cpu, changed_cpu
    cg.replay()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    samples = []
    for _ in range(15):
        torch.cuda.synchronize()
        start = time.perf_counter()
        cg.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    progress.update(status='PASS' if progress['measured_original_qualified'] else 'TIMED_WITH_SMALL_ORACLE_ONLY',
        stage='done', median_ms=statistics.median(samples), samples_ms=samples,
        capture_peak_allocated=capture_allocated,
        peak_reserved=max(capture_reserved, torch.cuda.memory_reserved()),
        imported_external_tp=sorted(k for k in sys.modules if k.startswith(('cuequivariance', 'openequivariance'))))
    if progress['imported_external_tp']:raise RuntimeError('Prohibited external TP library imported')
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as profiler:
            cg.replay()
            torch.cuda.synchronize()
        args.out.with_suffix('.profile.txt').write_text(profiler.key_averages(group_by_input_shape=True).table(
            sort_by='self_cuda_time_total', row_limit=60))
    try:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], text=True).splitlines()
        progress['process_resident_mib'] = [int(line.split(',')[1].strip()) for line in processes
            if line.split(',')[0].strip() == str(os.getpid())]
    except (OSError, ValueError, subprocess.CalledProcessError):pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--scope', choices=('final', 'persistent'), required=True)
    p.add_argument('--atoms', type=int, required=True)
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--radial-jvp', action='store_true')
    p.add_argument('--radial-fusion', action='store_true')
    p.add_argument('--radial-gate-pipeline', action='store_true')
    p.add_argument('--radial-tile-edges', type=int, default=65536)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    progress = {'status': 'RUNNING', 'stage': 'build'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    try:
        run(args, progress)
    except Exception as error:
        progress.update(status='OOM' if isinstance(error, torch.cuda.OutOfMemoryError) else 'FAIL',
                        error_type=type(error).__name__, error=str(error))
        if torch.cuda.is_initialized():
            progress['failure_memory'] = dict(peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())
        exit_code = 0 if progress['status'] == 'OOM' else 1
    finally:
        args.out.write_text(json.dumps(progress, indent=2)+'\n')
        print(json.dumps(progress, indent=2), flush=True)
    raise SystemExit(exit_code)


if __name__ == '__main__':main()
