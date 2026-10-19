"""Frozen-checkpoint FP32 ablations; diagnostics only, always via sbatch."""
import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

CASES = ('baseline', 'native_phase', 'split_phase', 'native_density',
         'native_activations', 'native_channel', 'native_species',
         'native_harmonics', 'native_geometry', 'unfactored_adjoint',
         'no_large_schedule', 'original_products', 'native_first_message',
         'native_maps', 'native_first_maps', 'only_messages', 'native_maps_forcecsr',
         'native_channel_forcecsr', 'native_species_forcecsr', 'native_activations_forcecsr',
         'native_channel_activation_forcecsr', 'native_channel_species_forcecsr',
         'native_activation_species_forcecsr')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--case', choices=CASES)
    p.add_argument('--cases', nargs='+', choices=CASES)
    p.add_argument('--atoms', type=int, default=32768)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Designated sbatch allocation required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.case is None:
        manifest = []
        for case in args.cases or CASES:
            command = [sys.executable, __file__, '--root', str(args.root), '--case', case,
                       '--atoms', str(args.atoms), '--repeats', str(args.repeats),
                       '--out', str(args.out.parent/f'{case}.json')]
            log = args.out.parent/f'{case}.log'
            with log.open('w') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                print(json.dumps(dict(case=case, pid=child.pid, command=command, log=str(log))), flush=True)
                try:code = child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait();code = -9
            row = dict(case=case, exit_code=code, log=str(log))
            manifest.append(row)
            args.out.write_text(json.dumps(manifest, indent=2)+'\n')
            print(json.dumps(row), flush=True)
        return

    case_name = args.case
    args.case = args.case.removesuffix('_forcecsr')
    if os.environ.get('SOURCE'):
        sys.path.insert(0, str(Path(os.environ['SOURCE'])/'benchmarks/hp'))
    import torch
    from profile_scaling import geometry, neighbor_statistics, _NoExternalTP
    from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
    from chorus.hp import configure_model, energy_forces_virial
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = '/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth'
    model = LAMMPS_MLIAP_MFF.from_checkpoint(checkpoint, element_types=['H', 'C', 'N', 'O'],
        device='cuda', ictd_fix_product_backend='ictd-bridge-u', ictd_tp_backend='pytorch',
        reuse_checkpoint_u=True).wrapper.model.eval().requires_grad_(False)
    graph = geometry(args.atoms, model.max_radius, 20260728, 50)
    cache = args.root/'cache/verification'/f'final_n{args.atoms}_k50_fp32.pt'
    saved = torch.load(cache, map_location='cpu', weights_only=False)
    stat = Path(checkpoint).stat()
    assert saved['identity']['checkpoint_size'] == stat.st_size
    assert saved['identity']['checkpoint_mtime_ns'] == stat.st_mtime_ns
    assert all(torch.equal(g.cpu(), old) for g, old in zip(graph, saved['geometry']))
    expected = saved['expected']
    record = dict(case=case_name, atoms=args.atoms, neighbors=neighbor_statistics(graph, 50),
        pid=os.getpid(), command=sys.argv, slurm_job=os.environ['SLURM_JOB_ID'],
        source_snapshot=os.environ.get('SOURCE', str(Path(__file__).resolve().parents[2])),
        checkpoint=checkpoint, reference=str(cache), reference_status=saved['reference_status'],
        tolerance='E/F/atomic: atol=1e-4, rtol=1e-3; virial: 1e-5 norm + N*1e-7',
        arithmetic='IEEE FP32, TF32 off; virial FP64', benchmark_timing=False, repeats=[])
    del saved
    model.skip_input_validation, model.preserve_edge_order = True, True
    options = dict(policy='generated', checkpoint_contraction=True, topology_backend='device')
    if args.case == 'only_messages':
        options.update(density_fusion=False, species_fusion=False, phase_fusion=False,
                       terminal_fusion=False)
    switch = dict(native_phase='phase_fusion', native_density='density_fusion',
                  native_species='species_fusion', no_large_schedule='large_graph_fusion')
    if args.case in switch:options[switch[args.case]] = False
    record['plan'] = configure_model(model, **options)
    if args.case == 'native_first_message':
        object.__setattr__(model.interactions[0], '_hp_message', None)
    if args.case in ('native_maps', 'native_first_maps', 'native_maps_forcecsr'):
        modules = model.interactions[0].modules() if args.case == 'native_first_maps' else model.modules()
        for module in modules:
            for attribute in ('_hp_channel_linear', '_hp_activation', '_hp_species_linear'):
                if hasattr(module, attribute):object.__setattr__(module, attribute, None)
    for module in model.modules():
        for case, attribute in (('native_activations', '_hp_activation'),
                                ('native_channel', '_hp_channel_linear'),
                                ('original_products', '_hp_contraction')):
            if (args.case == case or args.case == 'only_messages') and hasattr(module, attribute):
                object.__setattr__(module, attribute, None)
        combinations = {
            'native_channel_activation': ('_hp_channel_linear', '_hp_activation'),
            'native_channel_species': ('_hp_channel_linear', '_hp_species_linear'),
            'native_activation_species': ('_hp_activation', '_hp_species_linear'),
        }
        for attribute in combinations.get(args.case, ()):
            if hasattr(module, attribute):object.__setattr__(module, attribute, None)
    if args.case in ('native_harmonics', 'only_messages'):
        object.__setattr__(model, '_hp_harmonics', None)
    if args.case == 'only_messages':
        for module in model.modules():
            if hasattr(module, '_hp_species_linear'):object.__setattr__(module, '_hp_species_linear', None)
    if args.case == 'split_phase':
        for layer in model.interactions:
            trunk = getattr(layer, '_hp_phase_trunk', None)
            if trunk is not None:trunk.auto_network = False
    if args.case == 'unfactored_adjoint':
        import mace_ictc.hp._coherent_destination as destination
        original = destination.adjoint
        def adjoint(*a, **kw):
            kw['factored'] = False
            return original(*a, **kw)
        destination.adjoint = adjoint
    execution = {'fused_geometry': False} if args.case == 'native_geometry' else {}
    if case_name.endswith('_forcecsr'):execution['force_reduction'] = 'csr'
    if args.case in ('original_products', 'only_messages'):
        # Only atom-local original products are rematerialized in small chunks.
        # This does not replace the registered whole-geometry original oracle.
        from qualify_architecture_offload import checkpointed_original_products
        context = checkpointed_original_products(model, 64)
    else:
        from contextlib import nullcontext
        context = nullcontext()
    try:
        with context:
            for repeat in range(args.repeats):
                value = energy_forces_virial(model, *graph, **execution)
                actual = (value['energy'].sum().cpu(), value['forces'].cpu(),
                          value['atomic_energy'][:, None].cpu(), value['virial'].cpu())
                row = dict(repeat=repeat, parity={})
                for index, name in enumerate(('energy', 'forces', 'atomic_energy', 'virial')):
                    delta = (actual[index]-expected[index]).abs()
                    bound = (1e-4+1e-3*expected[index].abs()) if index < 3 else None
                    row['parity'][name] = dict(max_abs=float(delta.max()),
                        relative_l2=float(delta.double().norm()/expected[index].double().norm().clamp_min(1e-20)))
                    if bound is not None:
                        row['parity'][name].update(failed_components=int((delta > bound).sum()),
                            max_bound_ratio=float((delta/bound).max()))
                row['force_probes'] = {str(atom): dict(actual=actual[1][atom].tolist(),
                    expected=expected[1][atom].tolist()) for atom in (3702, 17998, 20382, 28018)
                    if atom < args.atoms}
                row['failed_force_indices'] = torch.nonzero(
                    (actual[1]-expected[1]).abs() > 1e-4+1e-3*expected[1].abs()).tolist()
                row['allocated_bytes'] = torch.cuda.memory_allocated()
                row['reserved_bytes'] = torch.cuda.memory_reserved()
                record['repeats'].append(row)
                args.out.write_text(json.dumps(record, indent=2)+'\n')
                print(json.dumps(row), flush=True)
                del actual, value
                gc.collect()
        record['status'] = 'DONE_DIAGNOSTIC_ONLY'
    except Exception as error:
        record.update(status='OOM' if isinstance(error, torch.cuda.OutOfMemoryError) else 'ERROR',
                      error=repr(error))
        raise
    finally:
        args.out.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':main()
