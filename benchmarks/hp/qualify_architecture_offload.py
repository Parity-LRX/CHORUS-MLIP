"""Independent full-size original-formula oracle with exact saved-tensor offload.

This is verification only. CPU offload preserves each tensor's dtype and values;
no HP expression is used to generate the reference and no time is a benchmark.
Run within a GPU-A100/XHPC-A100 sbatch allocation, never on a login node.
"""
import argparse
import gc
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from architecture_cases import MODEL_SEED, GEOMETRY_SEED, chorus_spec
from profile_scaling import geometry, reference_values, neighbor_statistics
from profile_architecture_chorus import errors
from chorus.synthetic import build_model
from chorus.hp import configure_model, energy_forces_virial


@contextmanager
def checkpointed_original_products(model, chunk):
    """Rematerialize the literal original formulas, splitting only local atoms.

    Bridge-U product/linear/self-connection operations have no cross-atom
    dependency. Every slice retains its original U and parameters. Checkpoint
    each slice separately so backward never materializes an all-atom SC
    temporary. Interactions and phase modules keep their full original graphs.
    """
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    handles = []
    for modules in (model.interactions, model.phase_adapters, model.charged_updates):
        for module in modules.values() if isinstance(modules, torch.nn.ModuleDict) else modules:
            original = module.forward
            def recompute(*a, _original=original, **kw):
                with set_checkpoint_early_stop(False):
                    return checkpoint(_original, *a, use_reentrant=False,
                                      preserve_rng_state=False, **kw)
            module.forward = recompute
            handles.append((module, original))
    for module in model.products:
        if type(module).__name__ != 'ICTDBridgeUProductBasisBlockSO3':
            raise ValueError('Atomwise reference requires the explicit local Bridge-U product')
        original = module.forward
        def product(node_feats, sc, node_attrs, node_type_idx=None, _original=original):
            outputs = []
            for start in range(0, node_feats.shape[0], chunk):
                stop = start+chunk
                with set_checkpoint_early_stop(False):
                    outputs.append(checkpoint(_original, node_feats[start:stop],
                        None if sc is None else sc[start:stop], node_attrs[start:stop],
                        None if node_type_idx is None else node_type_idx[start:stop],
                        use_reentrant=False, preserve_rng_state=False))
            return torch.cat(outputs, dim=0)
        module.forward = product
        handles.append((module, original))
    try:
        yield
    finally:
        for module, original in handles:module.forward = original


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--scope', choices=('final', 'persistent'), required=True)
    p.add_argument('--atoms', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--product-chunk', type=int, default=0)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Qualification requires the designated sbatch allocation')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    spec = chorus_spec(args.case, args.scope)
    record = dict(status='RUNNING', stage='build', case=args.case, scope=args.scope,
        atoms=args.atoms, spec=spec, slurm_job=os.environ['SLURM_JOB_ID'],
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_snapshot=str(Path(__file__).resolve().parents[2]),
        reference=('literal_original_formula_checkpointed_local_product_chunks'
                   if args.product_chunk else 'literal_original_formula_saved_tensors_CPU_offload'),
        product_chunk=args.product_chunk, pid=os.getpid(),
        benchmark_timing=False)

    def mark(stage):
        record['stage'] = stage
        args.out.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({'stage': stage, 'case': args.case, 'scope': args.scope,
                          'atoms': args.atoms}), flush=True)

    code = 0
    try:
        torch.manual_seed(MODEL_SEED)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = build_model(channels=spec['channels'], lmax=spec['lmax'],
            num_interaction=spec['interactions'], route='baseline', product_backend='ictd-bridge-u',
            dtype=torch.float32, device=torch.device('cuda'), correlation=3,
            avg_num_neighbors=50., ictd_fix_use_reduced_cg=True,
            ictd_fix_phase_mode='final-full-l-residual', ictd_fix_phase_amplitude='softplus',
            ictd_fix_phase_placement='pre-product-full-l',
            ictd_fix_phase_density_pairs='full-nonlinear', ictd_fix_phase_density_rank=16,
            ictd_fix_phase_scope=args.scope, ictd_fix_freeze_conv_tp_weight=True).eval().requires_grad_(False)
        model.skip_input_validation, model.preserve_edge_order = True, True
        state_file = args.root/'initializations'/args.case/f'chorus_{args.scope}_{args.case}_initialization.pt'
        saved = torch.load(state_file, map_location='cpu', weights_only=False)
        if saved['spec'] != spec:
            raise ValueError('Qualification initialization differs from the measured architecture')
        model.load_state_dict(saved['state'], strict=True)
        del saved
        record['initialization_file'] = str(state_file)
        record['parameters'] = sum(v.numel() for v in model.parameters())
        record['parameter_dtypes'] = sorted({str(v.dtype) for v in model.parameters()})
        graph = geometry(args.atoms, 5., GEOMETRY_SEED, 50)
        record['neighbor_statistics'] = neighbor_statistics(graph, 50)
        configure_model(model, policy='reference')
        if args.product_chunk:
            mark('reference_atom_chunking_check_128')
            small = geometry(128, 5., GEOMETRY_SEED, 50)
            plain = reference_values(model, small, True)
            with checkpointed_original_products(model, args.product_chunk):
                blocked = reference_values(model, small, True)
            record['chunking_control_parity'] = errors(blocked, plain, 128)
            del plain, blocked, small
            gc.collect()
            torch.cuda.empty_cache()
        mark('full_original_with_local_product_chunks' if args.product_chunk
             else 'full_original_with_saved_tensor_offload')
        started = time.monotonic()
        if args.product_chunk:
            with checkpointed_original_products(model, args.product_chunk):
                reference = reference_values(model, graph, True)
        else:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                reference = reference_values(model, graph, True)
        torch.cuda.synchronize()
        record['reference_wall_seconds_not_performance'] = time.monotonic()-started
        if any(not bool(v.isfinite().all()) for v in reference):
            raise RuntimeError('Original full-size reference produced nonfinite outputs')
        gc.collect()
        torch.cuda.empty_cache()
        mark('full_hp_comparison')
        configure_model(model, policy='generated', symmetric_contraction=True,
                        checkpoint_contraction=True, topology_backend='device')
        output = energy_forces_virial(model, *graph)
        actual = (output['energy'].sum(), output['forces'], output['atomic_energy'][:, None], output['virial'])
        record['parity'] = errors(actual, reference, args.atoms)
        record.update(status='PASS', stage='done', neural_precision='FP32', TF32=False,
                      virial_reduction='FP64', measured_original_qualified=True)
    except Exception as error:
        text = str(error)
        record.update(status='OOM' if isinstance(error, torch.cuda.OutOfMemoryError)
                      or 'out of memory' in text.lower() else 'FAIL',
                      error_type=type(error).__name__, error=text)
        code = 1
    finally:
        args.out.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
