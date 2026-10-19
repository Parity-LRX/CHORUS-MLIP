"""Independent diagnostic of original-model force-reference summation order."""
import argparse
import json
from pathlib import Path
import torch
from profile_scaling import geometry, reference_values
from coordinate_reference import coordinate_force_reference
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def metrics(got, ref):
    delta = (got.double()-ref.double()).abs()
    bound = 1e-4+1e-3*ref.double().abs()
    return {'max_abs': float(delta.max()), 'relative_l2': float(delta.norm()/ref.double().norm()),
        'failed_components': int((delta > bound).sum()), 'max_bound_ratio': float((delta/bound).max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reference-cache', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--full-coordinate', action='store_true')
    p.add_argument('--fp64-targets', action='store_true',
                   help='Higher-precision original-model diagnostic only; candidate stays FP32')
    args = p.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LAMMPS_MLIAP_MFF.from_checkpoint(args.checkpoint, element_types=['H','C','N','O'],
        device='cuda', ictd_fix_product_backend='ictd-bridge-u', ictd_tp_backend='pytorch',
        reuse_checkpoint_u=True).wrapper.model.eval().requires_grad_(False)
    result = {'checkpoint': args.checkpoint, 'source_snapshot': str(Path(__file__).resolve().parents[2])}
    for n in (512, 1024):
        small = geometry(n, model.max_radius, 20260728, 50)
        expected = reference_values(model, small, True)[1]
        actual = coordinate_force_reference(model, small, centres=32)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
        result[f'monolithic_{n}_crosscheck'] = metrics(actual, expected)
        print(json.dumps({f'monolithic_{n}_crosscheck': result[f'monolithic_{n}_crosscheck']}), flush=True)
        del small, expected, actual
        torch.cuda.empty_cache()
    cache = torch.load(args.reference_cache, map_location='cpu', weights_only=False)
    graph = tuple(value.to('cuda') for value in cache['geometry'])
    old_force = cache['expected'][1].to('cuda')
    candidate = torch.load(args.candidate, map_location='cpu', weights_only=False)['actual'][1].to('cuda')
    failed = ((candidate-old_force).abs() > 1e-4+1e-3*old_force.abs()).any(-1)
    targets = failed.nonzero().flatten().cpu().tolist()
    # Audit known sensitive components from every rejected, fixed-geometry arm,
    # not only the current candidate. The candidate execution is never changed.
    targets = sorted(set(targets+[1119, 3549, 4639, 9599]))
    queried = coordinate_force_reference(model, graph, targets=targets, centres=1)
    index = torch.tensor(targets, device='cuda', dtype=torch.long)
    result.update({'targets': targets, 'original_energy_halo_force': old_force[index].tolist(),
        'original_coordinate_halo_force': queried.tolist(), 'candidate_force': candidate[index].tolist(),
        'candidate_vs_coordinate': metrics(candidate[index], queried),
        'original_energy_vs_coordinate': metrics(old_force[index], queried)})
    if args.full_coordinate:
        full_force = coordinate_force_reference(model, graph, centres=32)
        result['full_coordinate_force'] = {'centres_per_block': 32,
            'all_force_components': full_force.numel(), 'finite': bool(full_force.isfinite().all()),
            'candidate_vs_coordinate': metrics(candidate, full_force),
            'energy_halo_vs_coordinate': metrics(old_force, full_force)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'force': full_force.cpu(), 'identity': cache['identity'],
            'source_snapshot': str(Path(__file__).resolve().parents[2]), 'centres_per_block': 32},
            args.out.with_suffix('.coordinate_force.pt'))
        print(json.dumps({'full_coordinate_force': result['full_coordinate_force']}), flush=True)
        del full_force
        torch.cuda.empty_cache()
    if args.fp64_targets:
        # This is a separate precision audit of the literal original model,
        # never a modified checkpoint/protocol for the timed FP32 candidate.
        model.double()
        for module in model.modules():
            for name in ('edge_compute_dtype', 'internal_compute_dtype', 'compute_dtype'):
                if name in vars(module) and isinstance(vars(module)[name], torch.dtype):
                    setattr(module, name, torch.float64)
        graph64 = tuple(value.double() if value.is_floating_point() else value for value in graph)
        precision_force = coordinate_force_reference(model, graph64, targets=targets, centres=1)
        result['fp64_original_targets'] = {'force': precision_force.tolist(),
            'candidate_fp32_vs_fp64': metrics(candidate[index], precision_force),
            'energy_halo_fp32_vs_fp64': metrics(old_force[index], precision_force),
            'coordinate_halo_fp32_vs_fp64': metrics(queried, precision_force),
            'timed_candidate_precision_changed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':main()
