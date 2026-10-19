"""Independent, bounded original-formula reference for the trained large graph."""
import argparse
import gc
import json
import os
from pathlib import Path
import torch
from profile_scaling import geometry, reference_values, assert_outputs, neighbor_statistics
from blocked_reference import blocked_reference
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--atoms', type=int, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Full large reference requires the designated sbatch allocation')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loaded = LAMMPS_MLIAP_MFF.from_checkpoint(str(args.checkpoint), element_types=['H', 'C', 'N', 'O'],
        device='cuda', ictd_fix_product_backend='ictd-bridge-u', ictd_tp_backend='pytorch',
        reuse_checkpoint_u=True)
    model = loaded.wrapper.model.eval().requires_grad_(False)
    graph = geometry(args.atoms, model.max_radius, 20260728, 50)
    neighbor_statistics(graph, 50)
    stat = args.checkpoint.stat()
    identity = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_size=stat.st_size,
        checkpoint_mtime_ns=stat.st_mtime_ns, atoms=args.atoms, neighbors=50, virial=True,
        parameters=sum(v.numel() for v in model.parameters()), reference_version=1)
    if args.out.exists():
        cached = torch.load(args.out, map_location='cpu', weights_only=False)
        if cached['identity'] != identity or any(not torch.equal(value.cpu(), saved)
                for value, saved in zip(graph, cached['geometry'])):
            raise ValueError('Existing numerical oracle identity/geometry differs')
        print(json.dumps(dict(status='REUSED', identity=identity)), flush=True)
        return
    print(json.dumps(dict(stage='halo_control_512', identity=identity)), flush=True)
    small = geometry(512, model.max_radius, 20260728, 50)
    plain = reference_values(model, small, True)
    split = blocked_reference(model, small, virial=True, centres=32)
    assert_outputs(split, plain, 512, virial=True)
    del plain, split, small
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage='complete_large_halos', atoms=args.atoms)), flush=True)
    expected = blocked_reference(model, graph, virial=True, centres=128)
    if any(not bool(value.isfinite().all()) for value in expected):
        raise RuntimeError('Full original reference is nonfinite')
    status = dict(status='FULL_HALOS_PASS', atoms=args.atoms, full_stress_geometry=True,
        halo_vs_monolithic_atoms=512, centres_per_block=128, timing_includes_reference=False,
        generator_slurm_job=os.environ['SLURM_JOB_ID'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name+f'.{os.getpid()}.tmp')
    torch.save(dict(identity=identity, geometry=tuple(value.cpu() for value in graph),
        expected=tuple(value.cpu() for value in expected), reference_status=status), temporary)
    os.replace(temporary, args.out)
    print(json.dumps(dict(status='PASS', identity=identity, reference_status=status)), flush=True)


if __name__ == '__main__':main()
