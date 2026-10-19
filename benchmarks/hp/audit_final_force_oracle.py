"""Diagnose near-zero force sensitivity using independent original local closures.

This does not replace the registered oracle, change its tolerance, or qualify
failed performance points. No HP operator generates these local references.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import numpy as np
import torch
from profile_scaling import geometry, reference_values
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def local_force(model, graph, target):
    n = graph[0].shape[0]
    src, dst = (graph[index].cpu().numpy() for index in (3, 4))
    centres = np.zeros(n, dtype=bool)
    centres[target] = True
    for _ in model.interactions:
        influenced = centres[src].copy()
        centres[dst[influenced]] = True
    reached = centres.copy()
    for _ in model.interactions:
        selected_edges = reached[dst].copy()
        reached[src[selected_edges]] = True
    nodes, edges = np.flatnonzero(reached), np.flatnonzero(selected_edges)
    mapping = np.full(n, -1, dtype=np.int64)
    mapping[nodes] = np.arange(len(nodes))
    device = graph[0].device
    take = torch.as_tensor(nodes, device=device)
    edge_take = torch.as_tensor(edges, device=device)
    pos = graph[0].index_select(0, take).detach().requires_grad_(True)
    s = torch.as_tensor(mapping[src[edges]], device=device)
    d = torch.as_tensor(mapping[dst[edges]], device=device)
    central = torch.as_tensor(mapping[np.flatnonzero(centres)], device=device)
    output = model(pos, graph[1].index_select(0, take), torch.zeros(len(nodes), dtype=torch.long, device=device),
        s, d, graph[5].index_select(0, edge_take), graph[6])
    atom = output[0] if isinstance(output, tuple) else output
    force = -torch.autograd.grad(atom.reshape(-1).index_select(0, central).sum(), pos)[0][mapping[target]]
    return force.detach(), dict(nodes=len(nodes), edges=len(edges), energy_centres=int(centres.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU-A100/XHPC-A100 sbatch required')
    out = args.root/'results/final_oracle_audit_v1.json'
    record = dict(status='RUNNING', slurm_job=os.environ['SLURM_JOB_ID'], pid=os.getpid(),
        diagnostic_only=True, changes_registered_oracle=False, precision='FP32/TF32 off', records=[])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    code = 0
    try:
        loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
            '/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth',
            element_types=['H', 'C', 'N', 'O'], device='cuda',
            ictd_fix_product_backend='ictd-bridge-u', ictd_tp_backend='pytorch', reuse_checkpoint_u=True)
        model = loaded.wrapper.model.eval().requires_grad_(False)
        record['avg_num_neighbors'] = [layer.avg_num_neighbors for layer in model.interactions]
        if any(value is None for value in record['avg_num_neighbors']):
            raise ValueError('Local closure cannot reproduce graph-global dynamic neighbor normalization')
        if model.long_range_module is not None or model.dispersion is not None:
            raise ValueError('Short-range locality is required')
        small = geometry(512, model.max_radius, 20260728, 50)
        full = reference_values(model, small, False)[1]
        for target in (0, 256, 511):
            force, shape = local_force(model, small, target)
            torch.testing.assert_close(force, full[target], atol=1e-4, rtol=1e-3)
        del small, full
        gc.collect()
        torch.cuda.empty_cache()
        record['small_original_control'] = 'PASS'
        cached = torch.load(args.root/'cache/verification/final_n32768_k50_fp32.pt', map_location='cpu', weights_only=False)
        graph = tuple(value.cuda() for value in cached['geometry'])
        original = cached['expected'][1]
        hp = torch.load(args.root/'results/capacity_v4/final/n32768/default.qualification.pt',
            map_location='cpu', weights_only=False)['actual'][1]
        for target in (3702, 17998, 20382, 28018):
            force, shape = local_force(model, graph, target)
            force = force.cpu()
            row = dict(atom=target, shape=shape, registered_original=original[target].tolist(),
                independent_local_original=force.tolist(), hp_default=hp[target].tolist(),
                original_variation=(force-original[target]).tolist(),
                hp_minus_local_original=(hp[target]-force).tolist(),
                unchanged_bound=(1e-4+1e-3*original[target].abs()).tolist())
            record['records'].append(row)
            out.write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        record['status'] = 'DONE_DIAGNOSTIC_ONLY'
    except Exception as error:
        record.update(status='OOM' if isinstance(error, torch.cuda.OutOfMemoryError) else 'FAIL',
            error_type=type(error).__name__, error=str(error))
        code = 1
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':main()
