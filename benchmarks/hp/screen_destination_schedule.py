"""Screen native launch schedules on live trained-model TP/VJP operands."""
import argparse
import gc
import json
import os
import statistics
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--atoms', type=int, default=10000)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--group', choices=('launch', 'register', 'paths'), default='launch')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('sbatch allocation required')
    import torch
    from profile_scaling import geometry, _NoExternalTP
    from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
    from chorus.hp import configure_model, energy_forces_virial
    import mace_ictc.hp.coherent as coherent
    from mace_ictc.hp._coherent_destination import adjoint
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = '/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth'
    model = LAMMPS_MLIAP_MFF.from_checkpoint(checkpoint, element_types=['H', 'C', 'N', 'O'],
        device='cuda', ictd_fix_product_backend='ictd-bridge-u', ictd_tp_backend='pytorch',
        reuse_checkpoint_u=True).wrapper.model.eval().requires_grad_(False)
    model.skip_input_validation, model.preserve_edge_order = True, True
    configure_model(model, policy='generated', checkpoint_contraction=True, topology_backend='device')
    graph = geometry(args.atoms, model.max_radius, 20260728, 50)
    captured = []
    original = coherent._Coherent.backward
    def collect(ctx, *grads):
        result = original(ctx, *grads)
        if ctx.p['xdim'] == (128 if args.layer == 0 else 1152):
            inputs = tuple(value.detach() for value in ctx.saved_tensors)
            cotangents = tuple(None if value is None else value.detach() for value in grads)
            needed = tuple(value is not None for value in (result[0], result[1], result[3], result[4], result[5]))
            expected = tuple(None if value is None else value.detach() for value in
                             (result[0], result[1], result[3], result[4], result[5]))
            captured.append((inputs, ctx.p, ctx.projection, cotangents, needed, expected))
        return result
    coherent._Coherent.backward = staticmethod(collect)
    energy_forces_virial(model, *graph)
    coherent._Coherent.backward = staticmethod(original)
    assert len(captured) == 1
    inputs, params, projection, grads, needed, expected = captured.pop()
    del model, graph, captured
    gc.collect();torch.cuda.empty_cache()
    record = dict(slurm_job=os.environ['SLURM_JOB_ID'], pid=os.getpid(), command=sys.argv,
        checkpoint=checkpoint, atoms=args.atoms, layer=args.layer, group=args.group,
        arithmetic='IEEE FP32, TF32 off', source_snapshot=str(Path(__file__).resolve().parents[2]),
        model_level_qualification=False, candidates=[])
    if args.group == 'launch':
        candidates = [dict(block_warps=value) for value in (4, 1, 2, 8)]
    elif args.group == 'register':
        candidates = [dict(register_limit=value) for value in (0, 96, 128, 160, 192)]
    else:
        candidates = [dict(group_paths=value) for value in (0, 3, 4, 5, 6)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        row = dict(options=candidate)
        print(json.dumps(dict(stage='candidate', options=candidate)), flush=True)
        try:
            def run():
                return adjoint(*inputs, params, projection, needed, grads, atomic_x=True,
                    factored=True, coalesced=True, **candidate)
            actual = run()
            torch.cuda.synchronize()
            row['parity'] = []
            for got, ref in zip(actual, expected):
                assert (got is None) == (ref is None)
                if got is not None:
                    delta = (got-ref).abs()
                    row['parity'].append(dict(max_abs=float(delta.max()),
                        relative_l2=float(delta.double().norm()/ref.double().norm().clamp_min(1e-20))))
                    torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-4)
            del actual
            for _ in range(3):run()
            torch.cuda.synchronize()
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):output = run()
            times = []
            for _ in range(15):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):capture.replay()
                end.record();end.synchronize()
                times.append(start.elapsed_time(end)/5)
            row.update(status='PASS_OPERATOR_ONLY', median_ms=statistics.median(times), samples_ms=times,
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
            del output, capture
        except Exception as error:
            row.update(status='ERROR', error=repr(error))
        record['candidates'].append(row)
        args.out.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        gc.collect();torch.cuda.empty_cache()
    record['status'] = 'DONE_OPERATOR_SCREEN_ONLY'
    args.out.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':main()
