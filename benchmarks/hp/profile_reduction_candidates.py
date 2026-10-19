"""Remote diagnostic of live CSR and density reduction reproducibility.

Timing rows with a failed check are explicitly not qualified. Formal speed
acceptance remains the independent original-model/halo and paired DPA run.
"""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from profile_scaling import assert_outputs, geometry, neighbor_statistics
from chorus.hp import configure_model, energy_forces_virial
from chorus.hp.density import DensityPlan
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
import mace_ictc.models.message_topology as topology_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--candidate', choices=('stable', 'density', 'stable_density'), required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LAMMPS_MLIAP_MFF.from_checkpoint(args.checkpoint,
        element_types=['H', 'C', 'N', 'O'], device='cuda',
        ictd_fix_product_backend='ictd-bridge-u', ictd_tp_backend='pytorch',
        reuse_checkpoint_u=True).wrapper.model.eval().requires_grad_(False)
    inputs = geometry(10000, model.max_radius, 20260728, 50)
    neighbors = neighbor_statistics(inputs, 50)
    model.skip_input_validation, model.preserve_edge_order = True, True
    plan = configure_model(model, policy='generated', symmetric_contraction=True,
        checkpoint_contraction=True, topology_backend='device', large_graph_fusion=False)

    def run():
        value = energy_forces_virial(model, *inputs)
        return tuple(value[name] for name in ('energy', 'forces', 'atomic_energy', 'virial'))

    original_topology, original_vjp = topology_module.topology, DensityPlan.vjp
    original_positions = inputs[0].clone()
    delta = (torch.arange(inputs[0].numel(), device='cuda').reshape_as(inputs[0]) % 7)*1e-5
    for _ in range(3):
        run()
    expected = tuple(v.clone() for v in run())
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for policy in ('default', args.candidate, 'default'):
        topology_module.topology = topology_module.stable_topology if 'stable' in policy else original_topology
        DensityPlan.vjp = DensityPlan.gather_vjp if 'density' in policy else original_vjp
        inputs[0].copy_(original_positions)
        for _ in range(3):
            actual = run()
        checks = []

        def check(got, ref, label):
            errors = {name: {'max_abs': float((a-b).abs().max()),
                'relative_l2': float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-20))}
                for name, a, b in zip(('energy', 'forces', 'atomic_energy', 'virial'), got, ref)}
            try:
                assert_outputs(got, ref, 10000, virial=True)
                error = None
            except AssertionError as exc:
                error = str(exc)
            checks.append({'label': label, 'passed': error is None, 'error': error, 'errors': errors})

        check(actual, expected, 'same_input_current_HP_oracle')
        del actual
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):captured = run()
        for changed in (False, True):
            inputs[0].copy_(original_positions + delta if changed else original_positions)
            cg.replay()
            for repeat in range(5):
                check(captured, run(), f'live_capture_changed_{changed}_repeat_{repeat}')
        inputs[0].copy_(original_positions)
        samples = []
        for _ in range(15):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cg.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        row = {'policy': policy, 'screen_checks_passed': all(c['passed'] for c in checks),
            'formal_qualification': False, 'checks': checks, 'median_ms': statistics.median(samples),
            'samples_ms': samples, 'peak_reserved': torch.cuda.max_memory_reserved(),
            'peak_allocated': torch.cuda.max_memory_allocated()}
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)
        del cg, captured
        gc.collect()
        torch.cuda.empty_cache()
    topology_module.topology, DensityPlan.vjp = original_topology, original_vjp
    args.out.write_text(json.dumps({'checkpoint': args.checkpoint, 'neighbors': neighbors,
        'plan': plan, 'results': results, 'source_snapshot': str(Path(__file__).resolve().parents[2])}, indent=2)+'\n')


if __name__ == '__main__':main()
