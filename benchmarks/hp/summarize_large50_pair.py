"""Summarize qualified same-allocation E/F/virial pairs; run in remote sbatch."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sets', nargs='+', required=True)
    parser.add_argument('--model', choices=('final', 'persistent'), required=True)
    parser.add_argument('--training-set', type=str)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in args.sets:
        folder = args.root/name
        full, lower = [json.loads((folder/f'dpa4_{b}_n10000_k50.json').read_text())
                       for b in ('full', 'lower')]
        value = json.loads((folder/f'chorus_{args.model}_fastest_n10000_k50.json').read_text())
        for baseline in (full, lower):
            assert baseline['DP_COMPILE_INFER'] == '1'
            assert baseline['DP_TRITON_INFER'] == '2'
            assert baseline['should_use_compile'] and baseline['compiled_core_slots'] > 0
            assert baseline['model_fp32_tf32_off']
            assert baseline['triton_modules'] and all(m['level'] == 2 and
                m['fused_value_path'] == '_TritonSO2ValuePath' for m in baseline['triton_modules'])
        assert value['atoms'] == 10000 and value['edges'] == 500000
        assert value['neighbors']['min'] == value['neighbors']['max'] == 50
        assert value['outputs'] == 'energy+forces+virial' and value['virial_reduction'] == 'FP64'
        assert value['capture'] == 'captured' and value['capture_validation']['changed_positions']
        assert not value['imported_external_tp']
        assert all(value['parity'][field]['failed_components'] == 0
                   for field in ('energy', 'forces', 'atomic_energy'))
        assert value['CUDA_VISIBLE_DEVICES'] == lower['CUDA_VISIBLE_DEVICES'] == full['CUDA_VISIBLE_DEVICES']
        reserved = max(value['peak_reserved'], value['capture_memory']['capture_peak_reserved'],
                       value['steady_memory']['reserved_including_graph_pool'])
        rows.append({'set': name, 'model': args.model, 'source': value['source_snapshot'],
            'execution_experiment': value['execution_experiment'],
            'chorus_ms': value['median_ms'], 'dpa_full_ms': full['median_ms'],
            'dpa_lower_ms': lower['median_ms'],
            'speedup_vs_full': full['median_ms']/value['median_ms'],
            'speedup_vs_lower': lower['median_ms']/value['median_ms'],
            'chorus_reserved_gib': reserved/2**30, 'dpa_lower_reserved_gib': lower['peak_reserved']/2**30,
            'reserved_reduction_fraction': 1-reserved/lower['peak_reserved'],
            'chorus_capture_allocated_gib': value['capture_memory']['capture_peak_allocated']/2**30,
            'dpa_lower_allocated_gib': lower['peak_allocated']/2**30,
            'force_max_bound_ratio': value['parity']['forces']['max_bound_ratio'],
            'parameters': value['parameters'], 'dispatch': value.get('execution_dispatch')})
    report = {'protocol': 'N10000 K50, original checkpoint, full E/F/virial, FP32/TF32 off; FP64 virial',
              'baseline': 'DPA4-mini compile=1, Triton=2, verified actual compiled/value paths',
              'rows': rows, 'all_at_least_2x': all(r['speedup_vs_lower'] >= 2 for r in rows),
              'all_less_reserved_memory': all(r['reserved_reduction_fraction'] > 0 for r in rows)}
    if args.training_set:
        training = []
        base = args.root/args.training_set
        for n in (128, 512):
            dpa = json.loads((base/f'dpa4_lower_n{n}_k50.json').read_text())
            entry = {'atoms': n, 'dpa_ms': dpa['median_ms']}
            for mode in ('eager', 'capture'):
                value = json.loads((base/f'persistent_{mode}_n{n}_k50.json').read_text())
                assert value['trainable_parameters'] == 1029967
                entry[mode] = {'generated_ms': value['median_ms']['generated'],
                    'reference_ms': value['median_ms']['reference'],
                    'speedup_vs_dpa': dpa['median_ms']/value['median_ms']['generated'],
                    'trainable_parameters': value['trainable_parameters'],
                    'tp_coefficient_trainability': value['tp_coefficient_trainability'],
                    'parameter_gradient_max_relative_l2': max(v['relative_l2']
                        for v in value['parameter_gradient_parity'].values()),
                    'strict_elementwise_gradient_failures': len(value['strict_elementwise_gradient_failures']),
                    'optimizer_update_max_relative_l2': value['optimizer_update_max_relative_l2']}
            training.append(entry)
        report['training_regression'] = training
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
