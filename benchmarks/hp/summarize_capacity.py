"""Aggregate remote large-graph records; execute through the XHPC batch job."""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('XHPC sbatch required')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((args.root/'results').rglob('*.json')):
        if any(s in path.name for s in ('.qualification.', '.trace.')):continue
        if path.name in ('manifest.json', 'case_status.json'):continue
        value = json.loads(path.read_text())
        if 'median_ms' not in value and value.get('status') not in ('OOM', 'FAIL', 'TIMEOUT'):continue
        relative = str(path.relative_to(args.root/'results'))
        if not relative.startswith(('tile_v2_', 'destination_v3_', 'capacity_', 'inline_', 'final_precision_')):continue
        dpa = path.name.startswith('dpa4_')
        reference = value.get('reference_status', {})
        qualification = path.with_suffix('.qualification.json')
        qualified = (json.loads(qualification.read_text()).get('status') == 'PASS'
                     if qualification.exists() else False)
        whole = bool(reference.get('full_stress_geometry'))
        if dpa:
            whole = value.get('accelerated_reference_full_stress_geometry', 'median_ms' in value)
            qualified = 'accelerated_reference_parity' in value
        dispatch = value.get('triton_modules', [])
        actual_t2 = (dpa and value.get('DP_COMPILE_INFER') == '1' and
            value.get('DP_TRITON_INFER') == '2' and bool(dispatch) and
            all(module['level'] == 2 and module['fused_value_path'] != 'NoneType' for module in dispatch))
        row = dict(file=relative, kind='DPA4-mini' if dpa else 'CHORUS',
            status=value.get('status', 'PASS' if qualified else 'UNQUALIFIED'),
            median_ms=value.get('median_ms'), atoms=value.get('atoms'),
            peak_reserved_bytes=value.get('peak_reserved'),
            peak_allocated_bytes=value.get('peak_allocated'),
            process_resident_mib=value.get('process_resident_mib'),
            original_reference_qualified=qualified, original_reference_full_size=whole,
            original_reference_atoms=value.get('accelerated_reference_atoms', reference.get('atoms')),
            actual_compile_and_triton2=bool(actual_t2), capture=value.get('capture'),
            live_coordinate_validation=value.get('capture_validation'),
            execution_experiment=value.get('execution_experiment'),
            radial_destination=value.get('plan', {}).get('radial_destination'),
            source_snapshot=value.get('source_snapshot'),
            CUDA_VISIBLE_DEVICES=value.get('CUDA_VISIBLE_DEVICES'),
            last_progress=value.get('last_progress'), imported_external_tp=value.get('imported_external_tp'))
        if row['median_ms'] is not None and row['atoms']:
            row['atoms_per_second'] = row['atoms']*1000/row['median_ms']
        rows.append(row)
    tests = []
    for path in sorted((args.root/'results').glob('tests*.xml')):
        suites = list(ET.parse(path).getroot().iter('testsuite'))
        tests.append(dict(file=path.name, tests=sum(int(v.get('tests', '0')) for v in suites),
            failures=sum(int(v.get('failures', '0'))+int(v.get('errors', '0')) for v in suites),
            skipped=sum(int(v.get('skipped', '0')) for v in suites)))
    pairs = []
    manifests = [path for prefix in ('capacity_*', 'inline_*')
        for path in (args.root/'results').glob(f'{prefix}/*/n*/manifest.json')]
    for path in sorted(manifests):
        manifest = json.loads(path.read_text())
        lower = path.parent/'dpa4_lower.json'
        if not lower.exists():continue
        baseline = json.loads(lower.read_text())
        if 'median_ms' not in baseline:continue
        for native in sorted(path.parent.glob('*.qualification.json')):
            measured = native.with_name(native.name.replace('.qualification.json', '.json'))
            if not measured.exists():continue
            value = json.loads(measured.read_text())
            if 'median_ms' not in value:continue
            pairs.append(dict(scope=manifest['scope'], atoms=manifest['atoms'], slurm_job=manifest['slurm_job'],
                native_file=str(measured.relative_to(args.root/'results')),
                lower_file=str(lower.relative_to(args.root/'results')),
                dpa_lower_over_native=baseline['median_ms']/value['median_ms'],
                native_reserved_fraction=value['peak_reserved']/baseline['peak_reserved'],
                dpa_original_reference_full_size=baseline.get('accelerated_reference_full_stress_geometry', True)))
    diagnostics = []
    baseline_file = args.root/'results/capacity_v4/final/n32768/default.qualification.pt'
    if baseline_file.exists():
        import torch
        baseline = torch.load(baseline_file, map_location='cpu', weights_only=False)
        baseline_force = baseline['actual'][1]
        reference = baseline['expected'][1]
        candidates = list((args.root/'results/capacity_v4/final/n32768').glob('jvp*.qualification.pt'))
        candidates += list((args.root/'results/final_precision_v5_trained_32768').glob('*.qualification.pt'))
        for path in sorted(candidates):
            value = torch.load(path, map_location='cpu', weights_only=False)
            if not torch.equal(value['expected'][1], reference):raise ValueError('Force diagnostic original oracles differ')
            force = value['actual'][1]
            bound = 1e-4+1e-3*reference.abs()
            failed = ((force-reference).abs() > bound).nonzero()
            components = []
            for n, axis in failed[:20].tolist():
                components.append(dict(atom=n, axis=axis, original=float(reference[n, axis]),
                    optimized=float(force[n, axis]), default=float(baseline_force[n, axis]),
                    bound=float(bound[n, axis]), optimized_delta=float(force[n, axis]-reference[n, axis]),
                    default_delta=float(baseline_force[n, axis]-reference[n, axis])))
            diagnostics.append(dict(file=str(path.relative_to(args.root/'results')),
                failed_force_components=len(failed), max_abs=float((force-reference).abs().max()),
                max_abs_vs_default=float((force-baseline_force).abs().max()), components=components))
    projection_controls = []
    for manifest in sorted((args.root/'results/capacity_public_v7').glob('*/n*/manifest.json')):
        source = manifest.parent/'public_source_tiles_131072.json'
        destination = manifest.parent/'public_destination_tiles_131072.json'
        if not (source.exists() and destination.exists()):continue
        previous, current = (json.loads(path.read_text()) for path in (source, destination))
        if 'median_ms' not in previous or 'median_ms' not in current:continue
        projection_controls.append(dict(atoms=current['atoms'],
            slurm_job=json.loads(manifest.read_text())['slurm_job'],
            source_ms=previous['median_ms'], destination_ms=current['median_ms'],
            throughput_ratio=previous['median_ms']/current['median_ms'],
            time_reduction_percent=100*(1-current['median_ms']/previous['median_ms']),
            reserved_bytes=current['peak_reserved'],
            source_original_reference=previous['reference_status'],
            destination_original_reference=current['reference_status']))
    heldout = args.root/'results/public_v7/persistent_heldout.json'
    heldout_record = json.loads(heldout.read_text()) if heldout.exists() else None
    oracle_audit = args.root/'results/final_oracle_audit_v1.json'
    oracle_audit_record = json.loads(oracle_audit.read_text()) if oracle_audit.exists() else None
    (args.out/'capacity_summary.json').write_text(json.dumps(dict(
        report_slurm_job=os.environ['SLURM_JOB_ID'], rows=rows, tests=tests,
        same_allocation_pairs=pairs, same_card_projection_controls=projection_controls,
        persistent_heldout=heldout_record, final_oracle_diagnostic=oracle_audit_record,
        final_force_diagnostics=diagnostics), indent=2)+'\n')
    lines = ['# A100 large-graph throughput and memory', '',
        'Exact 50 neighbors; complete E/F/virial; neural/force FP32 with TF32 off; FP64 virial reduction.',
        'Frozen inference measurements. Original checkpoint/seed/model terms are unchanged.',
        'Memory is PyTorch reserved bytes including the CHORUS graph pool, shown in decimal GB. Cold compilation, reference generation and profiling are outside timing.', '',
        '| Record | ms | Reserved GB | Original reference | Status |',
        '|---|---:|---:|---|---|']
    for row in rows:
        ms = '—' if row['median_ms'] is None else f'{row["median_ms"]:.3f}'
        memory = '—' if row['peak_reserved_bytes'] is None else f'{row["peak_reserved_bytes"]/1e9:.3f}'
        oracle = ('full size' if row['original_reference_full_size'] else
                  f'{row["original_reference_atoms"]} atoms only') if row['original_reference_qualified'] else 'not passed'
        lines.append(f'| {row["file"]} | {ms} | {memory} | {oracle} | {row["status"]} |')
    lines += ['', '## Same-card source/destination radial controls', '',
        '| Atoms | Source CSR ms | Destination ms | Throughput ratio | Time reduction | Reserved GB |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in projection_controls:
        lines.append(f'| {row["atoms"]} | {row["source_ms"]:.3f} | {row["destination_ms"]:.3f} | '
            f'{row["throughput_ratio"]:.3f} | {row["time_reduction_percent"]:.2f}% | {row["reserved_bytes"]/1e9:.3f} |')
    lines += ['', '## Regression and qualification', '']
    for row in tests:lines.append(f'- {row["file"]}: {row["tests"]} tests, {row["failures"]} failures, {row["skipped"]} skipped.')
    if heldout_record:
        lines += [f'- Public destination path, held-out checkpoint E/F: {heldout_record["status"]}.']
    lines += ['- Final large-size near-zero-force failures remain unqualified; repeated default execution also crossed the component threshold.',
        '- Independent local original-formula closures are diagnostic only and do not replace the registered oracle or qualify failed points.']
    lines += ['', 'DPA rows with a smaller original audit retain full-size full/lower API parity and finite outputs; they do not claim a full-size unaccelerated oracle.',
        'The actual compile/Triton dispatch, source snapshot, same-allocation ratios, failure stage and regression counts are retained in capacity_summary.json.',
        'Failed/OOM records are retained. A smaller-memory path is not automatically a higher-throughput path, and no experimental route is promoted by this report.', '']
    (args.out/'LARGE_CAPACITY.md').write_text('\n'.join(lines))
    print(json.dumps(dict(rows=len(rows), tests=tests, same_allocation_pairs=len(pairs))), flush=True)


if __name__ == '__main__':main()
