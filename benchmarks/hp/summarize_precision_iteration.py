"""XHPC-batch-only aggregation of raw numerical diagnostics and screens."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('sbatch allocation required')
    record = dict(slurm_job=os.environ['SLURM_JOB_ID'], precision=[], screens=[], paired=[])
    for directory in sorted((args.root/'results').glob('precision_ablation_*')):
        for path in sorted(directory.glob('*.json')):
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or 'case' not in data:continue
            repeats = data.get('repeats', [])
            row = dict(file=str(path), case=data['case'], status=data.get('status'),
                force_failed=[r['parity']['forces']['failed_components'] for r in repeats],
                force_max_bound_ratio=[r['parity']['forces']['max_bound_ratio'] for r in repeats])
            record['precision'].append(row)
    for path in sorted((args.root/'results/destination_schedule_v11').glob('*.json')):
        data = json.loads(path.read_text())
        record['screens'].append(dict(file=str(path), candidates=data.get('candidates', [])))
    for directory in sorted([*(args.root/'results').glob('cached_gates_*'),
                             *(args.root/'results').glob('final_stable_maps_v*')]):
        for path in sorted(directory.glob('*/n*/*.json')):
            if path.name.endswith('.qualification.json'):continue
            if path.name in ('case_status.json', 'manifest.json'):continue
            data = json.loads(path.read_text())
            if 'median_ms' not in data and 'status' not in data:continue
            record['paired'].append(dict(file=str(path), median_ms=data.get('median_ms'),
                peak_reserved=data.get('peak_reserved'), status=data.get('status'),
                capture_validation=data.get('capture_validation'), parity=data.get('parity')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(precision=record['precision'], paired=record['paired'],
        screens=[dict(file=screen['file'], candidates=[dict(options=row['options'],
            status=row['status'], median_ms=row.get('median_ms')) for row in screen['candidates']])
            for screen in record['screens']]), indent=2), flush=True)


if __name__ == '__main__':main()
