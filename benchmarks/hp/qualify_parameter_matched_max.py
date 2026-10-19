"""Independent high-host-memory original-formula qualification, never timed."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_training_scale_matrix import FUSION
from run_training_mixed_sweep import VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--scopes', nargs='+', choices=('persistent', 'final'), default=['persistent', 'final'])
    parser.add_argument('--atoms', type=int, default=512)
    parser.add_argument('--result-set', default='parameter_matched_oracle_v1')
    parser.add_argument('--reference-tp-chunk-edges', type=int, default=1024)
    parser.add_argument('--reference-sc-chunk-nodes', type=int, default=0)
    parser.add_argument('--reference-sc-chunk-channels', type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'GPU-A100':
        raise RuntimeError('Original qualification requires the designated A100 sbatch allocation')
    root, tools, case = args.root, args.tools, 'l3c256d4'
    manifest = json.loads((root/'results/highl_parameter_matched_v5'/case/'manifest.json').read_text())
    if manifest['source'] != os.environ['SOURCE'] or manifest['hp_source'] != os.environ['HP_SOURCE']:
        raise ValueError('Untimed oracle must use the same source snapshot as the timed run')
    outdir, logdir = root/'results'/args.result_set/case/f'n{args.atoms}', root/'logs'/args.result_set/case
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    records = []
    for scope in args.scopes:
        output, log = outdir/f'{scope}.json', logdir/f'{scope}.log'
        if output.exists() or log.exists():
            raise FileExistsError(output)
        checkpoint = Path('/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations')/case/f'chorus_{scope}_{case}_initialization.pt'
        command = ['/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python',
            str(tools/'profile_training_architecture_nograph.py'), '--architecture-case', case,
            '--scope', scope, '--checkpoint', str(checkpoint), '--atoms', str(args.atoms), '--neighbors', '50',
            '--rounds', '1', '--warmup', '1', '--repeats', '1', *FUSION,
            *VARIANTS[manifest['scoped_variants'][scope]], '--reference-save-on-cpu',
            '--reference-empty-cache', '--reference-tp-chunk-edges', str(args.reference_tp_chunk_edges),
            '--reference-sc-chunk-nodes', str(args.reference_sc_chunk_nodes),
            '--reference-sc-chunk-channels', str(args.reference_sc_chunk_channels),
            '--tag', 'untimed_same_size_original_qualification', '--out', str(output)]
        entry = dict(scope=scope, command=command, slurm_job=os.environ['SLURM_JOB_ID'],
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), log=str(log), output=str(output),
            purpose='untimed_original_oracle_not_a_throughput_measurement')
        with log.open('x') as stream:
            stream.write(json.dumps(entry)+'\n')
            stream.flush()
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=root)
            entry['pid'] = process.pid
            print(json.dumps(entry), flush=True)
            (outdir/'active_child.json').write_text(json.dumps(entry, indent=2)+'\n')
            try:
                code = process.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -9
        if code:
            content = log.read_text(errors='replace')
            status = 'OOM' if 'out of memory' in content.lower() else 'TIMEOUT' if code == -9 else 'FAIL'
            if not output.exists():output.write_text(json.dumps(dict(status=status, **entry), indent=2)+'\n')
        else:
            status = json.loads(output.read_text())['status']
        entry.update(status=status, exit_code=code)
        records.append(entry)
        (outdir/'child_records.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(entry), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
