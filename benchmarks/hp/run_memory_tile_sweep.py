"""Within-sbatch same-card tile controls against the original large oracle."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--atoms', type=int, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--tiles', type=int, nargs='+', default=[32768, 65536, 131072, 262144])
    p.add_argument('--experiment', default='none')
    p.add_argument('--tag', default='tile_v2')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('sbatch allocation required')
    if any(tile <= 0 for tile in args.tiles):raise ValueError('tile sizes must be positive')
    if Path(args.tag).name != args.tag or args.tag in ('', '.', '..'):
        raise ValueError('tag must be one directory name')
    result = args.root/'results'/f'{args.tag}_n{args.atoms}'
    logs = args.root/'logs'/f'{args.tag}_n{args.atoms}'
    result.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    oracle = (Path('/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache/verification/persistent_n10000_k50_fp32_v67.pt')
              if args.atoms == 10000 else args.root/'cache/verification'/f'persistent_n{args.atoms}_k50_fp32.pt')
    (result/'manifest.json').write_text(json.dumps(dict(atoms=args.atoms, slurm_job=os.environ['SLURM_JOB_ID'],
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), source=str(args.source),
        original_oracle=str(oracle), tiles=args.tiles, experiment=args.experiment,
        baseline_experiment='none'), indent=2)+'\n')
    failures = []
    for tile in (0, *args.tiles):
        label = 'default' if not tile else f'jvp_tiles_{tile}'
        flags = [] if not tile else ['--radial-jvp', '--radial-gate-pipeline', '--radial-tile-edges', str(tile)]
        if tile:flags += ['--execution-experiment', args.experiment]
        out = result/f'{label}.json'
        log = logs/f'{label}.log'
        command = ['/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python',
            str(args.source/'benchmarks/hp/profile_scaling.py'),
            '--checkpoint', '/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth',
            '--atoms', str(args.atoms), '--neighbors', '50', '--policy', 'generated', '--sc', 'on',
            '--checkpoint-sc', '--topology-backend', 'device', '--capture', '--virial',
            '--allow-reference-oom', '--reference-cache', str(oracle), '--profile', *flags, '--out', str(out)]
        with log.open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=args.root)
            print(json.dumps(dict(label=label, pid=child.pid, command=command, log=str(log))), flush=True)
            try:code = child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                code = -9
        if code:
            status = 'OOM' if 'out of memory' in log.read_text(errors='replace').lower() else 'FAIL'
            value = dict(status=status, log=str(log), command=command, exit_code=code)
            if not out.exists():out.write_text(json.dumps(value, indent=2)+'\n')
            if status != 'OOM':failures.append(value)
        print(json.dumps(dict(label=label, exit_code=code)), flush=True)
    raise SystemExit(bool(failures))


if __name__ == '__main__':main()
