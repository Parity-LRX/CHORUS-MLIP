"""Within-sbatch controls for increased atom count and high-L/deep inference."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--hp-source', type=Path, required=True)
    p.add_argument('--scope', choices=('persistent', 'final'), default='persistent')
    p.add_argument('--tag', default='v1')
    p.add_argument('--tile-edges', type=int, default=32768)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Large-graph computation requires the designated sbatch allocation')
    py = '/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python'
    assets = Path('/public/home/sps-xia/rxlin/a100_scaling_assets_20260804')
    label = ('trained_10k_20k', 'trained_32768', 'l3c192d3')[args.index]
    if Path(args.tag).name != args.tag or args.tag in ('', '.', '..'):
        raise ValueError('Result tag must be one directory name')
    destination = args.root/'results'/f'{args.tag}_{label}'
    logs = args.root/'logs'/f'{args.tag}_{label}'
    destination.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f'{args.source}:{args.hp_source}')
    manifest = dict(index=args.index, label=label, source=str(args.source), hp_source=str(args.hp_source),
        slurm_job=os.environ['SLURM_JOB_ID'], CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        scope=args.scope, tile_edges=args.tile_edges,
        protocol='Original model/initialization, exact 50 neighbors, full E/F/virial, FP32/TF32 off')
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)
    sizes = ([10000, 20000], [32768], [1024, 10000])[args.index]
    modes = {
        'default': [], 'jvp': ['--radial-jvp'],
        'tiles': ['--radial-fusion', '--radial-gate-pipeline'],
        'jvp_tiles': ['--radial-jvp', '--radial-gate-pipeline'],
    }
    failures = []
    for n in sizes:
        for mode, flags in modes.items():
            out = destination/f'{args.scope}_{mode}_n{n}.json'
            log = logs/f'{args.scope}_{mode}_n{n}.log'
            if args.index < 2:
                oracle = (Path('/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache/verification/persistent_n10000_k50_fp32_v67.pt')
                          if n == 10000 and args.scope == 'persistent' else args.root/'cache/verification'/f'{args.scope}_n{n}_k50_fp32.pt')
                checkpoint = (assets/'persistent_t1x.pth' if args.scope == 'persistent' else
                    Path('/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth'))
                command = [py, str(args.source/'benchmarks/hp/profile_scaling.py'),
                    '--checkpoint', str(checkpoint), '--atoms', str(n), '--neighbors', '50',
                    '--policy', 'generated', '--sc', 'on', '--checkpoint-sc', '--topology-backend', 'device',
                    '--capture', '--virial', '--allow-reference-oom', '--reference-cache', str(oracle),
                    '--profile', '--radial-tile-edges', str(args.tile_edges), *flags, '--out', str(out)]
            else:
                command = [py, str(args.source/'benchmarks/hp/profile_architecture_chorus.py'),
                    '--case', 'l3c192d3', '--scope', args.scope, '--atoms', str(n),
                    '--state-dir', '/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations/l3c192d3',
                    '--profile', '--radial-tile-edges', str(args.tile_edges), *flags, '--out', str(out)]
            started = time.time()
            with log.open('w') as stream:
                child = subprocess.Popen(command, env=env, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT)
                print(json.dumps(dict(label=label, mode=mode, atoms=n, pid=child.pid,
                    command=command, log=str(log))), flush=True)
                try:
                    code = child.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    code = -9
            if code:
                content = log.read_text(errors='replace')
                status = 'OOM' if 'out of memory' in content.lower() else 'TIMEOUT' if code == -9 else 'FAIL'
                record = dict(status=status, exit_code=code, log=str(log), mode=mode, atoms=n,
                    command=command, elapsed_seconds=time.time()-started)
                if not out.exists():out.write_text(json.dumps(record, indent=2)+'\n')
                if status != 'OOM':failures.append(record)
                print(json.dumps(record), flush=True)
            else:
                print(json.dumps(dict(mode=mode, atoms=n, exit_code=code)), flush=True)
    (destination/'case_status.json').write_text(json.dumps(dict(status='FAIL' if failures else 'DONE',
        failures=failures), indent=2)+'\n')
    raise SystemExit(bool(failures))


if __name__ == '__main__':main()
