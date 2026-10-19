"""Same-allocation large-capacity E/F/virial controls; invoked only by sbatch."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--hp-source', type=Path, required=True)
    p.add_argument('--atoms', type=int, required=True)
    p.add_argument('--scope', choices=('final', 'persistent'), required=True)
    p.add_argument('--experiment', default='persistent_distance_destination')
    p.add_argument('--experiments', nargs='+', help='Explicit same-card projection controls')
    p.add_argument('--tiles', type=int, nargs='+', default=[65536, 131072])
    p.add_argument('--tag', default='capacity_v4')
    p.add_argument('--dpa-reference-atoms', type=int)
    p.add_argument('--skip-dpa', action='store_true', help='Native-only numerical diagnostic; not a new paired DPA claim')
    p.add_argument('--native-wrapper', type=Path,
                   help='Explicit versioned execution-route qualifier; recorded in the manifest')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Designated sbatch allocation required')
    if Path(args.tag).name != args.tag or args.tag in ('', '.', '..'):
        raise ValueError('Result tag must be one directory name')
    if any(tile <= 0 for tile in args.tiles):raise ValueError('Positive tile sizes required')
    destination = args.root/'results'/args.tag/args.scope/f'n{args.atoms}'
    logs = args.root/'logs'/args.tag/args.scope/f'n{args.atoms}'
    destination.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    manifest = dict(scope=args.scope, atoms=args.atoms, neighbors=50, outputs='E/F/virial',
        slurm_job=os.environ['SLURM_JOB_ID'], pid=os.getpid(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), source=str(args.source),
        hp_source=str(args.hp_source), experiment=args.experiment, experiments=args.experiments, tiles=args.tiles,
        dpa_original_reference_atoms=args.dpa_reference_atoms or args.atoms, skip_dpa=args.skip_dpa,
        native_wrapper=str(args.native_wrapper) if args.native_wrapper else None)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)
    failures = []

    def execute(label, command, env):
        output, log = destination/f'{label}.json', logs/f'{label}.log'
        command = [*command, '--out', str(output)]
        with log.open('w') as stream:
            child = subprocess.Popen(command, cwd=args.root, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
            print(json.dumps(dict(label=label, pid=child.pid, command=command, log=str(log))), flush=True)
            try:code = child.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                code = -9
        if code:
            content = log.read_text(errors='replace')
            status = 'OOM' if 'out of memory' in content.lower() else 'TIMEOUT' if code == -9 else 'FAIL'
            record = dict(status=status, exit_code=code, command=command, log=str(log))
            stages = [line for line in content.splitlines() if line.startswith('{"stage":')]
            if stages:record['last_progress'] = json.loads(stages[-1])
            if not output.exists():output.write_text(json.dumps(record, indent=2)+'\n')
            if status != 'OOM':failures.append(record)
        print(json.dumps(dict(label=label, exit_code=code)), flush=True)

    assets = Path('/public/home/sps-xia/rxlin/a100_scaling_assets_20260804')
    overlay = assets/'dpa4_torch211_overlay'
    cache = Path('/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache')
    dpa_cache = cache/'dpa_fast_t2'
    libraries = [str(path) for path in sorted((overlay/'nvidia').glob('*/lib'))]
    dpa_env = dict(os.environ, PYTHONPATH=f'{overlay}:{args.source}',
        LD_LIBRARY_PATH=':'.join([str(assets/'mpi_libs'), str(overlay/'torch/lib'),
                                 *libraries, os.environ.get('LD_LIBRARY_PATH', '')]),
        XDG_CACHE_HOME=str(dpa_cache/'xdg'), TORCHINDUCTOR_CACHE_DIR=str(dpa_cache/'inductor'),
        TRITON_CACHE_DIR=str(dpa_cache/'triton'), PYTHONPYCACHEPREFIX=str(dpa_cache/'pycache'),
        TMPDIR=str(dpa_cache/'tmp'), DP_COMPILE_INFER='1', DP_TRITON_INFER='2',
        DP_TF32_INFER='0', DP_CUTE_INFER='0',
        DP_INTRA_OP_PARALLELISM_THREADS='1', DP_INTER_OP_PARALLELISM_THREADS='1')
    reference_flags = ([] if args.dpa_reference_atoms is None else
                       ['--reference-atoms', str(args.dpa_reference_atoms)])
    for boundary in (() if args.skip_dpa else ('full', 'lower')):
        execute(f'dpa4_{boundary}', ['/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python',
            str(args.source/'benchmarks/hp/profile_dpa_boundary.py'), '--atoms', str(args.atoms),
            '--neighbors', '50', '--boundary', boundary, '--config', str(assets/'dpa4/c32/input.json'),
            '--checkpoint', str(assets/'dpa4/c32/model.pt'), *reference_flags], dpa_env)
    native_cache = cache/'native_v1_1'
    native_env = dict(os.environ, PYTHONPATH=f'{args.source}:{args.hp_source}',
        XDG_CACHE_HOME=str(native_cache/'xdg'), TORCH_EXTENSIONS_DIR=str(native_cache/'extensions'),
        FSCETP_ICTD_CACHE_DIR=str(native_cache/'ictd'), TRITON_CACHE_DIR=str(native_cache/'triton'),
        PYTHONPYCACHEPREFIX=str(native_cache/'pycache'), TMPDIR=str(native_cache/'tmp'))
    checkpoint = (assets/'persistent_t1x.pth' if args.scope == 'persistent' else
        Path('/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth'))
    oracle = args.root/'cache/verification'/f'{args.scope}_n{args.atoms}_k50_fp32.pt'
    if args.scope == 'persistent' and args.atoms == 10000:
        oracle = cache/'verification/persistent_n10000_k50_fp32_v67.pt'
    cases = [('default', [])]
    for experiment in args.experiments or [args.experiment]:
        for tile in args.tiles:
            label = (f'{experiment}_tiles_{tile}' if args.experiments else f'jvp_tiles_{tile}')
            flags = (['--execution-experiment', 'none'] if experiment.startswith('public_') else
                     ['--execution-experiment', experiment])
            if experiment.removesuffix('_forcecsr') != 'none':
                flags += ['--radial-jvp', '--radial-gate-pipeline', '--radial-tile-edges', str(tile)]
            if experiment == 'public_destination':flags += ['--radial-destination']
            if experiment == 'public_cached':flags += ['--radial-destination', '--radial-cache-gates']
            cases.append((label, flags))
    for label, flags in cases:
        execute(label, ['/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python',
            str(args.native_wrapper or args.source/'benchmarks/hp/profile_scaling.py'), '--checkpoint', str(checkpoint),
            '--atoms', str(args.atoms), '--neighbors', '50', '--policy', 'generated', '--sc', 'on',
            '--checkpoint-sc', '--topology-backend', 'device', '--capture', '--virial',
            '--allow-reference-oom', '--reference-cache', str(oracle), '--profile', *flags], native_env)
    (destination/'case_status.json').write_text(json.dumps(dict(
        status='FAIL' if failures else 'DONE', failures=failures), indent=2)+'\n')
    raise SystemExit(bool(failures))


if __name__ == '__main__':main()
