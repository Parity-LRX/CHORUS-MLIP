"""Within-sbatch paired framework orchestration; no project computation locally."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from architecture_cases import CASES, MODEL_SEED, GEOMETRY_SEED, chorus_spec, dpa_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--hp-source', type=Path, required=True)
    p.add_argument('--sizes', nargs='+', type=int)
    p.add_argument('--result-set', default='results')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('This workload must run inside the designated A100 sbatch allocation')
    assets = Path('/public/home/sps-xia/rxlin/a100_scaling_assets_20260804')
    overlay = assets/'dpa4_torch211_overlay'
    native_python = '/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python'
    dpa_python = '/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python'
    scripts = args.source/'benchmarks/hp'
    case_dir = args.root/args.result_set/args.case
    logs = args.root/('logs' if args.result_set == 'results' else 'logs_'+args.result_set)/args.case
    logs.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    native_env = dict(os.environ)
    native_env.update(PYTHONPATH=f'{args.source}:{args.hp_source}',
        NVIDIA_TF32_OVERRIDE='0', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MAX_JOBS='4', TORCH_CUDA_ARCH_LIST='8.0',
        CUDA_HOME='/usr/local/cuda', PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    # Reuse this task's already-qualified compiler cache. Shapes form distinct
    # keys, and no source snapshot or original checkpoint is modified.
    cache = Path('/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache/native_v1_1')
    native_env.update(XDG_CACHE_HOME=str(cache/'xdg'), TORCH_EXTENSIONS_DIR=str(cache/'extensions'),
        FSCETP_ICTD_CACHE_DIR=str(cache/'ictd'), TRITON_CACHE_DIR=str(cache/'triton'),
        PYTHONPYCACHEPREFIX=str(cache/'pycache'), TMPDIR=str(cache/'tmp'))
    dpa_env = dict(native_env)
    libs = sorted(str(v) for v in (overlay/'nvidia').glob('*/lib') if v.is_dir())
    dpa_env.update(PYTHONPATH=f'{overlay}:{args.source}',
        LD_LIBRARY_PATH=':'.join([str(assets/'mpi_libs'), str(overlay/'torch/lib'), *libs,
                                 os.environ.get('LD_LIBRARY_PATH', '')]),
        DP_INTERFACE_PREC='low', DP_COMPILE_INFER='1', DP_TRITON_INFER='2',
        DP_TF32_INFER='0', DP_CUTE_INFER='0',
        DP_INTRA_OP_PARALLELISM_THREADS='1', DP_INTER_OP_PARALLELISM_THREADS='1')
    dcache = args.root/'cache/dpa_compile2'
    for part in ('xdg', 'inductor', 'triton', 'pycache', 'tmp'):(dcache/part).mkdir(parents=True, exist_ok=True)
    dpa_env.update(XDG_CACHE_HOME=str(dcache/'xdg'), TORCHINDUCTOR_CACHE_DIR=str(dcache/'inductor'),
        TRITON_CACHE_DIR=str(dcache/'triton'), PYTHONPYCACHEPREFIX=str(dcache/'pycache'), TMPDIR=str(dcache/'tmp'))
    sizes = args.sizes or ([1024, 10000] if CASES[args.case]['large'] else [1024])
    manifest = {'case': args.case, 'sizes': sizes, 'model_seed': MODEL_SEED,
        'geometry_seed': GEOMETRY_SEED, 'neighbors': 50,
        'chorus': [chorus_spec(args.case, s) for s in ('final', 'persistent')],
        'dpa4': dpa_spec(args.case), 'source': str(args.source), 'hp_source': str(args.hp_source),
        'slurm_job': os.environ['SLURM_JOB_ID'], 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'note': 'Constructor initializations for scaling only; not trained/accuracy matched. Channel definitions differ.'}
    (case_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)
    failures = []

    def child(label, command, env, out=None):
        log = logs/f'{label}.log'
        started = time.time()
        with log.open('w') as stream:
            process = subprocess.Popen(command, env=env, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT)
            print(json.dumps({'child': label, 'pid': process.pid, 'command': command, 'log': str(log)}), flush=True)
            try:
                code = process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -9
        if code:
            content = log.read_text(errors='replace')
            kind = 'OOM' if 'out of memory' in content.lower() else ('TIMEOUT' if code == -9 else 'FAIL')
            record = {'status': kind, 'child': label, 'exit_code': code,
                'elapsed_seconds': time.time()-started, 'log': str(log), 'case': args.case}
            if out is not None and not out.exists():out.write_text(json.dumps(record, indent=2)+'\n')
            if kind != 'OOM':failures.append(record)
            print(json.dumps(record), flush=True)
        else:print(json.dumps({'child': label, 'exit_code': code}), flush=True)
        return code

    init_dir = args.root/'initializations'/args.case/'dpa4'
    prepared = child('dpa_prepare', [dpa_python, str(scripts/'prepare_architecture_dpa.py'),
        '--case', args.case, '--template', str(assets/'dpa4/c32/input.json'), '--out-dir', str(init_dir)], dpa_env)
    for n in sizes:
        destination = case_dir/f'n{n}'
        destination.mkdir(parents=True, exist_ok=True)
        if prepared == 0:
            for boundary in ('full', 'lower'):
                out = destination/f'dpa4_{boundary}.json'
                child(f'dpa4_{boundary}_n{n}', [dpa_python, str(scripts/'profile_dpa_boundary.py'),
                    '--atoms', str(n), '--neighbors', '50', '--boundary', boundary,
                    '--config', str(init_dir/'input.json'), '--checkpoint', str(init_dir/'initialization.pt'),
                    '--out', str(out)], dpa_env, out)
        for scope in ('final', 'persistent'):
            out = destination/f'chorus_{scope}.json'
            child(f'chorus_{scope}_n{n}', [native_python, str(scripts/'profile_architecture_chorus.py'),
                '--case', args.case, '--scope', scope, '--atoms', str(n),
                '--state-dir', str(args.root/'initializations'/args.case), '--out', str(out)], native_env, out)
    (case_dir/'case_status.json').write_text(json.dumps({'status': 'FAIL' if failures else 'DONE',
        'failures': failures, 'slurm_job': os.environ['SLURM_JOB_ID']}, indent=2)+'\n')
    print(json.dumps({'case': args.case, 'status': 'FAIL' if failures else 'DONE', 'failures': failures}), flush=True)
    raise SystemExit(bool(failures))


if __name__ == '__main__':main()
