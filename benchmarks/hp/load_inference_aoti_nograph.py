"""Fresh-process standalone .pt2 loading: no CHORUS or HP imports."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from no_cuda_graph import forbid_cuda_graph


def check(actual,expected,n):
    for i,(a,b) in enumerate(zip(actual,expected)):
        if i==3:
            if float((a.double()-b.double()).norm())>float(1e-5*b.double().norm()+n*1e-7):
                raise AssertionError('Virial tensor-norm bound exceeded')
        else:
            torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-3)


def main(policy):
    p=argparse.ArgumentParser()
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from torch._inductor import aoti_load_package
    started=time.perf_counter()
    loaded=aoti_load_package(str(args.package))
    load_seconds=time.perf_counter()-started
    data=torch.load(args.package.with_suffix('.validation.pt'),map_location='cpu',weights_only=False)
    graphs=[tuple(t.cuda() for t in item['graph']) for item in data]
    for g,item in zip(graphs,data):
        expected=tuple(t.cuda() for t in item['expected'])
        check(loaded(*g),expected,g[0].shape[0])
    rows=[]
    for index,g in enumerate(graphs):
        for _ in range(5):loaded(*g)
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        samples=[]
        for _ in range(args.repeats):
            torch.cuda.synchronize();start=time.perf_counter()
            loaded(*g)
            torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
        rows.append(dict(atoms=g[0].shape[0],edges=g[3].shape[0],median_ms=statistics.median(samples),
            samples_ms=samples,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            process_resident_mib=[int(line.split(',')[1]) for line in subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
                if line.split(',')[0].strip()==str(os.getpid())]))
    imported=[name for name in sys.modules if name=='chorus' or name.startswith(('chorus.','mace_ictc.'))]
    if imported:
        raise RuntimeError('Standalone loader imported model/custom Python operators: '+str(imported))
    result=dict(status='PASS',package=str(args.package),package_load_seconds=load_seconds,
        graph_execution_policy=policy,standalone_without_model_imports=True,torch=torch.__version__,
        pid=os.getpid(),slurm_job=os.environ.get('SLURM_JOB_ID'),results=rows,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        command=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode())
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            loaded(*graphs[-1]);torch.cuda.synchronize()
        args.out.with_suffix('.profile.txt').write_text(prof.key_averages().table(
            sort_by='self_cuda_time_total',row_limit=60))


if __name__=='__main__':
    with forbid_cuda_graph(purpose='inference') as policy:
        main(policy)
