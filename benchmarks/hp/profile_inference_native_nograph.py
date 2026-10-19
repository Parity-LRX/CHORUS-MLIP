"""Native fused inference, qualified directly against the original formula.

The optional --package path only reuses historical reference tensors. The
recommended --case/--scope/--initialization entry needs no export or AOTI.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
from no_cuda_graph import forbid_cuda_graph
from inference_aoti_common import load_architecture,strict_fp32,identity
from profile_scaling import assert_outputs,geometry,reference_values
from chorus.hp import configure_model,energy_forces_virial


def main(policy):
    p=argparse.ArgumentParser()
    entry=p.add_mutually_exclusive_group(required=True)
    entry.add_argument('--package',type=Path)
    entry.add_argument('--case')
    p.add_argument('--scope',choices=('persistent','final'))
    p.add_argument('--initialization',type=Path)
    p.add_argument('--atoms',type=int,nargs='+',default=[128,256,1024])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('A100 native numerical qualification requires sbatch')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    strict_fp32()
    if args.package is not None:
        record=json.loads(args.package.with_suffix('.export.json').read_text())
        if record['status']!='PASS':raise ValueError('Original formula qualification has not passed')
        model,spec=load_architecture(record['case'],record['scope'],Path(record['initialization']))
        if sum(p.numel() for p in model.parameters())!=record['parameters']:
            raise ValueError('Native/reference parameter identity mismatch')
        data=torch.load(args.package.with_suffix('.validation.pt'),map_location='cpu',weights_only=False)
        reference_path=args.package.with_suffix('.validation.pt')
    else:
        if args.scope is None or args.initialization is None:
            p.error('--case requires --scope and --initialization')
        model,spec=load_architecture(args.case,args.scope,args.initialization)
        record=dict(case=args.case,scope=args.scope,initialization=str(args.initialization),
                    parameters=sum(p.numel() for p in model.parameters()))
        data=[]
        # The extra degree-32 graph checks live topology, not just repeated
        # execution of one fixed neighbor table.
        cases=[(n,50) for n in args.atoms]+[(min(args.atoms),32)]
        for n,degree in cases:
            g=geometry(n,model.max_radius,20260728 if n==128 else 20260729,degree)
            print(json.dumps(dict(stage='literal_original_efv',atoms=n,neighbors=degree)),flush=True)
            if n>256:
                from qualify_architecture_offload import checkpointed_original_products
                with torch.autograd.graph.save_on_cpu(pin_memory=True),checkpointed_original_products(model,32):
                    expected=reference_values(model,g,True)
            else:
                expected=reference_values(model,g,True)
            data.append(dict(graph=tuple(t.cpu() for t in g),expected=tuple(t.cpu() for t in expected)))
            del g,expected
        reference_path=args.out.with_suffix('.reference.pt')
        torch.save(data,reference_path)
    graphs=[tuple(t.cuda() for t in item['graph']) for item in data]
    info=configure_model(model,policy='generated',checkpoint_contraction=True,topology_backend='device')
    def evaluate(g):
        value=energy_forces_virial(model,*g)
        return value['energy'].sum(),value['forces'],value['atomic_energy'][:,None],value['virial']
    results=[]
    for g,item in zip(graphs,data):
        expected=tuple(t.cuda() for t in item['expected'])
        assert_outputs(evaluate(g),expected,g[0].shape[0],virial=True)
        for _ in range(5):evaluate(g)
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        samples=[]
        for _ in range(args.repeats):
            torch.cuda.synchronize();started=time.perf_counter()
            evaluate(g)
            torch.cuda.synchronize();samples.append((time.perf_counter()-started)*1000)
        results.append(dict(atoms=g[0].shape[0],edges=g[3].shape[0],median_ms=statistics.median(samples),
            samples_ms=samples,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    value=dict(status='PASS',graph_execution_policy=policy,reference_tensors=str(reference_path),
        package_reference=None if args.package is None else str(args.package),
        requires_aoti=False,case=record['case'],scope=record['scope'],
        initialization=record['initialization'],original_formula_full_geometry=True,
        spec=spec,parameters=record['parameters'],plan=info,execution='native fused eager without CUDA Graph',
        results=results,**identity())
    args.out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':
    with forbid_cuda_graph(purpose='inference') as policy:main(policy)
