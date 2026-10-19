"""Build a real dynamic-N/E E/F/virial .pt2 in an allocated remote batch job."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch
from no_cuda_graph import forbid_cuda_graph
from inference_aoti_common import load_architecture,compute,strict_fp32,identity
from profile_scaling import geometry,neighbor_statistics,reference_values,assert_outputs


def main(policy):
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True)
    p.add_argument('--scope',choices=('persistent','final'),required=True)
    p.add_argument('--initialization',type=Path,required=True)
    p.add_argument('--trace-atoms',type=int,default=128)
    p.add_argument('--vary-atoms',type=int,default=256)
    p.add_argument('--benchmark-atoms',type=int,nargs='+',default=[1024])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=('prepared','aten'),default='prepared')
    p.add_argument('--message-schedule',choices=('atomic','csr'),default='atomic')
    p.add_argument('--static',action='store_true')
    p.add_argument('--max-atoms',type=int,default=131072)
    p.add_argument('--max-edges',type=int,default=8388608)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('A100 export must run through sbatch')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    log=args.out.with_suffix('.export.json')
    record=dict(status='RUNNING',case=args.case,scope=args.scope,backend=args.backend,
        package=str(args.out),initialization=str(args.initialization),graph_execution_policy=policy,
        dynamic=not args.static,outputs='energy+forces+atomic_energy+virial',**identity())
    record['declared_shape_bounds']=dict(atoms=[2,args.max_atoms],edges=[2,args.max_edges])
    def mark(stage,**extra):
        record.update(stage=stage,**extra)
        log.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(stage=stage,case=args.case,scope=args.scope,**extra)),flush=True)
    try:
        strict_fp32()
        mark('load_fixed_state')
        model,spec=load_architecture(args.case,args.scope,args.initialization)
        record.update(spec=spec,parameters=sum(p.numel() for p in model.parameters()))
        graph=geometry(args.trace_atoms,model.max_radius,20260728,50)
        record['neighbors']=neighbor_statistics(graph,50)
        mark('literal_original_efv')
        original=reference_values(model,graph,True)
        if args.backend=='prepared':
            mark('prepare_triton_operators')
            from chorus.hp import prepare_for_export
            record['prepared']=prepare_for_export(model,dynamic=not args.static,message_schedule=args.message_schedule)
        evaluate=lambda *g:compute(model,*g)
        got=evaluate(*graph)
        assert_outputs(got,original,args.trace_atoms,virial=True)
        record['prepared_parity']={name:float((a.double()-b.double()).abs().max())
            for name,a,b in zip(('energy','forces','atomic_energy','virial'),got,original)}
        del got
        mark('flatten_first_derivatives')
        from chorus.training.makefx_compile import trace_and_compile_force
        gm=trace_and_compile_force(model,graph,training=False,compute_fn=evaluate,do_compile=False)
        record['fx_nodes']=len(list(gm.graph.nodes))
        args.out.with_suffix('.fx.py').write_text(gm.code)
        assert_outputs(gm(*graph),original,args.trace_atoms,virial=True)
        mark('torch_export',fx_nodes=record['fx_nodes'])
        shapes=None
        if not args.static:
            n=torch.export.Dim('atoms',min=2,max=args.max_atoms)
            e=torch.export.Dim('edges',min=2,max=args.max_edges)
            shapes=({0:n},{0:n},{0:n},{0:e},{0:e},{0:e},None)
        exported=torch.export.export(gm,tuple(graph),dynamic_shapes=shapes,strict=True)
        mark('compile_and_package')
        started=time.perf_counter()
        from torch._inductor import aoti_compile_and_package,aoti_load_package
        package=aoti_compile_and_package(exported,package_path=str(args.out),
            inductor_configs={'triton.cudagraphs':False})
        record['compile_seconds']=time.perf_counter()-started
        mark('load_compiled_package',compile_seconds=record['compile_seconds'])
        loaded=aoti_load_package(package)
        actual=loaded(*graph)
        assert_outputs(actual,original,args.trace_atoms,virial=True)
        verification=[dict(atoms=args.trace_atoms,edges=graph[3].numel(),neighbors=50,status='PASS')]
        # Save concrete live values for a fresh process which imports no model
        # or custom operator registration. This detects hidden Python fallback.
        validation=[dict(graph=tuple(v.cpu() for v in graph),expected=tuple(v.cpu() for v in original))]
        from chorus.hp import configure_model
        if not args.static:
            for n,degree in [(args.vary_atoms,50),(args.vary_atoms,32),
                             *((n,50) for n in args.benchmark_atoms)]:
                g=geometry(n,model.max_radius,20260729,degree)
                fast=loaded(*g)
                configure_model(model,policy='reference')
                mark('literal_original_dynamic_verification',verification_atoms=n,verification_neighbors=degree)
                if n>256:
                    from qualify_architecture_offload import checkpointed_original_products
                    with torch.autograd.graph.save_on_cpu(pin_memory=True),checkpointed_original_products(model,32):
                        expected=reference_values(model,g,True)
                else:
                    expected=reference_values(model,g,True)
                assert_outputs(fast,expected,n,virial=True)
                verification.append(dict(atoms=n,edges=g[3].numel(),neighbors=degree,status='PASS',
                    original_formula_full_geometry=True,parity={name:dict(
                        max_abs=float((a.double()-b.double()).abs().max()),
                        relative_l2=float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-20)))
                        for name,a,b in zip(('energy','forces','atomic_energy','virial'),fast,expected)}))
                validation.append(dict(graph=tuple(v.cpu() for v in g),expected=tuple(v.cpu() for v in expected)))
        torch.save(validation,args.out.with_suffix('.validation.pt'))
        record.update(status='PASS',stage='DONE',verification=verification,
            package_bytes=args.out.stat().st_size,external_tp_libraries=False)
        mark('DONE')
    except Exception as error:
        record.update(status='FAIL',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        log.write_text(json.dumps(record,indent=2)+'\n')
        raise


if __name__=='__main__':
    with forbid_cuda_graph(purpose='inference') as policy:
        main(policy)
