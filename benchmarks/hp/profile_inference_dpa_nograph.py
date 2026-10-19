"""Same-geometry DPA4 compiled E/F/virial inference, with CUDA Graph forbidden."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

os.environ.update(DP_INTERFACE_PREC='low',DP_COMPILE_INFER='1',DP_TRITON_INFER='2',
    DP_TF32_INFER='0',DP_CUTE_INFER='0')
import torch
from deepmd.pt.model.model import get_model
from deepmd.utils.argcheck import normalize
from no_cuda_graph import forbid_cuda_graph


def main(policy):
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--atoms',type=int,nargs='+',default=[128,256,1024])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    config=normalize(json.loads(args.config.read_text()))
    config['model']['descriptor']['use_amp']=False
    config['model'].update(use_compile=True,enable_tf32=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['model']
    state={key.removeprefix('model.Default.'):value for key,value in state.items() if key.startswith('model.Default.')}
    def build():
        model=get_model(copy.deepcopy(config['model']))
        model.load_state_dict(state,strict=True)
        return model.cuda().eval().requires_grad_(False)
    model=build()
    if not model.should_use_compile():raise RuntimeError('DPA inference compile is disabled')
    script=Path(os.environ['SOURCE'])/'benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py'
    spec=importlib.util.spec_from_file_location('aoti_common_geometry',script)
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    from ase.neighborlist import neighbor_list
    import numpy as np
    types={symbol:i for i,symbol in enumerate(config['model']['type_map'])}
    results=[]
    for n in args.atoms:
        atoms=common.make_atoms(n,model.get_rcut(),50,20260729 if n!=128 else 20260728)
        src,dst=neighbor_list('ij',atoms,model.get_rcut(),self_interaction=False)
        degree=np.bincount(src,minlength=n)
        if int(degree.min())!=50 or int(degree.max())!=50:raise RuntimeError('Cutoff graph differs')
        coord=torch.tensor(atoms.positions,device='cuda',dtype=torch.float32)[None]
        atype=torch.tensor([[types[s] for s in atoms.get_chemical_symbols()]],device='cuda')
        box=torch.tensor(atoms.cell.array,device='cuda',dtype=torch.float32).reshape(1,9)
        schema=model.build_neighbor_list(coord,atype,box)
        if int(schema.edge_mask.sum())!=n*50:raise RuntimeError('DPA edge count mismatch')
        def full():return model(coord,atype,box)
        def lower():
            out=model.forward_lower(schema.coord,schema.atype,schema.edge_index,schema.edge_vec,
                schema.edge_scatter_index,schema.edge_mask)
            return {'energy':out['energy'],'force':out['extended_force'],'virial':out['virial']}
        fast=full();compact=lower()
        for name in ('energy','force','virial'):
            torch.testing.assert_close(fast[name],compact[name],atol=2e-4,rtol=1e-3)
        # Literal formula parity is independent of the accelerated API pair.
        os.environ.update(DP_COMPILE_INFER='0',DP_TRITON_INFER='0')
        try:
            reference=build()
            expected=reference(coord,atype,box)
        finally:
            os.environ.update(DP_COMPILE_INFER='1',DP_TRITON_INFER='2')
        parity={}
        for name in ('energy','force','virial'):
            torch.testing.assert_close(fast[name],expected[name],atol=2e-4,rtol=1e-3)
            parity[name]=dict(max_abs=float((fast[name]-expected[name]).abs().max()),
                relative_l2=float((fast[name].double()-expected[name].double()).norm()/expected[name].double().norm().clamp_min(1e-20)))
        del reference,expected,fast,compact
        for boundary,run in (('lower',lower),('full',full)):
            for _ in range(5):run()
            torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            samples=[]
            for _ in range(args.repeats):
                torch.cuda.synchronize();started=time.perf_counter()
                value=run()
                torch.cuda.synchronize();samples.append((time.perf_counter()-started)*1000)
                if not all(bool(value[k].isfinite().all()) for k in ('energy','force','virial')):
                    raise RuntimeError('Nonfinite DPA output')
                del value
            results.append(dict(atoms=n,neighbors=50,boundary=boundary,median_ms=statistics.median(samples),
                samples_ms=samples,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                original_formula_full_geometry=True,original_parity=parity))
    modules=[dict(name=name,configured_level=module.triton_infer_level,
        actual_value_path=type(getattr(module,'_triton_value_path',None)).__name__)
        for name,module in model.named_modules() if hasattr(module,'triton_infer_level')]
    if not model.compiled_core_compute_cache:raise RuntimeError('No DPA compiled core was created')
    record=dict(status='PASS',checkpoint=str(args.checkpoint),config=str(args.config),
        parameters=sum(v.numel() for v in model.parameters()),descriptor=config['model']['descriptor'],
        graph_execution_policy=policy,compiled_core_slots=len(model.compiled_core_compute_cache),
        triton_modules=modules,level2_value_path_active=all(m['actual_value_path']=='_TritonSO2ValuePath' for m in modules),
        results=results,outputs='energy+forces+virial',torch=torch.__version__,
        arithmetic='FP32 network/forces; no AMP/TF32; upstream FP64 reductions',
        pid=os.getpid(),command=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),
        slurm_job=os.environ.get('SLURM_JOB_ID'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    with forbid_cuda_graph(purpose='inference') as policy:main(policy)
