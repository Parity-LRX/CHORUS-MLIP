"""Native OEQ before/after full E/F/virial and force-loss/AdamW benchmark."""
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from common import build,config,outputs,loss,periodic_graph,strict_fp32,variant
from no_cuda_graph import forbid_cuda_graph


def check(actual,expected,n):
    parity=[]
    for index,(a,b) in enumerate(zip(actual,expected)):
        b=b.to(a.device)
        delta=(a.double()-b.double()).norm()
        if index==2:
            if float(delta)>float(1e-4*b.double().norm()+n*1e-6):
                raise AssertionError('Native NequIP FP32 virial norm bound exceeded')
        else:
            torch.testing.assert_close(a,b,atol=2e-4,rtol=5e-4)
        parity.append(dict(max_abs=float((a.double()-b.double()).abs().max()),
            relative_l2=float(delta/b.double().norm().clamp_min(1e-20))))
    return parity


def main(policy):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('A100 benchmark requires sbatch')
    strict_fp32()
    scope=os.environ.get('SCOPE','persistent')
    channels=int(os.environ.get('CHANNELS','64'));lmax=int(os.environ.get('LMAX','2'))
    layers=int(os.environ.get('LAYERS','2'));rank=int(os.environ.get('RANK','16'))
    # Colons survive Slurm's comma-delimited --export syntax unchanged.
    sizes=[int(n) for n in os.environ.get('SIZES','128:1024:4096').replace(':',',').split(',')]
    tasks=os.environ.get('TASKS','inference,train').split(',')
    max_train=int(os.environ.get('MAX_TRAIN_ATOMS','1024'))
    out=Path(os.environ['OUT']);out.mkdir(parents=True,exist_ok=True)
    spec=config(scope,channels,lmax,layers,rank)
    original=build(spec,oeq=False)
    state={name:value.detach().cpu().clone() for name,value in original.state_dict().items()}
    parameters=sum(p.numel() for p in original.parameters())
    torch.save(dict(config=spec,state=state,seed=20261018),out/'initialization.pt')
    del original;gc.collect();torch.cuda.empty_cache()
    records=[];refs={};grad_refs={}
    report=dict(status='RUNNING',config=spec,sizes=sizes,parameters=parameters,torch=torch.__version__,
        slurm_job=os.environ['SLURM_JOB_ID'],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        graph_execution_policy=policy,arithmetic='FP32, no AMP or TF32; native NequIP FP32 virial',
        tasks='inference=full E/F/virial; train=E/F/virial loss + mixed backward + AdamW',
        initialization='same constructor state; no retraining or model changes',results=records)
    path=out/'benchmark.json'
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for mode in ('tp_only','legacy','optimized'):
        model=build(spec,state=state)
        count=variant(model,mode)
        if not count:raise RuntimeError('No active CHORUS module')
        for task in tasks:
            for n in sizes:
                if task=='train' and n>max_train:continue
                item=dict(mode=mode,task=task,atoms=n,neighbors=50,edges=n*50,status='RUNNING')
                records.append(item);save()
                print(json.dumps(dict(stage='case',**item)),flush=True)
                if mode!='tp_only' and not any(r['mode']=='tp_only' and r['task']==task
                        and r['atoms']==n and r['status']=='PASS' for r in records):
                    item['status']='SKIP_REFERENCE_OOM';save();continue
                try:
                    model.load_state_dict(state,strict=True)
                    model.train(task=='train').requires_grad_(task=='train')
                    model.zero_grad(set_to_none=True)
                    data,geometry=periodic_graph(n)
                    values=outputs(model,data)
                    key=(task,n)
                    if mode=='tp_only':refs[key]=tuple(v.detach().cpu() for v in values)
                    item['original_outputs']=check(values,refs[key],n)
                    if task=='train':
                        loss(values,n).backward()
                        gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()
                                   if p.grad is not None}
                        if mode=='tp_only':grad_refs[n]=gradients
                        reference=grad_refs[n]
                        if gradients.keys()!=reference.keys():raise AssertionError('Missing parameter gradient')
                        worst=0.
                        for name,g in gradients.items():
                            torch.testing.assert_close(g,reference[name],atol=1e-3,rtol=2e-3)
                            worst=max(worst,float((g.double()-reference[name].double()).norm()/
                                reference[name].double().norm().clamp_min(1e-20)))
                        item['parameter_gradient_max_relative_l2']=worst
                        item['gradient_tensors']=len(gradients)
                    del values
                    model.zero_grad(set_to_none=True)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3) if task=='train' else None
                    def step():
                        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
                        result=outputs(model,data)
                        if optimizer is not None:
                            loss(result,n).backward();optimizer.step()
                        return result
                    for _ in range(4):step()
                    torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    samples=[]
                    repeats=(6 if task=='train' else (20 if n<=1024 else 10))
                    for _ in range(repeats):
                        torch.cuda.synchronize();start=time.perf_counter()
                        step()
                        torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
                    item.update(status='PASS',median_ms=statistics.median(samples),samples_ms=samples,
                        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                    if n==1024 and os.environ.get('PROFILE','1')=='1':
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
                            step();torch.cuda.synchronize()
                        (out/f'{mode}_{task}_{n}.profile.txt').write_text(profiler.key_averages().table(
                            sort_by='self_cuda_time_total',row_limit=50))
                    del data,optimizer
                except torch.cuda.OutOfMemoryError as error:
                    item.update(status='OOM',error=str(error))
                    if 'data' in locals():del data
                    if 'optimizer' in locals():del optimizer
                    if 'values' in locals():del values
                    model.zero_grad(set_to_none=True)
                    gc.collect();torch.cuda.empty_cache()
                except Exception as error:
                    item.update(status='FAIL',error=str(error),traceback=traceback.format_exc())
                    report['status']='FAIL';save();raise
                save()
                print(json.dumps({k:v for k,v in item.items() if k!='samples_ms'}),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    report['status']='COMPLETE' if all(r['status']=='PASS' for r in records) else 'PARTIAL_CAPACITY'
    save();print(json.dumps(dict(status=report['status'],output=str(path))),flush=True)


if __name__=='__main__':
    with forbid_cuda_graph(purpose='NequIP native OEQ performance') as policy:main(policy)
