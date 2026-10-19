"""Remote full-output / force-double-backward qualification of OEQ CHORUS."""
import copy
import json
import os
from pathlib import Path

import torch
from e3nn import o3
from nequip.nn._chorus import HermitianDensityResidual
from common import build,config,complete_graph,outputs,loss,strict_fp32,variant
from no_cuda_graph import forbid_cuda_graph


def compare(a,b,atol=3e-4,rtol=5e-4):
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    return dict(max_abs=float((a.double()-b.double()).abs().max()),
        relative_l2=float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-20)))


def density_case(lmax):
    torch.manual_seed(33+lmax)
    kwargs=dict(irreps_message=' + '.join(f'{3+l}x{l}{"e" if l%2==0 else "o"}' for l in range(lmax+1)),
        irreps_node='4x0e',edge_invariant_dim=4,num_elements=3,rank=4,scale_init=0.3)
    ref=HermitianDensityResidual(**kwargs).cuda()
    acc=HermitianDensityResidual(**kwargs,openequivariance_enabled=True).cuda()
    acc.load_state_dict(copy.deepcopy(ref.state_dict()),strict=True)
    n=9
    r=torch.randn(n,ref.irreps_charged.dim,device='cuda',requires_grad=True)
    i=torch.randn_like(r,requires_grad=True)
    attrs=torch.randn(n,3,device='cuda',requires_grad=True)
    a=acc.contract_charged(r,i,attrs);b=ref.contract_charged(r,i,attrs)
    parity=compare(a,b)
    probe=torch.randn_like(a)
    ar=torch.autograd.grad((a*probe).sum(),(r,i,attrs,*acc.output_weights.values(),acc.residual_scale),create_graph=True)
    br=torch.autograd.grad((b*probe).sum(),(r,i,attrs,*ref.output_weights.values(),ref.residual_scale),create_graph=True)
    for x,y in zip(ar,br):compare(x,y,5e-4,7e-4)
    ah=torch.autograd.grad(sum(x.square().sum() for x in ar),(r,i,attrs),retain_graph=True)
    bh=torch.autograd.grad(sum(x.square().sum() for x in br),(r,i,attrs))
    for x,y in zip(ah,bh):compare(x,y,3e-3,1e-3)
    theta=0.61
    rot_r=r*torch.cos(r.new_tensor(theta))-i*torch.sin(r.new_tensor(theta))
    rot_i=r*torch.sin(r.new_tensor(theta))+i*torch.cos(r.new_tensor(theta))
    compare(acc.contract_charged(rot_r,rot_i,attrs),a,2e-4,5e-4)
    return dict(lmax=lmax,parity=parity,soft_attributes=True,mixed_second=True,u1=True)


def whole_case(scope,lmax,layers):
    spec=config(scope,channels=8,lmax=lmax,layers=layers,rank=4)
    ref=build(spec,oeq=False)
    state=copy.deepcopy(ref.state_dict())
    data=complete_graph()
    expected=outputs(ref,data)
    loss(expected,7).backward()
    modes=[]
    for mode in os.environ.get('QUAL_VARIANTS','tp_only,legacy,optimized').split(','):
        acc=build(spec,state=state)
        variant(acc,mode)
        ids={name:id(p) for name,p in acc.named_parameters()}
        actual=outputs(acc,data)
        parity=[compare(a,b) for a,b in zip(actual,expected)]
        loss(actual,7).backward()
        errors=[]
        for (name,p),(other,q) in zip(acc.named_parameters(),ref.named_parameters()):
            if name!=other:raise AssertionError('Parameter names differ')
            if p.grad is None and q.grad is None:continue
            if p.grad is None or q.grad is None:raise AssertionError('Missing gradient: '+name)
            errors.append((name,compare(p.grad,q.grad,8e-4,1e-3)))
        if ids!={name:id(p) for name,p in acc.named_parameters()}:
            raise AssertionError('Parameter identity changed during execution')
        if tuple(acc.state_dict())!=tuple(state):raise AssertionError('Checkpoint schema changed')
        if mode=='optimized':
            rotation=o3.rand_matrix(device='cuda',dtype=torch.float32)
            rotated={key:value.detach().clone() for key,value in data.items()}
            from nequip.data import AtomicDataDict as D
            rotated[D.POSITIONS_KEY]=rotated[D.POSITIONS_KEY]@rotation.T
            rotated[D.CELL_KEY]=rotated[D.CELL_KEY]@rotation.T
            acc.eval()
            ee,ff,vv=outputs(acc,rotated)
            compare(ee,actual[0].detach())
            compare(ff,actual[1].detach()@rotation.T)
            compare(vv,rotation@actual[2].detach()@rotation.T,6e-4,8e-4)
        modes.append(dict(mode=mode,outputs=parity,parameter_gradients=dict(errors),
            parameters=sum(p.numel() for p in acc.parameters())))
        del acc
    return dict(scope=scope,lmax=lmax,layers=layers,variants=modes,full_efv=True)


def main(policy):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('A100 qualification requires sbatch')
    strict_fp32()
    rows=[]
    for l in (1,2,3):
        print(json.dumps(dict(stage='density',lmax=l)),flush=True)
        rows.append(density_case(l))
    cases=[]
    for scope in ('final','persistent'):
        for lmax,layers in ((2,2),(3,3)):
            print(json.dumps(dict(stage='full_efv_and_gradients',scope=scope,lmax=lmax,layers=layers)),flush=True)
            cases.append(whole_case(scope,lmax,layers))
    out=Path(os.environ['OUT']);out.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',density=rows,whole_models=cases,graph_execution_policy=policy,
        slurm_job=os.environ['SLURM_JOB_ID'],torch=torch.__version__)
    (out/'optimization_qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',density_cases=len(rows),whole_models=len(cases)),indent=2),flush=True)


if __name__=='__main__':
    with forbid_cuda_graph(purpose='NequIP native OEQ qualification') as policy:main(policy)
