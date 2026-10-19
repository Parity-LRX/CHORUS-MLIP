"""Fixed-state NequIP OEQ qualification and native benchmark helpers."""
import copy
import importlib.util
import os
from pathlib import Path
from types import MethodType

import torch
from nequip.data import AtomicDataDict as D
from nequip.model import model_from_config
from nequip.nn._chorus import HermitianDensityResidual
from nequip.nn._openequivariance import OpenEquivarianceTensorProduct


def strict_fp32():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False


def config(scope,channels=64,lmax=2,layers=2,rank=16,oeq=True):
    return dict(model_builders=['EnergyModel','StressForceOutput'],num_types=4,
        chemical_symbol_to_type={'H':0,'C':1,'N':2,'O':3},
        chemical_embedding_irreps_out=f'{channels}x0e',
        irreps_edge_sh=' + '.join(f'{l}{"e" if l%2==0 else "o"}' for l in range(lmax+1)),
        feature_irreps_hidden=' + '.join(f'{channels}x{l}{"e" if l%2==0 else "o"}' for l in range(lmax+1)),
        conv_to_output_hidden_irreps_out=f'{max(1,channels//2)}x0e',r_max=4.0,
        num_layers=layers,num_basis=8,PolynomialCutoff_p=6,nonlinearity_type='gate',
        avg_num_neighbors=50.,use_sc=True,invariant_layers=2,invariant_neurons=64,
        interaction_backend='e3nn',chorus_enabled=True,chorus_scope=scope,
        chorus_rank=rank,chorus_hidden_channels=32,chorus_scale_init=0.05,
        chorus_density_pairs='full',model_dtype='float32',default_dtype='float32',
        allow_tf32=False,openequivariance_enabled=oeq)


def build(spec,oeq=True,state=None):
    torch.manual_seed(20261018)
    spec=copy.deepcopy(spec);spec['openequivariance_enabled']=oeq
    model=model_from_config(spec,initialize=True).cuda()
    if state is not None:model.load_state_dict(state,strict=True)
    return model


def clone_graph(data):
    return {key:value.detach().clone() for key,value in data.items()}


def outputs(model,data):
    value=model(clone_graph(data))
    return value[D.TOTAL_ENERGY_KEY],value[D.FORCE_KEY],value[D.VIRIAL_KEY]


def complete_graph(n=7):
    torch.manual_seed(911)
    i,j=torch.where(~torch.eye(n,device='cuda',dtype=torch.bool))
    return {D.POSITIONS_KEY:torch.randn(n,3,device='cuda'),
        D.EDGE_INDEX_KEY:torch.stack((i,j)),D.EDGE_CELL_SHIFT_KEY:torch.zeros(i.numel(),3,device='cuda'),
        D.CELL_KEY:torch.eye(3,device='cuda')*15,D.PBC_KEY:torch.zeros(3,dtype=torch.bool,device='cuda'),
        D.ATOM_TYPE_KEY:(torch.arange(n,device='cuda')%4)[:,None],
        D.BATCH_KEY:torch.zeros(n,dtype=torch.long,device='cuda'),
        D.BATCH_PTR_KEY:torch.tensor([0,n],device='cuda')}


def periodic_graph(n,cutoff=4.,degree=50,seed=20261018):
    import numpy as np
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    if degree not in (32,50):raise ValueError('Expected degree 32 or 50')
    dims=[1,1,1];remaining=n;factor=2
    while remaining>1:
        while remaining%factor:factor+=1
        dims[dims.index(min(dims))]*=factor;remaining//=factor
    spacing=(np.full(3,cutoff/2.05) if degree==32 else
             cutoff/np.sqrt(7.56)*np.asarray([1.,1.2,1.4]))
    grid=np.stack(np.meshgrid(*(np.arange(d) for d in dims),indexing='ij'),axis=-1).reshape(-1,3)
    positions=spacing*(grid+0.001*np.random.default_rng(seed+n).normal(size=(n,3)))
    atoms=Atoms(numbers=np.asarray([1,6,7,8])[np.arange(n)%4],positions=positions,
                cell=np.diag(np.asarray(dims)*spacing),pbc=True)
    center,neighbor,shifts=neighbor_list('ijS',atoms,cutoff,self_interaction=False)
    counts=np.bincount(center,minlength=n)
    if not (counts.min()==counts.max()==degree):raise ValueError('Cutoff degree mismatch')
    data={D.POSITIONS_KEY:torch.tensor(positions,device='cuda',dtype=torch.float32),
        D.EDGE_INDEX_KEY:torch.tensor(np.stack((center,neighbor)),device='cuda'),
        D.EDGE_CELL_SHIFT_KEY:torch.tensor(shifts,device='cuda',dtype=torch.float32),
        D.CELL_KEY:torch.tensor(atoms.cell.array,device='cuda',dtype=torch.float32),
        D.PBC_KEY:torch.ones(3,dtype=torch.bool,device='cuda'),
        D.ATOM_TYPE_KEY:(torch.arange(n,device='cuda')%4)[:,None],
        D.BATCH_KEY:torch.zeros(n,dtype=torch.long,device='cuda'),
        D.BATCH_PTR_KEY:torch.tensor([0,n],device='cuda')}
    return data,dict(atoms=n,edges=int(center.size),neighbors=degree)


def loss(values,n):
    energy,force,virial=values
    return (energy/n).square().mean()+force.square().mean()+0.01*(virial/n).square().mean()


def legacy_class():
    root=Path(os.environ.get('ROOT','/public/home/sps-xia/rxlin/nequip_chorus_oeq_native_20261018'))
    path=Path(__file__).parent/'reference/openequivariance_dense_rank.py'
    if not path.is_file():path=root/'legacy/openequivariance_dense_rank.py'
    spec=importlib.util.spec_from_file_location('original_dense_rank_oeq',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.OpenEquivarianceHermitianTensorProduct


def _legacy_contract(owner,density,node_attrs=None):
    blocks=[]
    for out_l,block in density.items():
        bank=owner.output_weights[str(out_l)]
        effective=(bank[0].expand(block.shape[0],-1,-1) if node_attrs is None
                   else bank.index_select(0,node_attrs.argmax(-1)))
        value=torch.bmm(effective,block)*owner.residual_scale[out_l]
        blocks.append(value.flatten(-2))
    return torch.cat(blocks,-1)


def variant(model,mode):
    """Benchmark-only controls, preserving every learned Parameter object."""
    before={name:id(p) for name,p in model.named_parameters()}
    keys=tuple(model.state_dict())
    count=0
    for module in model.modules():
        if isinstance(module,OpenEquivarianceTensorProduct) and mode in ('tp_only','legacy'):
            module.use_fused_convolution=False
            module.convolution=None
        if not isinstance(module,HermitianDensityResidual):continue
        count+=1
        if mode=='tp_only':
            module.openequivariance_density=None
            module.factorized_phase=module.factorized_density=False
        elif mode=='legacy':
            module.openequivariance_density=legacy_class()(module.irreps_charged,module.paths).cuda()
            module.factorized_phase=module.factorized_density=False
            module.contract_density=MethodType(_legacy_contract,module)
        elif mode!='optimized':
            raise ValueError('Unknown OEQ native variant')
    if before!={name:id(p) for name,p in model.named_parameters()} or keys!=tuple(model.state_dict()):
        raise RuntimeError('Execution variant changed parameters or checkpoint schema')
    return count
