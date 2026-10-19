"""Fixed-state loading and E/F/virial contract for graph-free inference."""
import os
from pathlib import Path

import torch


def load_architecture(case,scope,initialization):
    from e3nn import set_optimization_defaults
    set_optimization_defaults(jit_script_fx=False)
    from architecture_cases import chorus_spec
    from chorus.synthetic import build_model
    spec=chorus_spec(case,scope)
    saved=torch.load(initialization,map_location='cpu',weights_only=False)
    if saved['spec']!=spec:
        raise ValueError('Fixed architecture specification mismatch')
    u=[]
    for i in range(spec['interactions']):
        prefix=f'products.{i}.symmetric_contractions.symmetric_contractions.contractions.'
        tensors={key.removeprefix(prefix):value for key,value in saved['state'].items()
                 if key.startswith(prefix) and key.rsplit('.',1)[-1] in
                 {'U_matrix_1','U_matrix_2','U_matrix_3'}}
        if not tensors:
            raise ValueError('Missing checkpoint U tensors')
        u.append(tensors)
    model=build_model(channels=spec['channels'],lmax=spec['lmax'],num_interaction=spec['interactions'],
        route='baseline',product_backend='ictd-bridge-u',dtype=torch.float32,device=torch.device('cuda'),
        correlation=3,avg_num_neighbors=50.,ictd_fix_use_reduced_cg=True,
        ictd_fix_phase_mode='final-full-l-residual',ictd_fix_phase_amplitude='softplus',
        ictd_fix_phase_placement='pre-product-full-l',ictd_fix_phase_density_pairs='full-nonlinear',
        ictd_fix_phase_density_rank=16,ictd_fix_phase_scope=scope,ictd_fix_u_tensor_state=u,
        ictd_fix_freeze_conv_tp_weight=True).eval().requires_grad_(False)
    model.load_state_dict(saved['state'],strict=True)
    model.skip_input_validation=True;model.preserve_edge_order=True
    return model,spec


def compute(model,*graph):
    from chorus.hp import energy_forces_virial
    value=energy_forces_virial(model,*graph,fused_geometry=False,fused_forces=False)
    return value['energy'].sum(),value['forces'],value['atomic_energy'][:,None],value['virial']


def strict_fp32():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def identity():
    import subprocess
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    return dict(pid=os.getpid(),command=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),
        slurm_job=os.environ.get('SLURM_JOB_ID'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),gpu_inventory=uuid,
        optional_old_abi_wheels_disabled=os.environ.get('CHORUS_DISABLE_ABI_EXTENSIONS')=='1')
