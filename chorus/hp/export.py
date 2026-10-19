"""Prepared ICTC-O3-HP operators for frozen CHORUS AOTInductor export.

This is independent of the eager adapter and does not use CUDA Graph. The
result still needs FX tracing, export, compilation and numerical qualification.
"""
from typing import Tuple

import torch


def _dynamic_kernel(kernel, dimensions, stem):
    """Keep live N/E as runtime scalars rather than Triton specializations."""
    import re
    from mace_ictc.interfaces.deployment_triton import _module
    source=kernel.src
    for name in dimensions:
        source,count=re.subn(rf'\b{name}\s*:\s*tl\.constexpr',name,source)
        if count!=1:
            raise RuntimeError(f'Expected one constexpr declaration for {name} in {stem}')
    if not source.lstrip().startswith('@triton.jit'):
        source='@triton.jit\n'+source
    source=('import triton\nimport triton.language as tl\n'
            'import triton.language.extra.cuda.libdevice as libdevice\n\n'+source)
    return getattr(_module(source,stem),kernel.__name__)


class _PreparedMessage:
    def __init__(self, layer, device, *, csr_forward=False):
        from .runtime import _borrow_tp
        from types import SimpleNamespace
        from mace_ictc.interfaces.deployment_triton import prepare_message
        from mace_ictc.interfaces.deployment_coherent import prepare_coherent
        mirror=SimpleNamespace(tp=_borrow_tp(layer.tp),input_lmax=layer.input_lmax,
            phase_heads=layer.phase_heads,phase_angular_channels=layer.phase_angular_channels,
            lmax=layer.lmax,target_lmax=layer.target_lmax)
        self.phase=layer.phase_enabled
        if self.phase:
            self.evaluate=prepare_coherent(mirror,device,csr_forward=csr_forward)
        else:
            prepare_message(mirror,device,csr_forward=csr_forward)
            self.evaluate=mirror._deployment_message

    def radial_inputs(self, edge_feats, fc, edge_length=None):
        return fc(edge_feats)

    def __call__(self,blocks,harmonics,gates,src,dst,mask,cos,sin):
        if self.phase:
            return self.evaluate(blocks,harmonics,gates,src,dst,mask,cos,sin)
        return self.evaluate(blocks,harmonics,gates,src,dst,mask)


def _prepare_density(owner):
    import triton
    from torch.library import triton_op,wrap_triton
    from mace_ictc.interfaces.deployment_triton import _name
    from .density import DensityPlan,_quadratic,_adjoint_gather
    plan=DensityPlan(owner)
    args=plan._args()
    quadratic=_dynamic_kernel(_quadratic,('TOTAL',),'density_dynamic')
    adjoint=_dynamic_kernel(_adjoint_gather,('TOTAL',),'density_adjoint_dynamic')

    @triton_op(_name('chorus_density_backward'),mutates_args={})
    def backward_op(x:torch.Tensor,g:torch.Tensor)->torch.Tensor:
        x=x.contiguous();g=g.contiguous();dx=torch.empty_like(x)
        wrap_triton(adjoint)[(triton.cdiv(dx.numel(),16),)](
            x,g,plan.derivative_rows,plan.derivative_other,plan.derivative_weights,dx,
            dx.numel(),x.shape[1],plan.width,plan.derivative_k,16,enable_fp_fusion=False)
        return dx

    @triton_op(_name('chorus_density'),mutates_args={})
    def forward_op(x:torch.Tensor)->torch.Tensor:
        x=x.contiguous();out=x.new_empty(x.shape[0],plan.width)
        wrap_triton(quadratic)[(triton.cdiv(out.numel(),32),)](
            x,x,*args,out,out.numel(),x.shape[1],plan.width,plan.k,32,False,enable_fp_fusion=False)
        return out
    def setup(ctx,inputs,output):
        ctx.save_for_backward(inputs[0])
    def backward(ctx,g):
        return backward_op(ctx.saved_tensors[0],g)
    forward_op.register_autograd(backward,setup_context=setup)
    def evaluate(blocks):
        shape=blocks[0].shape[:-3]
        x=torch.cat([blocks[l].flatten(-2) for l in range(plan.lmax+1)],-1)
        value=forward_op(x.reshape(-1,2*plan.xhalf))
        result={};offset=0
        for l,size in enumerate(plan.sizes):
            result[l]=value[:,offset:offset+size].reshape(*shape,owner.input_multiplicities[l],2*l+1)
            offset+=size
        return result
    return evaluate


def _prepare_phase(layer):
    import triton
    from torch.library import triton_op,wrap_triton
    from mace_ictc.interfaces.deployment_triton import _name
    from .phase_coefficients import _phase,_MODES
    mode=_MODES[layer.phase_coefficient];has_raw=layer.phase_amplitude_head is not None
    phase=_dynamic_kernel(_phase,('N',),'phase_dynamic')

    @triton_op(_name('chorus_phase_backward'),mutates_args={})
    def backward_op(theta:torch.Tensor,raw:torch.Tensor,gc:torch.Tensor,gs:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        dt=torch.empty_like(theta);dr=torch.empty_like(raw)
        wrap_triton(phase)[(triton.cdiv(theta.numel(),256),)](
            theta,raw,gc.contiguous(),gs.contiguous(),theta,theta,dt,dr,
            theta.numel(),mode,has_raw,True,256,enable_fp_fusion=False)
        return dt,dr

    @triton_op(_name('chorus_phase'),mutates_args={})
    def forward_op(theta:torch.Tensor,raw:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        c=torch.empty_like(theta);s=torch.empty_like(theta)
        wrap_triton(phase)[(triton.cdiv(theta.numel(),256),)](
            theta,raw,theta,theta,c,s,theta,raw,theta.numel(),mode,has_raw,False,256,enable_fp_fusion=False)
        return c,s
    def setup(ctx,inputs,output):
        ctx.save_for_backward(*inputs)
    def backward(ctx,gc,gs):
        theta,raw=ctx.saved_tensors
        gc=torch.zeros_like(theta) if gc is None else gc
        gs=torch.zeros_like(theta) if gs is None else gs
        dt,dr=backward_op(theta,raw,gc,gs)
        return dt,dr if has_raw else None
    forward_op.register_autograd(backward,setup_context=setup)
    def evaluate(theta,raw,selected_mode):
        if selected_mode!=layer.phase_coefficient:
            raise ValueError('Phase coefficient changed after preparation')
        c,s=forward_op(theta.contiguous(),theta.new_empty(0) if raw is None else raw.contiguous())
        return c.unsqueeze(-1),s.unsqueeze(-1)
    return evaluate


def prepare_for_export(model, *, dynamic=True, message_schedule='atomic'):
    """Prepare frozen CUDA FP32 CHORUS operators without exporting a package.

    Keeps the checkpoint's actual U/CG tensors, weights and parameter identities.
    Reprepare after any weight/device/dtype change. Coordinate first derivatives
    are covered; this export API is not a training/double-backward backend.
    """
    from .runtime import _core,_clear,check_model,_borrow_contraction
    from chorus.models.pure_cartesian_ictd_fix import (
        ICTDBridgeUSymmetricContractionSO3,PhaseHermitianFullLResidual)
    from mace_ictc.interfaces.deployment_triton import prepare_harmonics,prepare_contraction
    from mace_ictc.models.fp32_policy import strict_fp32
    core=_core(model)
    if message_schedule not in ('atomic','csr'):
        raise ValueError('Prepared message_schedule must be atomic or csr')
    coverage=check_model(model,checkpoint_contraction=True)
    reasons=list(coverage['reasons'])
    if core is not model:
        reasons.append('Export requires the unwrapped backbone')
    if model.training or any(p.requires_grad for p in model.parameters()):
        reasons.append('Freeze every parameter and select eval before export preparation')
    if not hasattr(torch.library,'triton_op'):
        reasons.append('PyTorch torch.library.triton_op is required')
    if dynamic and any(isinstance(child,torch.jit.ScriptModule) for child in model.modules()):
        reasons.append('Dynamic export requires rebuilding with e3nn jit_script_fx=False')
    if getattr(core,'long_range_module',None) is not None or getattr(core,'dispersion',None) is not None:
        reasons.append('This prepared export covers the short-range backbone only')
    if any(getattr(layer,'phase_enabled',False) for layer in getattr(core,'interactions',())) and getattr(
            core,'ictd_fix_phase_density_pairs',None) not in ('full','full-nonlinear','full-nonlinear-readout'):
        reasons.append('Prepared charged messages currently cover full/full-nonlinear density pairs')
    if reasons:
        raise ValueError('CHORUS prepared export unavailable: '+'; '.join(dict.fromkeys(reasons)))
    strict_fp32()
    identities={name:(id(p),p.requires_grad) for name,p in model.named_parameters()}
    keys=tuple(model.state_dict())
    device=next(model.parameters()).device
    prepared=[]
    counts={'messages':0,'coherent_messages':0,'phase_coefficients':0,'hermitian_density':0,'symmetric_contraction':0}
    for layer in core.interactions:
        prepared.append((layer,'_hp_message',_PreparedMessage(layer,device,csr_forward=message_schedule=='csr')))
        counts['messages']+=1
        if layer.phase_enabled:
            prepared.append((layer,'_hp_phase_coefficients',_prepare_phase(layer)))
            counts['coherent_messages']+=1;counts['phase_coefficients']+=1
    for module in core.modules():
        if isinstance(module,ICTDBridgeUSymmetricContractionSO3):
            borrowed=_borrow_contraction(module,training=False,checkpoint_contraction=True)
            prepare_contraction(borrowed,device)
            prepared.append((module,'_hp_contraction',borrowed))
            counts['symmetric_contraction']+=1
        if isinstance(module,PhaseHermitianFullLResidual):
            prepared.append((module,'_hp_density',_prepare_density(module)))
            counts['hermitian_density']+=1
    # Borrow only the prepared harmonic callable; do not apply the mother's
    # neutral whole-model configurator to a charged/recurrent CHORUS backbone.
    from types import SimpleNamespace
    holder=SimpleNamespace(ictd_fix_edge_lmax=core.ictd_fix_edge_lmax)
    prepare_harmonics(holder,device)
    prepared.append((core,'_hp_harmonics',lambda n,lmax:holder._deployment_harmonics(n)))
    _clear(core)
    for module,name,value in prepared:
        object.__setattr__(module,name,value)
    if identities!={name:(id(p),p.requires_grad) for name,p in model.named_parameters()} or keys!=tuple(model.state_dict()):
        _clear(core)
        raise RuntimeError('Prepared export changed checkpoint schema or parameter identity')
    return {'selected':'prepared_triton','execution':'FX/AOTInductor without CUDA Graph',
            'dynamic':bool(dynamic),'counts':counts,'frozen_weights':True,'message_schedule':message_schedule,
            'arithmetic':'FP32 network/forces; TF32 off','external_tp_libraries':False}
