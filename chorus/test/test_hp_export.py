"""Remote-only numerical/FX qualification of prepared inference operators."""
import copy

import pytest
import torch

from chorus.synthetic import build_model,make_fixed_graph
from chorus.hp import prepare_for_export,configure_model,energy_forces_virial


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')


def _build(scope,channels,lmax,layers=2):
    from e3nn import set_optimization_defaults
    set_optimization_defaults(jit_script_fx=False)
    return build_model(channels=channels,lmax=lmax,num_interaction=layers,route='baseline',
        product_backend='ictd-bridge-u',dtype=torch.float32,device=torch.device('cuda'),
        correlation=2,ictd_fix_freeze_conv_tp_weight=True,
        ictd_fix_phase_mode='final-full-l-residual',ictd_fix_phase_amplitude='softplus',
        ictd_fix_phase_placement='pre-product-full-l',ictd_fix_phase_density_pairs='full-nonlinear',
        ictd_fix_phase_density_rank=4,ictd_fix_phase_scope=scope).eval().requires_grad_(False)


@cuda
@pytest.mark.parametrize('scope',['persistent','final'])
@pytest.mark.parametrize('channels,lmax,layers',[(8,1,2),(10,2,2),(32,3,2),(64,1,2),(64,3,3)])
def test_prepared_whole_efv_identity_and_live_inputs(scope,channels,lmax,layers):
    torch.manual_seed(20261017)
    model=_build(scope,channels,lmax,layers)
    model.skip_input_validation=True;model.preserve_edge_order=True
    state=copy.deepcopy(model.state_dict())
    ids={name:id(p) for name,p in model.named_parameters()}
    graphs=[make_fixed_graph(num_nodes=n,avg_degree=d,dtype=torch.float32,
        device=torch.device('cuda'),seed=91+i) for i,(n,d) in enumerate(((17,5),(23,4)))]
    def evaluate(graph):
        return energy_forces_virial(model,*graph,fused_geometry=False,fused_forces=False)
    refs=[evaluate(g) for g in graphs]
    info=prepare_for_export(model,message_schedule='csr')
    assert info['execution']=='FX/AOTInductor without CUDA Graph'
    assert ids=={name:id(p) for name,p in model.named_parameters()}
    assert tuple(state)==tuple(model.state_dict())
    for key in state:
        torch.testing.assert_close(state[key],model.state_dict()[key],atol=0,rtol=0)
    for graph,ref in zip(graphs,refs):
        got=evaluate(graph)
        for key in ref:
            torch.testing.assert_close(got[key],ref[key],atol=3e-5,rtol=5e-4,
                msg=lambda message:f'{key}\n{message}')
    configure_model(model,policy='reference')
    for key,value in evaluate(graphs[-1]).items():
        torch.testing.assert_close(value,refs[-1][key],atol=1e-6,rtol=5e-5)


@cuda
@pytest.mark.parametrize('scope',['persistent','final'])
def test_prepared_symbolic_force_graph(scope):
    from chorus.training.makefx_compile import trace_and_compile_force
    model=_build(scope,8,1)
    model.preserve_edge_order=True
    prepare_for_export(model,message_schedule='csr')
    graph=make_fixed_graph(num_nodes=17,avg_degree=5,dtype=torch.float32,
        device=torch.device('cuda'),seed=91)
    def evaluate(*inputs):
        value=energy_forces_virial(model,*inputs,fused_geometry=False,fused_forces=False)
        return value['energy'],value['forces'],value['atomic_energy'],value['virial']
    ref=evaluate(*graph)
    gm=trace_and_compile_force(model,graph,training=False,compute_fn=evaluate,do_compile=False)
    for got,expected in zip(gm(*graph),ref):
        torch.testing.assert_close(got,expected,atol=3e-5,rtol=5e-4)
    n=torch.export.Dim('atoms',min=2,max=131072);e=torch.export.Dim('edges',min=2,max=8388608)
    ep=torch.export.export(gm,graph,dynamic_shapes=({0:n},{0:n},{0:n},{0:e},{0:e},{0:e},None),strict=True)
    changed=make_fixed_graph(num_nodes=23,avg_degree=4,dtype=torch.float32,
        device=torch.device('cuda'),seed=92)
    for got,expected in zip(ep.module()(*changed),evaluate(*changed)):
        torch.testing.assert_close(got,expected,atol=3e-5,rtol=5e-4)


@cuda
def test_prepared_requires_frozen_eval():
    model=_build('persistent',8,1).train()
    with pytest.raises(ValueError,match='Freeze every parameter'):
        prepare_for_export(model)


@cuda
def test_prepared_coherent_matches_literal_streams():
    from chorus.hp.export import _PreparedMessage
    from chorus.models.pure_cartesian_ictd_fix import direction_harmonics_all
    model=_build('persistent',8,1)
    layer=next(layer for layer in model.interactions if layer.phase_enabled)
    e,n=31,11
    blocks={l:torch.randn(n,layer.channels,2*l+1,device='cuda',requires_grad=True)
            for l in range(layer.input_lmax+1)}
    direction=torch.randn(e,3,device='cuda',requires_grad=True)
    harm={l:value.unsqueeze(-2) for l,value in enumerate(direction_harmonics_all(direction,layer.lmax))}
    gates=torch.randn(e,len(layer.tp.paths)*layer.channels,device='cuda',requires_grad=True)
    count=layer.phase_heads*((layer.target_lmax+1) if layer.phase_angular_channels else 1)
    cos=torch.randn(e,count,1,device='cuda',requires_grad=True)
    sin=torch.randn(e,count,1,device='cuda',requires_grad=True)
    src=torch.arange(e,device='cuda')%n;dst=(src+1)%n
    edge=layer.tp({l:value[src] for l,value in blocks.items()},harm,gates)
    ref={}
    for l,value in edge.items():
        offset=l*layer.phase_heads if layer.phase_angular_channels else 0
        coeff=[c[:,offset:offset+layer.phase_heads].reshape(e,layer.phase_heads,1)
               .expand(-1,-1,layer.channels//layer.phase_heads).reshape(e,layer.channels)
               for c in (cos,sin)]
        # Paths are contiguous groups of C channels, not head-major across paths.
        coeff=[c.repeat(1,layer.tp.path_counts_by_l[l]).unsqueeze(-1) for c in coeff]
        streams=[value,value*coeff[0],value*coeff[1]]
        ref[l]=torch.stack([value.new_zeros(n,*value.shape[1:]).index_add(0,dst,v) for v in streams],1)
    prepared=_PreparedMessage(layer,torch.device('cuda'))
    got=prepared(blocks,harm,gates,src,dst,None,cos,sin)
    for l in ref:
        torch.testing.assert_close(got[l],ref[l],atol=2e-5,rtol=3e-4)
    probes={l:torch.randn_like(value) for l,value in ref.items()}
    inputs=[*blocks.values(),direction,gates,cos,sin]
    a=torch.autograd.grad(sum((got[l]*probes[l]).sum() for l in ref),inputs,retain_graph=True)
    b=torch.autograd.grad(sum((ref[l]*probes[l]).sum() for l in ref),inputs)
    for actual,expected in zip(a,b):
        torch.testing.assert_close(actual,expected,atol=8e-5,rtol=5e-4)
