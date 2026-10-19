import math
import copy

import pytest
import torch
from e3nn import o3

from nequip.model import model_from_config
from nequip.nn import (
    HermitianDensityResidual,
    ICTCInteractionBlock,
    InteractionBlock,
    PersistentChargedUpdate,
)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native dataflow requires CUDA')
@pytest.mark.parametrize("dtype",(torch.float32,torch.float64))
def test_native_irrep_linear_preserves_full_mixed_gradients(dtype):
    from nequip.nn._native_dataflow import linear,_plan
    torch.manual_seed(106)
    reference=o3.Linear('5x0e + 3x1o + 2x2e','4x0e + 7x1o + 3x2e').cuda().to(dtype)
    accelerated=copy.deepcopy(reference)
    assert _plan(accelerated) is not None
    value=torch.randn(2,5,reference.irreps_in.dim*2,device='cuda',dtype=dtype)[...,::2]
    value=value.detach().requires_grad_()
    expected=reference(value);actual=linear(accelerated,value)
    tol=3e-6 if dtype==torch.float32 else 2e-12
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
    probe=torch.randn_like(expected)
    a=torch.autograd.grad((actual*probe).sum(),(value,accelerated.weight),create_graph=True)
    b=torch.autograd.grad((expected*probe).sum(),(value,reference.weight),create_graph=True)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=tol*4,rtol=tol*4)
    ah=torch.autograd.grad(sum(x.square().sum() for x in a),(value,accelerated.weight))
    bh=torch.autograd.grad(sum(x.square().sum() for x in b),(value,reference.weight))
    for x,y in zip(ah,bh):torch.testing.assert_close(x,y,atol=tol*30,rtol=tol*30)
    assert tuple(reference.state_dict())==tuple(accelerated.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native dataflow requires CUDA')
def test_native_irrep_linear_preserves_intervention_hooks():
    from nequip.nn._native_dataflow import linear
    module=o3.Linear('3x0e + 3x1o','4x0e + 2x1o').cuda()
    seen=[]
    handle=module.register_forward_hook(lambda m,inputs,out:(seen.append(True) or out*1.7))
    value=torch.randn(7,module.irreps_in.dim,device='cuda')
    expected=module(value);actual=linear(module,value)
    torch.testing.assert_close(actual,expected)
    assert len(seen)==2
    handle.remove()


def test_factorized_phase_and_density_preserve_soft_attribute_mixed_gradients():
    torch.manual_seed(97)
    reference=HermitianDensityResidual(
        irreps_message='3x0e + 2x1o',irreps_node='4x0e + 3x1o',
        edge_invariant_dim=3,num_elements=2,rank=2,hidden_channels=8,
        avg_num_neighbors=4.,scale_init=0.2).double()
    accelerated=copy.deepcopy(reference)
    accelerated.factorized_phase=accelerated.factorized_density=True
    inputs=[torch.randn(9,reference.irreps_message.dim,dtype=torch.float64,requires_grad=True),
        torch.randn(5,reference.irreps_node.dim,dtype=torch.float64,requires_grad=True),
        torch.randn(5,2,dtype=torch.float64,requires_grad=True),
        torch.randn(9,3,dtype=torch.float64,requires_grad=True)]
    src=torch.arange(9)%5;dst=(src+2)%5
    def run(module):return module(inputs[0],inputs[1],inputs[2],inputs[3],src,dst,5)
    expected=run(reference);actual=run(accelerated)
    torch.testing.assert_close(actual,expected,atol=2e-11,rtol=2e-11)
    a=torch.autograd.grad(actual.square().sum(),(*inputs,*accelerated.parameters()),create_graph=True)
    b=torch.autograd.grad(expected.square().sum(),(*inputs,*reference.parameters()),create_graph=True)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=2e-10,rtol=2e-10)
    ah=torch.autograd.grad(sum(x.square().sum() for x in a),inputs)
    bh=torch.autograd.grad(sum(x.square().sum() for x in b),inputs)
    for x,y in zip(ah,bh):torch.testing.assert_close(x,y,atol=2e-9,rtol=2e-9)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native OEQ requires CUDA')
def test_openequivariance_hermitian_uses_scalar_rank_batch_weights():
    pytest.importorskip('openequivariance')
    torch.manual_seed(98)
    kwargs=dict(irreps_message='3x0e + 3x1o + 3x2e',irreps_node='3x0e',
                edge_invariant_dim=3,num_elements=2,rank=3,scale_init=0.2)
    reference=HermitianDensityResidual(**kwargs).cuda()
    accelerated=HermitianDensityResidual(**kwargs,openequivariance_enabled=True).cuda()
    accelerated.load_state_dict(reference.state_dict(),strict=True)
    assert tuple(reference.state_dict())==tuple(accelerated.state_dict())
    assert accelerated.openequivariance_density.constant_weights.numel()==2*len(reference.paths)
    real=torch.randn(5,reference.irreps_charged.dim,device='cuda')
    imag=torch.randn_like(real)
    expected=reference.density_blocks(real,imag)
    actual=accelerated.density_blocks(real,imag)
    for l in expected:torch.testing.assert_close(actual[l],expected[l],atol=2e-5,rtol=5e-4)


def test_factorized_phase_preserves_diagnostic_and_intervention_hooks():
    torch.manual_seed(99)
    reference=HermitianDensityResidual(
        irreps_message='2x0e + 2x1o',irreps_node='2x0e + 2x1o',
        edge_invariant_dim=3,num_elements=1,rank=2,scale_init=0.2).double()
    accelerated=copy.deepcopy(reference)
    accelerated.factorized_phase=True
    counts=[0,0]
    def hook(index):
        def intervene(module,inputs,output):
            counts[index]+=1
            return torch.stack((output[:,0],output[:,1]*0.0),-1)
        return intervene
    handles=[reference.phase_network.register_forward_hook(hook(0)),
             accelerated.phase_network.register_forward_hook(hook(1))]
    messages=torch.randn(8,reference.irreps_message.dim,dtype=torch.float64)
    nodes=torch.randn(5,reference.irreps_node.dim,dtype=torch.float64)
    invariants=torch.randn(8,3,dtype=torch.float64)
    src=torch.arange(8)%5;dst=(src+2)%5
    a=accelerated.charged_edge_messages(messages,nodes,invariants,src,dst)
    b=reference.charged_edge_messages(messages,nodes,invariants,src,dst)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=2e-12,rtol=2e-12)
    assert counts==[1,1]
    for handle in handles:handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native OEQ requires CUDA')
def test_openequivariance_neutral_conv_live_graph_and_mixed_gradients():
    pytest.importorskip('openequivariance')
    from nequip.nn._openequivariance import OpenEquivarianceTensorProduct
    torch.manual_seed(100)
    ir=o3.Irreps('3x0e + 3x1o');sh=o3.Irreps('0e + 1o')
    module=OpenEquivarianceTensorProduct(ir,sh,ir,
        [(0,0,0,'uvu',True),(0,1,1,'uvu',True),
         (1,0,1,'uvu',True),(1,1,0,'uvu',True)],fused_convolution=True).cuda()
    for n,pairs in ((5,[(0,1),(1,2),(0,3)]),(7,[(0,2),(1,4),(2,4),(4,5)])):
        dst=torch.tensor([a for a,b in pairs]+[b for a,b in pairs],device='cuda')
        src=torch.tensor([b for a,b in pairs]+[a for a,b in pairs],device='cuda')
        x=torch.randn(n,ir.dim,device='cuda',requires_grad=True)
        y=torch.randn(src.numel(),sh.dim,device='cuda',requires_grad=True)
        w=torch.randn(src.numel(),module.tensor_product.weight_numel,device='cuda',requires_grad=True)
        ref=x.new_zeros(n,ir.dim).index_add(0,dst,module(x[src],y,w))
        actual=module.aggregate(x,y,w,dst,src)
        torch.testing.assert_close(actual,ref,atol=2e-5,rtol=5e-4)
        a=torch.autograd.grad(actual.square().sum(),(x,y,w),create_graph=True)
        b=torch.autograd.grad(ref.square().sum(),(x,y,w),create_graph=True)
        for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=5e-4,rtol=5e-4)
        ah=torch.autograd.grad(sum(t.square().sum() for t in a),(x,y,w))
        bh=torch.autograd.grad(sum(t.square().sum() for t in b),(x,y,w))
        for u,v in zip(ah,bh):torch.testing.assert_close(u,v,atol=5e-3,rtol=1e-3)


NATURAL_CONFIG = {
    "model_builders": ["EnergyModel"],
    "num_types": 2,
    "chemical_symbol_to_type": {"H": 0, "C": 1},
    "chemical_embedding_irreps_out": "4x0e",
    "irreps_edge_sh": "0e + 1o + 2e",
    "feature_irreps_hidden": "4x0e + 4x1o + 4x2e",
    "conv_to_output_hidden_irreps_out": "2x0e",
    "r_max": 4.0,
    "num_layers": 2,
    "num_basis": 4,
    "PolynomialCutoff_p": 6,
    "nonlinearity_type": "gate",
    "avg_num_neighbors": 4.0,
    "use_sc": True,
    "invariant_layers": 1,
    "invariant_neurons": 8,
}


def test_hermitian_contraction_is_u1_invariant():
    torch.manual_seed(11)
    module = HermitianDensityResidual(
        irreps_message="3x0e + 2x1o + 1x2e",
        irreps_node="4x0e + 4x1o + 4x2e",
        edge_invariant_dim=4,
        num_elements=2,
        rank=2,
        scale_init=1.0,
    ).double()
    real = torch.randn(5, module.irreps_charged.dim, dtype=torch.float64)
    imag = torch.randn_like(real)
    angle = 0.731
    rotated_real = math.cos(angle) * real - math.sin(angle) * imag
    rotated_imag = math.sin(angle) * real + math.cos(angle) * imag
    torch.testing.assert_close(
        module.contract_charged(real, imag),
        module.contract_charged(rotated_real, rotated_imag),
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_full_aligned_and_diagonal_are_parameter_matched():
    kwargs = dict(
        irreps_message="3x0e + 2x1o + 1x2e",
        irreps_node="4x0e + 4x1o + 4x2e",
        edge_invariant_dim=4,
        num_elements=2,
        rank=2,
        scale_init=1.0,
    )
    full = HermitianDensityResidual(**kwargs, density_pairs="full")
    aligned = HermitianDensityResidual(**kwargs, density_pairs="aligned")
    diagonal = HermitianDensityResidual(**kwargs, density_pairs="diagonal")
    assert tuple(full.state_dict()) == tuple(aligned.state_dict())
    assert tuple(full.state_dict()) == tuple(diagonal.state_dict())
    assert sum(p.numel() for p in full.parameters()) == sum(
        p.numel() for p in aligned.parameters()
    ) == sum(p.numel() for p in diagonal.parameters())


def test_phase_initialisation_breaks_the_aligned_stationary_point():
    torch.manual_seed(31)
    module = HermitianDensityResidual(
        irreps_message="2x0e + 2x1o",
        irreps_node="2x0e + 2x1o",
        edge_invariant_dim=3,
        num_elements=1,
        rank=2,
        scale_init=1.0,
        density_pairs="full",
    ).double()
    output = module.phase_network[-1]
    torch.testing.assert_close(output.weight[0], torch.zeros_like(output.weight[0]))
    torch.testing.assert_close(output.bias, torch.zeros_like(output.bias))
    assert output.weight[1].abs().sum() > 0.0

    edge_messages = torch.randn(4, module.irreps_message.dim, dtype=torch.float64)
    node_features = torch.randn(3, module.irreps_node.dim, dtype=torch.float64)
    edge_invariants = torch.randn(4, 3, dtype=torch.float64)
    edge_src = torch.tensor([1, 2, 0, 2])
    edge_dst = torch.tensor([0, 0, 1, 1])
    node_attrs = torch.ones(3, 1, dtype=torch.float64)
    kwargs = dict(
        edge_messages=edge_messages,
        node_features=node_features,
        node_attrs=node_attrs,
        edge_invariants=edge_invariants,
        edge_src=edge_src,
        edge_dst=edge_dst,
        num_nodes=3,
    )
    full = module(**kwargs)
    module.density_pairs = "aligned"
    aligned = module(**kwargs)
    assert not torch.allclose(full, aligned, atol=1.0e-12, rtol=1.0e-12)


def test_diagonal_removes_cross_edge_terms_and_independent_phases():
    torch.manual_seed(23)
    module = HermitianDensityResidual(
        irreps_message="2x0e + 2x1o",
        irreps_node="2x0e + 2x1o",
        edge_invariant_dim=3,
        num_elements=1,
        rank=2,
        scale_init=1.0,
        density_pairs="diagonal",
    ).double()
    edge_messages = torch.randn(5, module.irreps_message.dim, dtype=torch.float64)
    node_features = torch.randn(3, module.irreps_node.dim, dtype=torch.float64)
    edge_invariants = torch.randn(5, 3, dtype=torch.float64)
    edge_src = torch.tensor([1, 2, 0, 2, 0])
    edge_dst = torch.tensor([0, 0, 1, 1, 2])
    node_attrs = torch.ones(3, 1, dtype=torch.float64)
    charged_real, charged_imag = module.charged_edge_messages(
        edge_messages,
        node_features,
        edge_invariants,
        edge_src,
        edge_dst,
    )
    angles = torch.randn(5, 1, dtype=torch.float64)
    rotated_real = torch.cos(angles) * charged_real - torch.sin(angles) * charged_imag
    rotated_imag = torch.sin(angles) * charged_real + torch.cos(angles) * charged_imag
    original_density = module.density_blocks(charged_real, charged_imag)
    rotated_density = module.density_blocks(rotated_real, rotated_imag)
    for out_l in original_density:
        torch.testing.assert_close(
            original_density[out_l], rotated_density[out_l], atol=2e-11, rtol=2e-11
        )


def test_aligned_ignores_phase_head_but_retains_amplitude_head():
    torch.manual_seed(29)
    module = HermitianDensityResidual(
        irreps_message="2x0e + 2x1o",
        irreps_node="2x0e + 2x1o",
        edge_invariant_dim=3,
        num_elements=1,
        rank=2,
        scale_init=1.0,
        density_pairs="aligned",
    ).double()
    edge_messages = torch.randn(4, module.irreps_message.dim, dtype=torch.float64)
    node_features = torch.randn(3, module.irreps_node.dim, dtype=torch.float64)
    edge_invariants = torch.randn(4, 3, dtype=torch.float64)
    edge_src = torch.tensor([1, 2, 0, 2])
    edge_dst = torch.tensor([0, 0, 1, 1])
    before = module.charged_edge_messages(
        edge_messages, node_features, edge_invariants, edge_src, edge_dst
    )
    with torch.no_grad():
        module.phase_network[-1].weight[1].normal_(mean=3.0, std=2.0)
        module.phase_network[-1].bias[1].fill_(4.0)
    after = module.charged_edge_messages(
        edge_messages, node_features, edge_invariants, edge_src, edge_dst
    )
    torch.testing.assert_close(before[0], after[0], atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(before[1], after[1], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize(
    ("backend", "expected_type"),
    (("e3nn", InteractionBlock), ("ictc", ICTCInteractionBlock)),
)
@pytest.mark.parametrize("chorus_enabled", (False, True))
def test_four_model_variants_build(backend, expected_type, chorus_enabled):
    config = dict(NATURAL_CONFIG)
    config.update(
        interaction_backend=backend,
        chorus_enabled=chorus_enabled,
        chorus_rank=2,
        chorus_hidden_channels=8,
        chorus_scale_init=0.0,
    )
    model = model_from_config(config=config, initialize=True)
    first = model.model.layer0_convnet.conv
    last = model.model.layer1_convnet.conv
    assert isinstance(first, expected_type)
    assert first.chorus is None
    assert (last.chorus is not None) is chorus_enabled


@pytest.mark.parametrize(
    ("backend", "expected_type"),
    (("e3nn", InteractionBlock), ("ictc", ICTCInteractionBlock)),
)
def test_chorus_all_scope_builds_every_layer(backend, expected_type):
    config = dict(NATURAL_CONFIG)
    config.update(
        interaction_backend=backend,
        chorus_enabled=True,
        chorus_scope="all",
        chorus_rank=2,
        chorus_hidden_channels=8,
        chorus_scale_init=0.0,
    )
    model = model_from_config(config=config, initialize=True)
    for layer_index in range(config["num_layers"]):
        conv = getattr(model.model, f"layer{layer_index}_convnet").conv
        assert isinstance(conv, expected_type)
        assert conv.chorus is not None


@pytest.mark.parametrize(
    ("backend", "expected_type"),
    (("e3nn", InteractionBlock), ("ictc", ICTCInteractionBlock)),
)
def test_chorus_persistent_scope_builds_charged_stream(backend, expected_type):
    config = dict(NATURAL_CONFIG)
    config.update(
        interaction_backend=backend,
        chorus_enabled=True,
        chorus_scope="persistent",
        chorus_rank=2,
        chorus_hidden_channels=8,
        chorus_scale_init=0.0,
    )
    model = model_from_config(config=config, initialize=True)
    for layer_index in range(config["num_layers"]):
        conv = getattr(model.model, f"layer{layer_index}_convnet").conv
        assert isinstance(conv, expected_type)
        assert conv.chorus is not None
        assert (conv.chorus_update is not None) is (layer_index > 0)
        assert conv.chorus_persistent
        assert conv.chorus_clear_state is (
            layer_index == config["num_layers"] - 1
        )


def test_persistent_charged_update_is_o3_u1_equivariant():
    torch.manual_seed(17)
    update = PersistentChargedUpdate("2x0e + 2x1o + 2x2e").double()
    previous_real = torch.randn(4, 18, dtype=torch.float64)
    previous_imag = torch.randn_like(previous_real)
    incoming_real = torch.randn_like(previous_real)
    incoming_imag = torch.randn_like(previous_real)
    angle = 0.417
    rotation = o3.rand_matrix(dtype=torch.float64)
    representation = update.irreps_charged.D_from_matrix(rotation)

    def transform(real, imag):
        rotated_real = math.cos(angle) * real - math.sin(angle) * imag
        rotated_imag = math.sin(angle) * real + math.cos(angle) * imag
        return rotated_real @ representation.T, rotated_imag @ representation.T

    expected_real, expected_imag = update(
        previous_real, previous_imag, incoming_real, incoming_imag
    )
    transformed = update(
        *transform(previous_real, previous_imag),
        *transform(incoming_real, incoming_imag),
    )
    expected_real_rotated = (
        math.cos(angle) * expected_real - math.sin(angle) * expected_imag
    ) @ representation.T
    expected_imag_rotated = (
        math.sin(angle) * expected_real + math.cos(angle) * expected_imag
    ) @ representation.T
    torch.testing.assert_close(
        transformed[0], expected_real_rotated,
        atol=2.0e-6, rtol=2.0e-6,
    )
    torch.testing.assert_close(
        transformed[1], expected_imag_rotated,
        atol=2.0e-6, rtol=2.0e-6,
    )


def test_channelwise_density_is_o3_equivariant():
    torch.manual_seed(12)
    module = HermitianDensityResidual(
        irreps_message="3x0e + 2x1o + 1x2e",
        irreps_node="4x0e + 4x1o + 4x2e",
        edge_invariant_dim=4,
        num_elements=2,
        rank=2,
        scale_init=1.0,
    ).double()
    real = torch.randn(5, module.irreps_charged.dim, dtype=torch.float64)
    imag = torch.randn_like(real)
    node_attrs = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 1, 0, 1]), num_classes=2
    ).double()
    rotation = o3.rand_matrix(dtype=torch.float64)
    charged_d = module.irreps_charged.D_from_matrix(rotation)
    message_d = module.irreps_message.D_from_matrix(rotation)
    expected = module.contract_charged(real, imag, node_attrs) @ message_d.T
    actual = module.contract_charged(
        real @ charged_d.T,
        imag @ charged_d.T,
        node_attrs,
    )
    # e3nn's float32-default Wigner-D construction retains approximately
    # single-precision rotation constants even when this probe module is cast
    # to double, so the appropriate equivariance tolerance is around 1e-6.
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_channelwise_chorus_backward_is_finite():
    torch.manual_seed(13)
    module = HermitianDensityResidual(
        irreps_message="6x0e + 4x1o + 3x2e",
        irreps_node="4x0e + 4x1o + 4x2e",
        edge_invariant_dim=4,
        num_elements=2,
        rank=2,
        hidden_channels=8,
        avg_num_neighbors=3.0,
        scale_init=0.05,
    )
    edge_messages = torch.randn(
        9, module.irreps_message.dim, requires_grad=True
    )
    node_features = torch.randn(4, module.irreps_node.dim, requires_grad=True)
    edge_invariants = torch.randn(9, 4, requires_grad=True)
    edge_src = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3, 0])
    edge_dst = torch.tensor([1, 2, 3, 0, 2, 0, 3, 1, 3])
    node_attrs = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 0, 1]), num_classes=2
    ).to(dtype=edge_messages.dtype)
    output = module(
        edge_messages=edge_messages,
        node_features=node_features,
        node_attrs=node_attrs,
        edge_invariants=edge_invariants,
        edge_src=edge_src,
        edge_dst=edge_dst,
        num_nodes=4,
    )
    output.square().mean().backward()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    phase_gradient = module.phase_network[-1].weight.grad[1]
    assert phase_gradient.abs().sum() > 0.0
    assert torch.isfinite(edge_messages.grad).all()
    assert torch.isfinite(node_features.grad).all()
    assert torch.isfinite(edge_invariants.grad).all()


def test_density_has_no_cross_rank_terms():
    torch.manual_seed(14)
    module = HermitianDensityResidual(
        irreps_message="3x0e + 2x1o + 1x2e",
        irreps_node="4x0e + 4x1o + 4x2e",
        edge_invariant_dim=4,
        num_elements=2,
        rank=2,
        scale_init=1.0,
    ).double()
    real = torch.randn(3, module.irreps_charged.dim, dtype=torch.float64)
    imag = torch.randn_like(real)
    rank_zero_real = torch.zeros_like(real)
    rank_zero_imag = torch.zeros_like(imag)
    rank_one_real = torch.zeros_like(real)
    rank_one_imag = torch.zeros_like(imag)
    for (_, ir), component_slice in zip(
        module.irreps_charged, module.irreps_charged.slices()
    ):
        source_real = real[:, component_slice].reshape(3, 2, ir.dim)
        source_imag = imag[:, component_slice].reshape(3, 2, ir.dim)
        rank_zero_real[:, component_slice] = torch.stack(
            (source_real[:, 0], torch.zeros_like(source_real[:, 1])), dim=1
        ).reshape(3, -1)
        rank_zero_imag[:, component_slice] = torch.stack(
            (source_imag[:, 0], torch.zeros_like(source_imag[:, 1])), dim=1
        ).reshape(3, -1)
        rank_one_real[:, component_slice] = torch.stack(
            (torch.zeros_like(source_real[:, 0]), source_real[:, 1]), dim=1
        ).reshape(3, -1)
        rank_one_imag[:, component_slice] = torch.stack(
            (torch.zeros_like(source_imag[:, 0]), source_imag[:, 1]), dim=1
        ).reshape(3, -1)
    full = module.density_blocks(real, imag)
    rank_zero = module.density_blocks(rank_zero_real, rank_zero_imag)
    rank_one = module.density_blocks(rank_one_real, rank_one_imag)
    for out_l in full:
        torch.testing.assert_close(
            full[out_l], rank_zero[out_l] + rank_one[out_l]
        )


def test_realistic_channelwise_parameter_budget():
    base_config = dict(NATURAL_CONFIG)
    base_config.update(
        chemical_embedding_irreps_out="32x0e",
        feature_irreps_hidden="32x0e + 32x1o + 32x2e",
        conv_to_output_hidden_irreps_out="16x0e",
        num_layers=3,
        invariant_layers=2,
        invariant_neurons=64,
    )
    baseline = model_from_config(config=base_config, initialize=True)
    chorus_config = dict(base_config)
    chorus_config.update(
        chorus_enabled=True,
        chorus_rank=8,
        chorus_hidden_channels=32,
        chorus_scale_init=0.0,
    )
    chorus = model_from_config(config=chorus_config, initialize=True)
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    chorus_parameters = sum(p.numel() for p in chorus.parameters())
    assert chorus_parameters - baseline_parameters < 50_000
    assert chorus_parameters < 1.3 * baseline_parameters
    final_chorus = chorus.model.layer2_convnet.conv.chorus
    assert final_chorus is not None
    assert not any(
        isinstance(module, o3.FullyConnectedTensorProduct)
        for module in final_chorus.modules()
    )


def test_ictc_filters_unsupported_unnatural_parity_features():
    config = dict(NATURAL_CONFIG)
    config.update(
        interaction_backend="ictc",
        feature_irreps_hidden="4x0e + 4x1e",
    )
    model = model_from_config(config=config, initialize=True)
    first = model.model.layer0_convnet.conv
    for _, ir in first.irreps_out["node_features"]:
        assert ir.p == (1 if ir.l % 2 == 0 else -1)
