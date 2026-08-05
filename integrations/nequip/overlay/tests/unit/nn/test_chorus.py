import math

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
