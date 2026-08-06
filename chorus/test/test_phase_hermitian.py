"""Tests for the U1-CHORUS real-doublet/Hermitian residual operator."""

from __future__ import annotations

import pytest
import torch

import chorus.models.pure_cartesian_ictd_fix as pure_cartesian_ictd_fix
from chorus.cli.train import build_baseline_model
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from chorus.models.pure_cartesian_ictd_fix import (
    PhaseHermitianFullLResidual,
    PhaseHermitianScalarResidual,
    PersistentChargedUpdate,
)
from chorus.synthetic import (
    build_model,
    compute_energy_forces,
    make_fixed_graph,
    random_rotation,
)
from chorus.training.train_loop import ForceTrainer
from chorus.training.makefx_compile import trace_and_compile_force
from chorus.utils.config import ModelConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build(
    *,
    phase_mode: str = "none",
    phase_amplitude: str = "unit",
    phase_coefficient: str = "polar",
    phase_context: str = "content",
    phase_placement: str = "post-product",
    phase_density_rank: int = 8,
    phase_density_pairs: str = "full",
    phase_coherence_init: float = 0.1,
    phase_normalization: str = "avg-neighbors",
    phase_scope: str = "final",
    phase_heads: int = 1,
    nonlinear_layer_readouts: bool = False,
    final_layer_readout_only: bool = False,
    element_energy_correction: bool = False,
    scalar_ffn: bool = False,
):
    return build_model(
        channels=8,
        lmax=1,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-pure-u",
        dtype=torch.float64,
        device=DEVICE,
        correlation=2,
        attn_heads=0,
        ictd_fix_phase_mode=phase_mode,
        ictd_fix_phase_hidden_channels=12,
        ictd_fix_phase_residual_scale_init=0.05,
        ictd_fix_phase_amplitude=phase_amplitude,
        ictd_fix_phase_coefficient=phase_coefficient,
        ictd_fix_phase_context=phase_context,
        ictd_fix_phase_placement=phase_placement,
        ictd_fix_phase_density_rank=phase_density_rank,
        ictd_fix_phase_density_pairs=phase_density_pairs,
        ictd_fix_phase_coherence_init=phase_coherence_init,
        ictd_fix_phase_normalization=phase_normalization,
        ictd_fix_phase_scope=phase_scope,
        ictd_fix_phase_heads=phase_heads,
        ictd_fix_nonlinear_layer_readouts=nonlinear_layer_readouts,
        ictd_fix_final_layer_readout_only=final_layer_readout_only,
        ictd_fix_element_energy_correction=element_energy_correction,
        ictd_fix_scalar_ffn=scalar_ffn,
    ).to(device=DEVICE, dtype=torch.float64)


def test_hermitian_pair_formula_and_global_u1_invariance():
    torch.manual_seed(3)
    operator = PhaseHermitianScalarResidual(
        num_elements=2,
        channels=3,
        lmax=2,
        residual_scale_init=0.05,
        internal_compute_dtype=torch.float64,
    ).to(device=DEVICE, dtype=torch.float64)
    dim = 3 * (2 + 1) ** 2
    real = torch.randn(5, dim, dtype=torch.float64, device=DEVICE)
    imag = torch.randn(5, dim, dtype=torch.float64, device=DEVICE)

    phi = torch.tensor(0.731, dtype=torch.float64, device=DEVICE)
    c, s = torch.cos(phi), torch.sin(phi)
    real_rot = c * real - s * imag
    imag_rot = s * real + c * imag
    rho = operator.hermitian_features(real, imag)
    rho_rot = operator.hermitian_features(real_rot, imag_rot)
    real_blocks = pure_cartesian_ictd_fix._split_irreps(
        real, operator.channels, operator.lmax
    )
    imag_blocks = pure_cartesian_ictd_fix._split_irreps(
        imag, operator.channels, operator.lmax
    )
    rho_reference = operator.hermitian_product(
        real_blocks, real_blocks
    ) + operator.hermitian_product(imag_blocks, imag_blocks)
    torch.testing.assert_close(rho_rot, rho, atol=2.0e-14, rtol=2.0e-14)
    torch.testing.assert_close(rho, rho_reference, atol=2.0e-14, rtol=2.0e-14)

    # The l=0, one-channel special case makes the intended pairwise coherence
    # explicit: |sum_j b_j exp(i theta_j)|^2 = sum_jk b_j b_k cos(theta_j-theta_k).
    scalar_operator = PhaseHermitianScalarResidual(
        num_elements=1,
        channels=1,
        lmax=0,
        internal_compute_dtype=torch.float64,
    ).to(device=DEVICE, dtype=torch.float64)
    b = torch.tensor([0.7, -1.1, 0.4], dtype=torch.float64, device=DEVICE)
    theta = torch.tensor([-0.2, 0.8, 2.0], dtype=torch.float64, device=DEVICE)
    re = (b * torch.cos(theta)).sum().reshape(1, 1)
    im = (b * torch.sin(theta)).sum().reshape(1, 1)
    got = scalar_operator.hermitian_features(re, im).squeeze()
    got_doublet = scalar_operator.hermitian_features_doublet(
        torch.stack((re, im), dim=-2)
    ).squeeze()
    expected = sum(
        b[j] * b[k] * torch.cos(theta[j] - theta[k])
        for j in range(3)
        for k in range(3)
    )
    torch.testing.assert_close(got, expected, atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(got_doublet, got, atol=1.0e-14, rtol=1.0e-14)


def test_full_l_low_rank_density_u1_invariance_and_shapes():
    torch.manual_seed(43)
    operator = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=4,
        lmax=2,
        density_rank=2,
        residual_scale_init=0.05,
    ).to(device=DEVICE, dtype=torch.float64)
    dim = 4 * (2 + 1) ** 2
    real = torch.randn(5, dim, dtype=torch.float64, device=DEVICE)
    imag = torch.randn(5, dim, dtype=torch.float64, device=DEVICE)
    phi = torch.tensor(-0.417, dtype=torch.float64, device=DEVICE)
    c, s = torch.cos(phi), torch.sin(phi)

    blocks = operator.hermitian_blocks(real, imag)
    doublet_blocks = operator.hermitian_blocks_doublet(
        torch.stack((real, imag), dim=-2)
    )
    rotated = operator.hermitian_blocks(c * real - s * imag, s * real + c * imag)

    # Reference the original unfused formulation path by path.  This protects
    # the eager kernel fusion from silently changing CG ordering or the sign of
    # the independent off-diagonal Hermitian component.
    real_in = pure_cartesian_ictd_fix._split_irreps(
        real, operator.channels, operator.lmax
    )
    imag_in = pure_cartesian_ictd_fix._split_irreps(
        imag, operator.channels, operator.lmax
    )
    real_rank = {
        l: operator._project_block(real_in[l], operator.rank_projections[str(l)])
        for l in range(operator.lmax + 1)
    }
    imag_rank = {
        l: operator._project_block(imag_in[l], operator.rank_projections[str(l)])
        for l in range(operator.lmax + 1)
    }
    reference_parts = {l: [] for l in range(operator.lmax + 1)}
    for l1, l2, out_l, component, cg_name in operator.paths:
        cg = getattr(operator, cg_name).to(dtype=real.dtype, device=real.device)
        if component == "real":
            value = operator._couple(
                real_rank[l1], real_rank[l2], cg
            ) + operator._couple(
                imag_rank[l1], imag_rank[l2], cg
            )
        else:
            value = operator._couple(
                imag_rank[l1], real_rank[l2], cg
            ) - operator._couple(
                real_rank[l1], imag_rank[l2], cg
            )
        reference_parts[out_l].append(value)
    reference = {l: torch.cat(reference_parts[l], dim=-2) for l in range(operator.lmax + 1)}

    assert blocks[0].shape == (5, 6, 1)
    assert blocks[1].shape == (5, 8, 3)
    assert blocks[2].shape == (5, 8, 5)
    for out_l in range(3):
        torch.testing.assert_close(
            doublet_blocks[out_l], blocks[out_l], atol=3.0e-13, rtol=3.0e-13
        )
        torch.testing.assert_close(
            rotated[out_l], blocks[out_l], atol=3.0e-13, rtol=3.0e-13
        )
        torch.testing.assert_close(
            blocks[out_l], reference[out_l], atol=3.0e-13, rtol=3.0e-13
        )
        assert blocks[out_l].abs().max().item() > 0.0


def test_charge2_quadratic_density_formula_breaks_u1_with_matched_parameters():
    torch.manual_seed(143)
    hermitian = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=4,
        lmax=2,
        density_rank=2,
        residual_scale_init=0.05,
        quadratic_form="hermitian",
    ).to(device=DEVICE, dtype=torch.float64)
    charge2 = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=4,
        lmax=2,
        density_rank=2,
        residual_scale_init=0.05,
        quadratic_form="charge2",
    ).to(device=DEVICE, dtype=torch.float64)
    charge2.load_state_dict(hermitian.state_dict(), strict=True)
    assert sum(p.numel() for p in charge2.parameters()) == sum(
        p.numel() for p in hermitian.parameters()
    )
    assert tuple(charge2.paths) == tuple(hermitian.paths)

    dim = charge2.channels * (charge2.lmax + 1) ** 2
    real = torch.randn(5, dim, dtype=torch.float64, device=DEVICE)
    imag = torch.randn(5, dim, dtype=torch.float64, device=DEVICE)
    blocks = charge2.quadratic_blocks(real, imag)

    real_in = pure_cartesian_ictd_fix._split_irreps(
        real, charge2.channels, charge2.lmax
    )
    imag_in = pure_cartesian_ictd_fix._split_irreps(
        imag, charge2.channels, charge2.lmax
    )
    real_rank = {
        ell: charge2._project_block(
            real_in[ell], charge2.rank_projections[str(ell)]
        )
        for ell in range(charge2.lmax + 1)
    }
    imag_rank = {
        ell: charge2._project_block(
            imag_in[ell], charge2.rank_projections[str(ell)]
        )
        for ell in range(charge2.lmax + 1)
    }
    reference_parts = {ell: [] for ell in range(charge2.lmax + 1)}
    for l1, l2, out_l, component, cg_name in charge2.paths:
        cg = getattr(charge2, cg_name).to(dtype=real.dtype, device=real.device)
        if component == "real":
            value = charge2._couple(
                real_rank[l1], real_rank[l2], cg
            ) - charge2._couple(
                imag_rank[l1], imag_rank[l2], cg
            )
        else:
            value = charge2._couple(
                real_rank[l1], imag_rank[l2], cg
            ) + charge2._couple(
                imag_rank[l1], real_rank[l2], cg
            )
        reference_parts[out_l].append(value)
    reference = {
        ell: torch.cat(reference_parts[ell], dim=-2)
        for ell in range(charge2.lmax + 1)
    }
    for out_l in range(charge2.lmax + 1):
        torch.testing.assert_close(
            blocks[out_l], reference[out_l], atol=3.0e-13, rtol=3.0e-13
        )

    phi = torch.tensor(0.371, dtype=torch.float64, device=DEVICE)
    c, s = torch.cos(phi), torch.sin(phi)
    rotated = charge2.quadratic_blocks(
        c * real - s * imag,
        s * real + c * imag,
    )
    assert any(
        not torch.allclose(
            rotated[out_l], blocks[out_l], atol=1.0e-10, rtol=1.0e-10
        )
        for out_l in range(charge2.lmax + 1)
    )


def test_charge2_and_full_u1_models_are_exactly_parameter_matched():
    torch.manual_seed(145)
    full = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full",
    )
    charge2 = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="charge2",
    )
    assert tuple(full.state_dict()) == tuple(charge2.state_dict())
    assert sum(p.numel() for p in full.parameters()) == sum(
        p.numel() for p in charge2.parameters()
    )
    charge2.load_state_dict(full.state_dict(), strict=True)


def test_full_l_diagonal_density_is_invariant_to_independent_edge_phases():
    torch.manual_seed(44)
    operator = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=3,
        lmax=2,
        density_rank=2,
        residual_scale_init=0.05,
    ).to(device=DEVICE, dtype=torch.float64)
    num_edges = 9
    num_nodes = 4
    dim = operator.channels * (operator.lmax + 1) ** 2
    edge_doublet = torch.randn(
        num_edges,
        2,
        dim,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    edge_dst = torch.tensor(
        [0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=torch.long, device=DEVICE
    )
    node_type_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=DEVICE)
    diagonal = operator.forward_diagonal_edges_doublet(
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )

    phi = torch.linspace(
        -1.2, 0.9, num_edges, dtype=torch.float64, device=DEVICE
    ).reshape(num_edges, 1)
    c, s = torch.cos(phi), torch.sin(phi)
    real, imag = edge_doublet[:, 0], edge_doublet[:, 1]
    rotated = torch.stack((c * real - s * imag, s * real + c * imag), dim=1)
    diagonal_rotated = operator.forward_diagonal_edges_doublet(
        rotated,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    torch.testing.assert_close(
        diagonal_rotated, diagonal, atol=4.0e-13, rtol=4.0e-13
    )

    # The full post-aggregation density retains cross-edge relative phases.
    node_doublet = pure_cartesian_ictd_fix.scatter(
        edge_doublet, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
    )
    node_doublet_rotated = pure_cartesian_ictd_fix.scatter(
        rotated, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
    )
    full = operator.forward_doublet(
        node_doublet, node_attrs=None, node_type_idx=node_type_idx
    )
    full_rotated = operator.forward_doublet(
        node_doublet_rotated, node_attrs=None, node_type_idx=node_type_idx
    )
    assert not torch.allclose(full_rotated, full, atol=1.0e-10, rtol=1.0e-10)

    diagonal.square().mean().backward()
    assert edge_doublet.grad is not None
    assert edge_doublet.grad.abs().max().item() > 0.0


def test_full_l_coherence_gate_interpolates_diagonal_and_full_density():
    torch.manual_seed(46)
    operator = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=3,
        lmax=2,
        density_rank=2,
        residual_scale_init=0.05,
        coherence_gate=True,
    ).to(device=DEVICE, dtype=torch.float64)
    num_edges = 10
    num_nodes = 4
    dim = operator.channels * (operator.lmax + 1) ** 2
    edge_doublet = torch.randn(
        num_edges,
        2,
        dim,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    edge_dst = torch.tensor(
        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
        dtype=torch.long,
        device=DEVICE,
    )
    node_type_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=DEVICE)
    node_doublet = pure_cartesian_ictd_fix.scatter(
        edge_doublet, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
    )
    full = operator.forward_doublet(
        node_doublet, node_attrs=None, node_type_idx=node_type_idx
    )
    diagonal = operator.forward_diagonal_edges_doublet(
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )

    with torch.no_grad():
        operator.coherence_scale.fill_(1.0)
    gated_full = operator.forward_coherence_gated_doublet(
        node_doublet,
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    torch.testing.assert_close(gated_full, full, atol=4.0e-13, rtol=4.0e-13)

    with torch.no_grad():
        operator.coherence_scale.zero_()
    gated_diagonal = operator.forward_coherence_gated_doublet(
        node_doublet,
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    torch.testing.assert_close(
        gated_diagonal, diagonal, atol=4.0e-13, rtol=4.0e-13
    )

    with torch.no_grad():
        operator.coherence_scale.fill_(0.7)
    operator.zero_grad(set_to_none=True)
    gated = operator.forward_coherence_gated_doublet(
        node_doublet,
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    gated.square().mean().backward()
    assert operator.coherence_scale.grad is not None
    assert operator.coherence_scale.grad.abs().max().item() > 0.0
    assert edge_doublet.grad is not None
    assert edge_doublet.grad.abs().max().item() > 0.0


def test_full_l_adaptive_coherence_is_bounded_and_diagonal_initialized():
    torch.manual_seed(146)
    initial_gamma = 0.1
    operator = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=3,
        lmax=2,
        density_rank=2,
        residual_scale_init=0.05,
        adaptive_coherence=True,
        adaptive_coherence_init=initial_gamma,
    ).to(device=DEVICE, dtype=torch.float64)
    num_edges = 10
    num_nodes = 4
    dim = operator.channels * (operator.lmax + 1) ** 2
    edge_doublet = torch.randn(
        num_edges,
        2,
        dim,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    edge_dst = torch.tensor(
        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
        dtype=torch.long,
        device=DEVICE,
    )
    node_type_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=DEVICE)
    node_doublet = pure_cartesian_ictd_fix.scatter(
        edge_doublet, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
    )
    full = operator.forward_doublet(
        node_doublet, node_attrs=None, node_type_idx=node_type_idx
    )
    diagonal = operator.forward_diagonal_edges_doublet(
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    adaptive = operator.forward_coherence_gated_doublet(
        node_doublet,
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    gamma = operator.effective_coherence_scale()
    torch.testing.assert_close(
        gamma,
        torch.full_like(gamma, initial_gamma),
        atol=5.0e-8,
        rtol=5.0e-8,
    )
    assert torch.all((gamma > 0.0) & (gamma < 1.0))
    expected = diagonal + gamma[0] * (full - diagonal)
    torch.testing.assert_close(adaptive, expected, atol=8.0e-13, rtol=8.0e-13)

    adaptive.square().mean().backward()
    assert operator.coherence_logit is not None
    assert operator.coherence_logit.grad is not None
    assert operator.coherence_logit.grad.abs().max().item() > 0.0
    assert edge_doublet.grad is not None
    assert edge_doublet.grad.abs().max().item() > 0.0


def test_full_l_environment_adaptive_coherence_is_local_and_invariant_initialized():
    torch.manual_seed(246)
    initial_gamma = 0.5
    operator = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=4,
        lmax=2,
        density_rank=2,
        residual_scale_init=0.05,
        environment_adaptive_coherence=True,
        adaptive_coherence_init=initial_gamma,
    ).to(device=DEVICE, dtype=torch.float64)
    num_edges = 10
    num_nodes = 4
    dim = operator.channels * (operator.lmax + 1) ** 2
    edge_doublet = torch.randn(
        num_edges,
        2,
        dim,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    gate_features = torch.randn(
        num_nodes,
        operator.channels,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    edge_dst = torch.tensor(
        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
        dtype=torch.long,
        device=DEVICE,
    )
    node_type_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=DEVICE)
    node_doublet = pure_cartesian_ictd_fix.scatter(
        edge_doublet, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
    )
    full = operator.forward_doublet(
        node_doublet, node_attrs=None, node_type_idx=node_type_idx
    )
    diagonal = operator.forward_diagonal_edges_doublet(
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    adaptive = operator.forward_coherence_gated_doublet(
        node_doublet,
        edge_doublet,
        gate_features=gate_features,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    gamma = operator.effective_coherence_scale(gate_features)
    assert gamma.shape == (num_nodes, operator.lmax + 1)
    torch.testing.assert_close(
        gamma,
        torch.full_like(gamma, initial_gamma),
        atol=5.0e-8,
        rtol=5.0e-8,
    )
    torch.testing.assert_close(
        adaptive,
        diagonal + initial_gamma * (full - diagonal),
        atol=8.0e-13,
        rtol=8.0e-13,
    )

    adaptive.square().mean().backward()
    assert operator.coherence_context is not None
    final = operator.coherence_context[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.weight.grad is not None
    assert final.weight.grad.abs().max().item() > 0.0

    with torch.no_grad():
        final.weight.normal_(mean=0.0, std=0.1)
    local_gamma = operator.effective_coherence_scale(gate_features.detach())
    assert torch.all((local_gamma > 0.0) & (local_gamma < 1.0))
    assert not torch.allclose(local_gamma[0], local_gamma[1])


def test_factorized_diagonal_and_coherence_gate_match_explicit_doublet():
    torch.manual_seed(47)
    operator = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=4,
        lmax=2,
        density_rank=3,
        residual_scale_init=0.05,
        coherence_gate=True,
    ).to(device=DEVICE, dtype=torch.float64)
    num_edges = 11
    num_nodes = 5
    dim = operator.channels * (operator.lmax + 1) ** 2
    edge_orbital = torch.randn(
        num_edges,
        dim,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    edge_coefficient = torch.randn(
        num_edges,
        2,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    edge_doublet = (
        edge_coefficient.unsqueeze(-1) * edge_orbital.unsqueeze(-2)
    )
    edge_norm_sq = edge_coefficient.square().sum(dim=-1)
    edge_dst = torch.tensor(
        [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        dtype=torch.long,
        device=DEVICE,
    )
    node_type_idx = torch.tensor(
        [0, 1, 0, 1, 0], dtype=torch.long, device=DEVICE
    )

    explicit_diagonal = operator.forward_diagonal_edges_doublet(
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    factorized_diagonal = operator.forward_diagonal_edges_factorized(
        edge_orbital,
        edge_norm_sq,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    torch.testing.assert_close(
        factorized_diagonal,
        explicit_diagonal,
        atol=8.0e-13,
        rtol=8.0e-13,
    )

    node_doublet = pure_cartesian_ictd_fix.scatter(
        edge_doublet, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
    )
    with torch.no_grad():
        operator.coherence_scale.copy_(
            torch.tensor([0.2, 0.7, 1.3], dtype=torch.float64, device=DEVICE)
        )
    explicit_gated = operator.forward_coherence_gated_doublet(
        node_doublet,
        edge_doublet,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    factorized_gated = operator.forward_coherence_gated_factorized(
        node_doublet,
        edge_orbital,
        edge_norm_sq,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    torch.testing.assert_close(
        factorized_gated,
        explicit_gated,
        atol=8.0e-13,
        rtol=8.0e-13,
    )

    operator.zero_grad(set_to_none=True)
    factorized_gated.square().mean().backward()
    assert edge_orbital.grad is not None
    assert edge_orbital.grad.abs().max().item() > 0.0
    assert edge_coefficient.grad is not None
    assert edge_coefficient.grad.abs().max().item() > 0.0
    assert operator.coherence_scale.grad is not None
    assert operator.coherence_scale.grad.abs().max().item() > 0.0


def test_pair_count_balanced_density_matches_manual_coordination_scaling():
    torch.manual_seed(48)
    operator = PhaseHermitianFullLResidual(
        num_elements=2,
        channels=4,
        lmax=2,
        density_rank=3,
        residual_scale_init=0.05,
    ).to(device=DEVICE, dtype=torch.float64)
    num_edges = 12
    num_nodes = 5
    dim = operator.channels * (operator.lmax + 1) ** 2
    edge_orbital = torch.randn(
        num_edges,
        dim,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    edge_coefficient = torch.randn(
        num_edges,
        2,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    edge_doublet = edge_coefficient.unsqueeze(-1) * edge_orbital.unsqueeze(-2)
    edge_norm_sq = edge_coefficient.square().sum(dim=-1)
    edge_dst = torch.tensor(
        [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        dtype=torch.long,
        device=DEVICE,
    )
    node_type_idx = torch.tensor(
        [0, 1, 0, 1, 0], dtype=torch.long, device=DEVICE
    )
    node_doublet = pure_cartesian_ictd_fix.scatter(
        edge_doublet, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
    )
    effective_coordination = torch.tensor(
        [1.0, 2.5, 4.0, 6.0, 8.0],
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    reference_coordination = 4.0

    balanced = operator.forward_pair_count_balanced_factorized(
        node_doublet,
        edge_orbital,
        edge_norm_sq,
        effective_coordination,
        reference_coordination=reference_coordination,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    full_blocks = operator.hermitian_blocks_doublet(node_doublet)
    diagonal_blocks = operator._diagonal_blocks_factorized(
        edge_orbital,
        edge_norm_sq,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
    )
    n_eff = effective_coordination.clamp_min(1.0)
    d_scale = reference_coordination / n_eff
    c_scale = (
        reference_coordination * (reference_coordination - 1.0) + 1.0
    ) / (n_eff * (n_eff - 1.0) + 1.0)
    expected_blocks = {
        ell: (
            d_scale[:, None, None] * diagonal_blocks[ell]
            + c_scale[:, None, None]
            * (full_blocks[ell] - diagonal_blocks[ell])
        )
        for ell in range(operator.lmax + 1)
    }
    expected = operator._mix_density_blocks(
        expected_blocks,
        reference=node_doublet,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    torch.testing.assert_close(balanced, expected, atol=8.0e-13, rtol=8.0e-13)

    reference_nodes = torch.full_like(
        effective_coordination, reference_coordination
    )
    balanced_at_reference = operator.forward_pair_count_balanced_factorized(
        node_doublet,
        edge_orbital,
        edge_norm_sq,
        reference_nodes,
        reference_coordination=reference_coordination,
        edge_dst=edge_dst,
        num_nodes=num_nodes,
        node_attrs=None,
        node_type_idx=node_type_idx,
    )
    full = operator.forward_doublet(
        node_doublet, node_attrs=None, node_type_idx=node_type_idx
    )
    torch.testing.assert_close(
        balanced_at_reference, full, atol=8.0e-13, rtol=8.0e-13
    )

    operator.zero_grad(set_to_none=True)
    balanced.square().mean().backward()
    assert edge_orbital.grad is not None
    assert edge_orbital.grad.abs().max().item() > 0.0
    assert edge_coefficient.grad is not None
    assert edge_coefficient.grad.abs().max().item() > 0.0
    assert effective_coordination.grad is not None
    assert effective_coordination.grad.abs().max().item() > 0.0


def test_persistent_charged_update_global_u1_equivariance_and_gradients():
    torch.manual_seed(45)
    update = PersistentChargedUpdate(channels=4, lmax=2).to(
        device=DEVICE, dtype=torch.float64
    )
    dim = 4 * (2 + 1) ** 2
    previous_real = torch.randn(
        5, dim, dtype=torch.float64, device=DEVICE, requires_grad=True
    )
    previous_imag = torch.randn(
        5, dim, dtype=torch.float64, device=DEVICE, requires_grad=True
    )
    incoming_real = torch.randn(5, dim, dtype=torch.float64, device=DEVICE)
    incoming_imag = torch.randn(5, dim, dtype=torch.float64, device=DEVICE)
    out_real, out_imag = update(
        previous_real, previous_imag, incoming_real, incoming_imag
    )
    out_doublet = update.forward_doublet(
        torch.stack((previous_real, previous_imag), dim=-2),
        torch.stack((incoming_real, incoming_imag), dim=-2),
    )
    torch.testing.assert_close(out_doublet[..., 0, :], out_real, atol=3.0e-13, rtol=3.0e-13)
    torch.testing.assert_close(out_doublet[..., 1, :], out_imag, atol=3.0e-13, rtol=3.0e-13)

    # Reference the original two-linear formulation. The optimized forward
    # fuses the previous/incoming channel maps into one GEMM per l but must
    # preserve the same learned operator and checkpoint parameters.
    previous_doublet = update.previous_linear(
        torch.stack((previous_real, previous_imag), dim=-2)
    )
    incoming_doublet = update.incoming_linear(
        torch.stack((incoming_real, incoming_imag), dim=-2)
    )
    previous_blocks = pure_cartesian_ictd_fix._split_irreps(
        previous_doublet, update.channels, update.lmax
    )
    incoming_blocks = pure_cartesian_ictd_fix._split_irreps(
        incoming_doublet, update.channels, update.lmax
    )
    gates = update.memory_logits.sigmoid()
    reference = pure_cartesian_ictd_fix._merge_irreps(
        {
            l: gates[l] * previous_blocks[l]
            + (1.0 - gates[l]) * incoming_blocks[l]
            for l in range(update.lmax + 1)
        },
        update.channels,
        update.lmax,
    )
    torch.testing.assert_close(out_real, reference[..., 0, :], atol=3.0e-13, rtol=3.0e-13)
    torch.testing.assert_close(out_imag, reference[..., 1, :], atol=3.0e-13, rtol=3.0e-13)

    phi = torch.tensor(0.619, dtype=torch.float64, device=DEVICE)
    c, s = torch.cos(phi), torch.sin(phi)
    rotated_real, rotated_imag = update(
        c * previous_real - s * previous_imag,
        s * previous_real + c * previous_imag,
        c * incoming_real - s * incoming_imag,
        s * incoming_real + c * incoming_imag,
    )
    torch.testing.assert_close(
        rotated_real, c * out_real - s * out_imag, atol=3.0e-13, rtol=3.0e-13
    )
    torch.testing.assert_close(
        rotated_imag, s * out_real + c * out_imag, atol=3.0e-13, rtol=3.0e-13
    )

    (out_real.square().mean() + out_imag.square().mean()).backward()
    assert previous_real.grad is not None and previous_real.grad.abs().max().item() > 0.0
    assert previous_imag.grad is not None and previous_imag.grad.abs().max().item() > 0.0
    assert update.memory_logits.grad is not None
    assert update.memory_logits.grad.abs().max().item() > 0.0
    assert update.previous_linear.adapters["1"].weight.grad is not None
    assert update.incoming_linear.adapters["1"].weight.grad is not None


@pytest.mark.parametrize(
    "density_pairs",
    ["full", "charge2", "full-nonlinear", "full-nonlinear-readout"],
)
def test_full_l_phase_model_rotation_force_covariance_and_gradients(
    density_pairs: str,
):
    torch.manual_seed(47)
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs=density_pairs,
    )
    graph = make_fixed_graph(
        num_nodes=8,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=49,
    )
    model.eval()
    energy, forces, _ = compute_energy_forces(model, graph, create_graph=False)
    rotation = random_rotation(dtype=torch.float64, seed=51).to(DEVICE)
    rotated_graph = (graph[0] @ rotation.T,) + graph[1:]
    energy_rot, forces_rot, _ = compute_energy_forces(model, rotated_graph, create_graph=False)
    torch.testing.assert_close(energy_rot, energy, atol=3.0e-12, rtol=3.0e-12)
    torch.testing.assert_close(
        forces_rot, forces @ rotation.T, atol=3.0e-11, rtol=3.0e-11
    )

    model.train()
    model.zero_grad(set_to_none=True)
    model(*graph).sum().backward()
    adapter = model.phase_adapters["1"]
    assert adapter.rank_projections["1"].weight.grad is not None
    assert adapter.output_weights["1"].grad is not None
    assert adapter.residual_scale.grad is not None
    assert adapter.rank_projections["1"].weight.grad.abs().max().item() > 0.0
    assert adapter.output_weights["1"].grad.abs().max().item() > 0.0
    phase_head_grad = model.interactions[-1].phase_head.weight.grad
    assert phase_head_grad is not None and phase_head_grad.abs().max().item() > 0.0


@pytest.mark.parametrize(
    "coefficient,context,density_pairs",
    [
        ("positive", "content", "full"),
        ("signed", "content", "full"),
        ("cartesian", "content", "full"),
        ("polar", "radial", "full"),
        ("polar", "irrep-norm", "full"),
        ("polar", "content-irrep-norm", "full"),
        ("polar", "content", "diagonal"),
    ],
)
def test_full_l_coefficient_controls_forward_backward(
    coefficient: str, context: str, density_pairs: str
):
    torch.manual_seed(52)
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_coefficient=coefficient,
        phase_context=context,
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs=density_pairs,
    )
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=54,
    )
    model.train()
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    interaction = model.interactions[-1]
    amplitude_grad = interaction.phase_amplitude_head.weight.grad
    assert amplitude_grad is not None
    assert amplitude_grad.abs().max().item() > 0.0
    if density_pairs == "full":
        phase_grad = interaction.phase_head.weight.grad
        assert phase_grad is not None
        assert phase_grad.abs().max().item() > 0.0


@pytest.mark.parametrize(
    "phase_mode,phase_placement",
    [
        ("final-scalar-residual", "pre-product-l0"),
        ("final-full-l-residual", "pre-product-full-l"),
    ],
)
def test_persistent_phase_model_cross_layer_rotation_and_gradients(
    phase_mode: str, phase_placement: str
):
    torch.manual_seed(53)
    model = _build(
        phase_mode=phase_mode,
        phase_amplitude="softplus",
        phase_placement=phase_placement,
        phase_density_rank=4,
        phase_scope="persistent",
    )
    assert all(interaction.phase_enabled for interaction in model.interactions)
    assert set(model.phase_adapters) == {"0", "1"}
    assert set(model.charged_updates) == {"1"}
    graph = make_fixed_graph(
        num_nodes=8,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=55,
    )

    model.eval()
    energy, forces, _ = compute_energy_forces(model, graph, create_graph=False)
    rotation = random_rotation(dtype=torch.float64, seed=57).to(DEVICE)
    rotated_graph = (graph[0] @ rotation.T,) + graph[1:]
    energy_rot, forces_rot, _ = compute_energy_forces(
        model, rotated_graph, create_graph=False
    )
    torch.testing.assert_close(energy_rot, energy, atol=4.0e-12, rtol=4.0e-12)
    torch.testing.assert_close(
        forces_rot, forces @ rotation.T, atol=4.0e-11, rtol=4.0e-11
    )

    # Rotate the charged outputs of every layer by the same phase. This tests
    # the assembled recurrence and all per-layer Hermitian injections, rather
    # than only its component operators in isolation.
    phi = torch.tensor(-0.371, dtype=torch.float64, device=DEVICE)
    c, s = torch.cos(phi), torch.sin(phi)

    def rotate_charged(_module, _inputs, output):
        if len(output) == 3:
            message, sc, doublet = output
            real, imag = doublet[..., 0, :], doublet[..., 1, :]
            return message, sc, torch.stack(
                (c * real - s * imag, s * real + c * imag), dim=-2
            )
        message, sc, real, imag = output
        return message, sc, c * real - s * imag, s * real + c * imag

    handles = [
        interaction.register_forward_hook(rotate_charged)
        for interaction in model.interactions
    ]
    try:
        energy_u1, forces_u1, _ = compute_energy_forces(
            model, graph, create_graph=False
        )
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(energy_u1, energy, atol=4.0e-12, rtol=4.0e-12)
    torch.testing.assert_close(forces_u1, forces, atol=4.0e-11, rtol=4.0e-11)

    # A layer-specific phase rotation is intentionally not a symmetry: all
    # depth sources share one global phase frame. This guards against calling
    # the present implementation a local or independently gauged U(1) model.
    handle = model.interactions[0].register_forward_hook(rotate_charged)
    try:
        energy_layer_shifted = model(*graph)
    finally:
        handle.remove()
    assert not torch.allclose(
        energy_layer_shifted, energy, atol=1.0e-12, rtol=1.0e-12
    )

    model.train()
    model.zero_grad(set_to_none=True)
    model(*graph).sum().backward()
    first_phase_grad = model.interactions[0].phase_head.weight.grad
    second_phase_grad = model.interactions[1].phase_head.weight.grad
    update = model.charged_updates["1"]
    assert first_phase_grad is not None and first_phase_grad.abs().max().item() > 0.0
    assert second_phase_grad is not None and second_phase_grad.abs().max().item() > 0.0
    assert update.memory_logits.grad is not None
    assert update.memory_logits.grad.abs().max().item() > 0.0
    assert update.previous_linear.adapters["1"].weight.grad is not None
    assert update.incoming_linear.adapters["1"].weight.grad is not None


def test_all_layer_diagonal_uses_fresh_edge_self_density():
    """Repeated j=k injection must retain edge identity at every layer."""
    torch.manual_seed(59)
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="diagonal",
        phase_scope="persistent",
    )
    assert all(interaction.phase_enabled for interaction in model.interactions)
    assert set(model.phase_adapters) == {"0", "1"}
    # A strict diagonal control is reconstructed from fresh edge messages.  It
    # must not pass through the atom-level charged recurrence, which has already
    # summed over neighbours and therefore contains off-diagonal j != k terms.
    assert len(model.charged_updates) == 0

    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=61,
    )
    model.train()
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    for layer_idx, interaction in enumerate(model.interactions):
        amplitude_grad = interaction.phase_amplitude_head.weight.grad
        assert amplitude_grad is not None
        assert amplitude_grad.abs().max().item() > 0.0
        adapter = model.phase_adapters[str(layer_idx)]
        assert adapter.residual_scale.grad is not None
        assert adapter.residual_scale.grad.abs().max().item() > 0.0


def test_phase_mode_none_preserves_baseline_state_and_forward():
    torch.manual_seed(17)
    implicit = build_model(
        channels=8,
        lmax=1,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-pure-u",
        dtype=torch.float64,
        device=DEVICE,
        correlation=2,
        attn_heads=0,
    ).to(device=DEVICE, dtype=torch.float64)
    torch.manual_seed(17)
    explicit = _build(phase_mode="none")

    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    assert not any("phase_" in key for key in explicit_state)
    for key in implicit_state:
        torch.testing.assert_close(implicit_state[key], explicit_state[key], atol=0.0, rtol=0.0)

    graph = make_fixed_graph(
        num_nodes=8,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=9,
    )
    implicit.eval()
    explicit.eval()
    with torch.no_grad():
        torch.testing.assert_close(implicit(*graph), explicit(*graph), atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("amplitude", ["unit", "softplus"])
@pytest.mark.parametrize(
    "placement", ["post-product", "pre-product-l0", "pre-and-post"]
)
def test_phase_model_rotation_force_covariance_and_gradients(
    amplitude: str, placement: str
):
    torch.manual_seed(11)
    model = _build(
        phase_mode="final-scalar-residual",
        phase_amplitude=amplitude,
        phase_placement=placement,
    )
    graph = make_fixed_graph(
        num_nodes=9,
        avg_degree=5,
        dtype=torch.float64,
        device=DEVICE,
        seed=5,
    )

    model.eval()
    energy, forces, _ = compute_energy_forces(model, graph, create_graph=False)
    rotation = random_rotation(dtype=torch.float64, seed=23).to(DEVICE)
    rotated_graph = (graph[0] @ rotation.T,) + graph[1:]
    energy_rot, forces_rot, _ = compute_energy_forces(model, rotated_graph, create_graph=False)
    torch.testing.assert_close(energy_rot, energy, atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(
        forces_rot,
        forces @ rotation.T,
        atol=2.0e-11,
        rtol=2.0e-11,
    )

    model.train()
    model.zero_grad(set_to_none=True)
    model(*graph).sum().backward()
    phase_head_grad = model.interactions[-1].phase_head.weight.grad
    residual_weight_grad = model.phase_adapters["1"].scalar_linear.weight.grad
    assert phase_head_grad is not None and torch.isfinite(phase_head_grad).all()
    assert residual_weight_grad is not None and torch.isfinite(residual_weight_grad).all()
    assert phase_head_grad.abs().max().item() > 0.0
    assert residual_weight_grad.abs().max().item() > 0.0
    if amplitude == "softplus":
        amplitude_grad = model.interactions[-1].phase_amplitude_head.weight.grad
        assert amplitude_grad is not None and amplitude_grad.abs().max().item() > 0.0


def test_phase_attention_combination_is_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_model(
            channels=8,
            lmax=1,
            num_interaction=2,
            route="baseline",
            product_backend="ictd-pure-u",
            dtype=torch.float64,
            device=DEVICE,
            correlation=2,
            attn_heads=1,
            ictd_fix_phase_mode="final-scalar-residual",
        )
    with pytest.raises(ValueError, match="persistent scalar phase"):
        _build(
            phase_mode="final-scalar-residual",
            phase_placement="post-product",
            phase_scope="persistent",
        )


def test_pre_product_phase_changes_symmetric_contraction_input():
    torch.manual_seed(29)
    post = _build(
        phase_mode="final-scalar-residual",
        phase_amplitude="softplus",
        phase_placement="post-product",
    ).eval()
    pre = _build(
        phase_mode="final-scalar-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-l0",
    ).eval()
    pre.load_state_dict(post.state_dict(), strict=True)
    graph = make_fixed_graph(
        num_nodes=8,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=37,
    )
    with torch.no_grad():
        post_energy = post(*graph)
        pre_energy = pre(*graph)
    assert post.ictd_fix_phase_placement == "post-product"
    assert pre.ictd_fix_phase_placement == "pre-product-l0"
    assert not torch.allclose(post_energy, pre_energy, atol=1.0e-14, rtol=1.0e-14)


@pytest.mark.parametrize(
    "phase_mode,phase_placement,phase_scope,phase_density_pairs,phase_normalization",
    [
        (
            "final-scalar-residual",
            "pre-product-l0",
            "final",
            "full",
            "avg-neighbors",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "final",
            "full",
            "avg-neighbors",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "final",
            "charge2",
            "avg-neighbors",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "final",
            "full-gated",
            "local-effective",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "final",
            "full-adaptive",
            "avg-neighbors",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "final",
            "full-adaptive-env",
            "avg-neighbors",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "final",
            "full-balanced",
            "avg-neighbors",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "final",
            "full-nonlinear",
            "avg-neighbors",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "final",
            "full-nonlinear-readout",
            "avg-neighbors",
        ),
        (
            "final-scalar-residual",
            "pre-product-l0",
            "persistent",
            "full",
            "avg-neighbors",
        ),
        (
            "final-full-l-residual",
            "pre-product-full-l",
            "persistent",
            "full",
            "avg-neighbors",
        ),
    ],
)
def test_phase_checkpoint_strict_deployment_round_trip(
    tmp_path,
    phase_mode: str,
    phase_placement: str,
    phase_scope: str,
    phase_density_pairs: str,
    phase_normalization: str,
):
    dtype = torch.float64
    phase_heads = (
        4
        if phase_density_pairs in {"full-nonlinear", "full-nonlinear-readout"}
        else 1
    )
    nonlinear_layer_readouts = phase_density_pairs == "full-nonlinear"
    final_layer_readout_only = phase_density_pairs == "full-nonlinear-readout"
    element_energy_correction = phase_density_pairs == "full-nonlinear-readout"
    scalar_ffn = phase_density_pairs == "full-nonlinear-readout"
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = 8
    cfg.irreps_output_conv_channels = 8
    cfg.lmax = 1
    cfg.num_layers = 1
    cfg.max_radius = 5.0
    cfg.max_radius_main = 5.0
    cfg.number_of_basis = 8
    cfg.number_of_basis_main = 8
    cfg.function_type = "bessel"
    cfg.internal_compute_dtype = dtype
    model = build_baseline_model(
        cfg,
        avg_num_neighbors=4.0,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-pure-u",
        correlation=2,
        first_layer_self_connection=True,
        polynomial_cutoff_p=5,
        radial_sqrt_num_basis=False,
        edge_lmax=None,
        attn_heads=0,
        phase_mode=phase_mode,
        phase_hidden_channels=12,
        phase_residual_scale_init=0.05,
        phase_amplitude="softplus",
        phase_placement=phase_placement,
        phase_density_rank=4,
        phase_density_pairs=phase_density_pairs,
        phase_normalization=phase_normalization,
        phase_scope=phase_scope,
        phase_heads=phase_heads,
        nonlinear_layer_readouts=nonlinear_layer_readouts,
        final_layer_readout_only=final_layer_readout_only,
        element_energy_correction=element_energy_correction,
        scalar_ffn=scalar_ffn,
        atomic_numbers=[1, 6, 7, 8],
        ictd_save_tp_mode="fully-connected",
        invariant_channels=8,
        device=DEVICE,
        dtype=dtype,
    )
    trainer = ForceTrainer(
        model,
        [],
        device=DEVICE,
        config=cfg,
        dtype=dtype,
        max_radius=5.0,
        epochs=1,
        extra_hparams={
            "num_interaction": 2,
            "invariant_channels": 8,
            "ictd_fix_route": "baseline",
            "ictd_fix_product_backend": "ictd-pure-u",
            "ictd_fix_first_layer_self_connection": True,
            "ictd_fix_readout_hidden_channels": 16,
            "ictd_fix_edge_lmax": 1,
            "save_contraction_order": 2,
            "ictd_save_tp_mode": "fully-connected",
            "ictd_fix_interaction_attn_heads": 0,
            "ictd_fix_phase_mode": phase_mode,
            "ictd_fix_phase_hidden_channels": 12,
            "ictd_fix_phase_residual_scale_init": 0.05,
            "ictd_fix_phase_amplitude": "softplus",
            "ictd_fix_phase_placement": phase_placement,
            "ictd_fix_phase_density_rank": 4,
            "ictd_fix_phase_density_pairs": phase_density_pairs,
            "ictd_fix_phase_normalization": phase_normalization,
            "ictd_fix_phase_scope": phase_scope,
            "ictd_fix_phase_heads": phase_heads,
            "ictd_fix_nonlinear_layer_readouts": nonlinear_layer_readouts,
            "ictd_fix_final_layer_readout_only": final_layer_readout_only,
            "ictd_fix_element_energy_correction": element_energy_correction,
            "ictd_fix_scalar_ffn": scalar_ffn,
            "radial_sqrt_num_basis": False,
            "polynomial_cutoff_p": 5,
            "avg_num_neighbors": 4.0,
        },
    )
    checkpoint = tmp_path / "phase.pth"
    trainer.save_checkpoint(str(checkpoint), epoch=0)
    deployed = LAMMPS_MLIAP_MFF.from_checkpoint(
        str(checkpoint),
        element_types=["H", "C", "N", "O"],
        device=str(DEVICE),
        atomic_energy_keys=[1, 6, 7, 8],
        atomic_energy_values=[0.0, 0.0, 0.0, 0.0],
    ).wrapper.model
    assert deployed.ictd_fix_phase_mode == phase_mode
    assert deployed.ictd_fix_phase_hidden_channels == 12
    assert deployed.ictd_fix_phase_amplitude == "softplus"
    assert deployed.ictd_fix_phase_placement == phase_placement
    assert deployed.ictd_fix_phase_density_rank == 4
    assert deployed.ictd_fix_phase_density_pairs == phase_density_pairs
    assert deployed.ictd_fix_phase_normalization == phase_normalization
    assert deployed.ictd_fix_phase_scope == phase_scope
    assert deployed.ictd_fix_phase_heads == phase_heads
    assert (
        deployed.ictd_fix_nonlinear_layer_readouts
        == nonlinear_layer_readouts
    )
    assert (
        deployed.ictd_fix_final_layer_readout_only
        == final_layer_readout_only
    )
    assert (
        deployed.ictd_fix_element_energy_correction
        == element_energy_correction
    )
    assert deployed.ictd_fix_scalar_ffn == scalar_ffn

    graph = make_fixed_graph(
        num_nodes=8,
        avg_degree=4,
        dtype=dtype,
        device=DEVICE,
        seed=19,
    )
    model.eval()
    deployed.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(*graph), deployed(*graph), atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize(
    "phase_mode,phase_placement,phase_scope",
    [
        ("final-scalar-residual", "pre-product-l0", "final"),
        ("final-full-l-residual", "pre-product-full-l", "final"),
        ("final-scalar-residual", "pre-product-l0", "persistent"),
        ("final-full-l-residual", "pre-product-full-l", "persistent"),
    ],
)
def test_phase_force_graph_makefx_trace(
    phase_mode: str, phase_placement: str, phase_scope: str
):
    model = _build(
        phase_mode=phase_mode,
        phase_placement=phase_placement,
        phase_density_rank=4,
        phase_scope=phase_scope,
    ).train()
    graph = make_fixed_graph(
        num_nodes=6,
        avg_degree=3,
        dtype=torch.float64,
        device=DEVICE,
        seed=31,
    )
    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    energy, forces = graph_module(*graph)
    assert torch.isfinite(energy)
    assert forces.shape == (6, 3)
    assert torch.isfinite(forces).all()


def test_charge2_force_graph_makefx_trace():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="charge2",
    ).train()
    graph = make_fixed_graph(
        num_nodes=6,
        avg_degree=3,
        dtype=torch.float64,
        device=DEVICE,
        seed=131,
    )
    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    energy, forces = graph_module(*graph)
    assert torch.isfinite(energy)
    assert forces.shape == (6, 3)
    assert torch.isfinite(forces).all()


def test_coherence_gated_local_normalization_forward_backward_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-gated",
        phase_normalization="local-effective",
    ).train()
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=83,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    adapter = model.phase_adapters["1"]
    assert adapter.coherence_scale is not None
    assert adapter.coherence_scale.grad is not None
    assert adapter.coherence_scale.grad.abs().max().item() > 0.0

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()


def test_adaptive_coherence_forward_backward_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-adaptive",
        phase_coherence_init=0.1,
        phase_normalization="avg-neighbors",
    ).train()
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=87,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    adapter = model.phase_adapters["1"]
    assert adapter.coherence_logit is not None
    assert adapter.coherence_logit.grad is not None
    assert adapter.coherence_logit.grad.abs().max().item() > 0.0
    torch.testing.assert_close(
        adapter.effective_coherence_scale().detach(),
        torch.full_like(adapter.coherence_logit, 0.1),
        atol=5.0e-8,
        rtol=5.0e-8,
    )

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()


def test_environment_adaptive_coherence_forward_backward_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-adaptive-env",
        phase_coherence_init=0.5,
        phase_normalization="avg-neighbors",
    ).train()
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=97,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    adapter = model.phase_adapters["1"]
    assert adapter.coherence_context is not None
    final = adapter.coherence_context[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.weight.grad is not None
    assert final.weight.grad.abs().max().item() > 0.0

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()


def test_pair_count_balanced_forward_backward_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-balanced",
        phase_normalization="avg-neighbors",
    ).train()
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=89,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    phase_head = model.interactions[1].phase_head
    assert phase_head is not None
    assert phase_head.weight.grad is not None
    assert phase_head.weight.grad.abs().max().item() > 0.0

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()


def test_full_nonlinear_angular_phase_and_output_gate_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-nonlinear",
        phase_normalization="avg-neighbors",
    ).train()
    interaction = model.interactions[1]
    adapter = model.phase_adapters["1"]
    assert interaction.phase_head is not None
    assert interaction.phase_head.out_features == interaction.target_lmax + 1
    assert adapter.output_gate_context is not None

    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=101,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    assert interaction.phase_head.weight.grad is not None
    assert interaction.phase_head.weight.grad.abs().max().item() > 0.0
    gate_output = adapter.output_gate_context[-1]
    assert isinstance(gate_output, torch.nn.Linear)
    assert gate_output.weight.grad is not None
    assert gate_output.weight.grad.abs().max().item() > 0.0

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()


def test_single_phase_head_preserves_shared_broadcast_exactly():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-nonlinear",
    )
    interaction = model.interactions[1]
    assert interaction.phase_heads == 1
    edge_count = 5
    for l in range(interaction.target_lmax + 1):
        path_channels = interaction.tp.path_counts_by_l[l] * interaction.channels
        edge_block = torch.randn(
            edge_count,
            path_channels,
            2 * l + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        phase_cos = torch.randn(
            edge_count,
            interaction.target_lmax + 1,
            1,
            dtype=torch.float64,
            device=DEVICE,
        )
        phase_sin = torch.randn_like(phase_cos)
        got = interaction._phase_stream_block(
            edge_block, phase_cos, phase_sin, l
        )
        cos_l = phase_cos[:, l : l + 1]
        sin_l = phase_sin[:, l : l + 1]
        expected = edge_block.unsqueeze(1) * torch.stack(
            (torch.ones_like(cos_l), cos_l, sin_l),
            dim=1,
        )
        torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)


def test_full_nonlinear_grouped_phase_gradients_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-nonlinear",
        phase_normalization="avg-neighbors",
        phase_heads=4,
    ).train()
    interaction = model.interactions[1]
    assert interaction.phase_head is not None
    assert interaction.phase_heads == 4
    assert interaction.phase_head.out_features == (
        interaction.target_lmax + 1
    ) * 4

    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=103,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    phase_grad = interaction.phase_head.weight.grad
    assert phase_grad is not None
    assert torch.isfinite(phase_grad).all()
    assert phase_grad.abs().sum(dim=1).min().item() > 0.0

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()


def test_nonlinear_intermediate_readout_gradients_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-nonlinear",
        nonlinear_layer_readouts=True,
    ).train()
    assert model.ictd_fix_nonlinear_layer_readouts
    assert (
        type(model.layer_energy_readouts[0]).__name__
        == "MACEStyleScalarReadoutSO3"
    )
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=107,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    energy.sum().backward()
    readout_grads = [
        parameter.grad
        for parameter in model.layer_energy_readouts[0].parameters()
    ]
    assert readout_grads
    assert all(grad is not None for grad in readout_grads)
    assert all(torch.isfinite(grad).all() for grad in readout_grads)

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert torch.isfinite(forces).all()


def test_final_layer_only_readout_removes_intermediate_head_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-nonlinear",
        final_layer_readout_only=True,
    ).train()
    assert model.ictd_fix_final_layer_readout_only
    assert len(model.layer_energy_readouts) == 0

    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=108,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    energy.sum().backward()
    final_readout_grads = [
        parameter.grad
        for parameter in model.last_layer_energy_readout.parameters()
    ]
    assert final_readout_grads
    assert all(grad is not None for grad in final_readout_grads)
    assert all(torch.isfinite(grad).all() for grad in final_readout_grads)

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert torch.isfinite(forces).all()


def test_element_energy_correction_is_force_neutral_and_makefx_safe():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-nonlinear",
        element_energy_correction=True,
    ).train()
    assert model.ictd_fix_element_energy_correction
    torch.testing.assert_close(
        model.element_energy_correction,
        torch.zeros_like(model.element_energy_correction),
    )
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=110,
    )

    def force_for_current_correction():
        pos = graph[0].detach().clone().requires_grad_(True)
        energy = model(pos, *graph[1:]).sum()
        return -torch.autograd.grad(energy, pos)[0]

    force_zero = force_for_current_correction()
    with torch.no_grad():
        model.element_energy_correction.copy_(
            torch.tensor(
                [0.3, -1.2, 0.4, 0.8],
                dtype=torch.float64,
                device=DEVICE,
            )
        )
    force_shifted = force_for_current_correction()
    torch.testing.assert_close(
        force_shifted,
        force_zero,
        atol=1.0e-11,
        rtol=1.0e-11,
    )

    model.zero_grad(set_to_none=True)
    model(*graph).sum().backward()
    correction_grad = model.element_energy_correction.grad
    assert correction_grad is not None
    assert torch.isfinite(correction_grad).all()
    assert correction_grad.abs().sum() > 0

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert torch.isfinite(forces).all()


def test_train_only_closed_form_element_energy_calibration():
    class ToyElementEnergyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atomic_numbers = (1, 6)
            self.num_elements = 2
            self.element_energy_correction = torch.nn.Parameter(
                torch.zeros(2, dtype=torch.float64, device=DEVICE)
            )
            lookup = torch.full((7,), -1, dtype=torch.long, device=DEVICE)
            lookup[1] = 0
            lookup[6] = 1
            self.register_buffer(
                "atomic_number_to_index", lookup, persistent=False
            )

        def forward(
            self,
            pos,
            A,
            batch_idx,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
        ):
            del pos, batch_idx, edge_src, edge_dst, edge_shifts, cell
            compact = self.atomic_number_to_index[A]
            return self.element_energy_correction[compact].unsqueeze(-1)

    model = ToyElementEnergyModel()
    pos = torch.zeros(6, 3, dtype=torch.float64)
    atomic_numbers = torch.tensor([1, 1, 6, 6, 1, 6], dtype=torch.long)
    batch_idx = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    force = torch.zeros_like(pos)
    # Desired per-element corrections are H=+0.3 and C=-0.2 eV.
    target_energy = torch.tensor([0.6, -0.4, 0.1], dtype=torch.float64)
    empty_edges = torch.empty(0, dtype=torch.long)
    batch = (
        pos,
        atomic_numbers,
        batch_idx,
        force,
        target_energy,
        empty_edges,
        empty_edges,
        torch.empty(0, 3, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64).repeat(3, 1, 1),
        torch.zeros(3, 3, 3, dtype=torch.float64),
    )
    trainer = ForceTrainer(
        model,
        train_loader=[batch],
        device=DEVICE,
        dtype=torch.float64,
        atomic_energy_keys=[1, 6],
        atomic_energy_values=[0.0, 0.0],
        lr_scheduler="none",
    )
    report = trainer.fit_element_energy_correction_from_training_set()
    torch.testing.assert_close(
        model.element_energy_correction,
        torch.tensor([0.3, -0.2], dtype=torch.float64, device=DEVICE),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert report["normal_matrix_rank"] == 2
    assert report["num_structures"] == 3


def test_scalar_ffn_identity_initialization_gradients_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-nonlinear",
        scalar_ffn=True,
    ).train()
    assert model.scalar_ffns is not None
    probe = torch.randn(
        5,
        model.channels * (model.lmax + 1) ** 2,
        dtype=torch.float64,
        device=DEVICE,
    )
    torch.testing.assert_close(
        model.scalar_ffns[0](probe),
        probe,
        atol=0.0,
        rtol=0.0,
    )

    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=109,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    energy.sum().backward()
    for ffn in model.scalar_ffns:
        grad = ffn.linear_2.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().max().item() > 0.0

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert torch.isfinite(forces).all()


def test_full_nonlinear_direct_readout_gradients_and_makefx():
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
        phase_density_pairs="full-nonlinear-readout",
        phase_normalization="avg-neighbors",
    ).train()
    assert set(model.phase_direct_readouts) == {"1"}
    assert set(model.phase_direct_readout_scales) == {"1"}
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=103,
    )
    model.zero_grad(set_to_none=True)
    energy = model(*graph)
    assert torch.isfinite(energy).all()
    energy.sum().backward()
    readout = model.phase_direct_readouts["1"]
    assert readout.linear_2.weight.grad is not None
    assert readout.linear_2.weight.grad.abs().max().item() > 0.0
    scale = model.phase_direct_readout_scales["1"]
    assert scale.grad is not None
    assert scale.grad.abs().item() > 0.0

    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    traced_energy, forces = graph_module(*graph)
    assert torch.isfinite(traced_energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()
