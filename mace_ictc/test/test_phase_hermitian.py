"""Tests for the PEMP real-doublet/Hermitian residual operator."""

from __future__ import annotations

import pytest
import torch

from mace_ictc.cli.train import build_baseline_model
from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from mace_ictc.models.pure_cartesian_ictd_fix import PhaseHermitianScalarResidual
from mace_ictc.synthetic import (
    build_model,
    compute_energy_forces,
    make_fixed_graph,
    random_rotation,
)
from mace_ictc.training.train_loop import ForceTrainer
from mace_ictc.training.makefx_compile import trace_and_compile_force
from mace_ictc.utils.config import ModelConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build(*, phase_mode: str = "none", phase_amplitude: str = "unit"):
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
    torch.testing.assert_close(rho_rot, rho, atol=2.0e-14, rtol=2.0e-14)

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
    expected = sum(
        b[j] * b[k] * torch.cos(theta[j] - theta[k])
        for j in range(3)
        for k in range(3)
    )
    torch.testing.assert_close(got, expected, atol=1.0e-14, rtol=1.0e-14)


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
def test_phase_model_rotation_force_covariance_and_gradients(amplitude: str):
    torch.manual_seed(11)
    model = _build(phase_mode="final-scalar-residual", phase_amplitude=amplitude)
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


def test_phase_checkpoint_strict_deployment_round_trip(tmp_path):
    dtype = torch.float64
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
        phase_mode="final-scalar-residual",
        phase_hidden_channels=12,
        phase_residual_scale_init=0.05,
        phase_amplitude="softplus",
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
            "ictd_fix_phase_mode": "final-scalar-residual",
            "ictd_fix_phase_hidden_channels": 12,
            "ictd_fix_phase_residual_scale_init": 0.05,
            "ictd_fix_phase_amplitude": "softplus",
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
    assert deployed.ictd_fix_phase_mode == "final-scalar-residual"
    assert deployed.ictd_fix_phase_hidden_channels == 12
    assert deployed.ictd_fix_phase_amplitude == "softplus"

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


def test_phase_force_graph_makefx_trace():
    model = _build(phase_mode="final-scalar-residual").train()
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
