"""Tests for the PEMP real-doublet/Hermitian residual operator."""

from __future__ import annotations

import pytest
import torch

import mace_ictc.models.pure_cartesian_ictd_fix as pure_cartesian_ictd_fix
from mace_ictc.cli.train import build_baseline_model
from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from mace_ictc.models.pure_cartesian_ictd_fix import (
    PhaseHermitianFullLResidual,
    PhaseHermitianScalarResidual,
    PersistentChargedUpdate,
)
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


def _build(
    *,
    phase_mode: str = "none",
    phase_amplitude: str = "unit",
    phase_placement: str = "post-product",
    phase_density_rank: int = 8,
    phase_scope: str = "final",
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
        ictd_fix_phase_placement=phase_placement,
        ictd_fix_phase_density_rank=phase_density_rank,
        ictd_fix_phase_scope=phase_scope,
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


def test_full_l_phase_model_rotation_force_covariance_and_gradients():
    torch.manual_seed(47)
    model = _build(
        phase_mode="final-full-l-residual",
        phase_amplitude="softplus",
        phase_placement="pre-product-full-l",
        phase_density_rank=4,
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
    "phase_mode,phase_placement,phase_scope",
    [
        ("final-scalar-residual", "pre-product-l0", "final"),
        ("final-full-l-residual", "pre-product-full-l", "final"),
        ("final-scalar-residual", "pre-product-l0", "persistent"),
        ("final-full-l-residual", "pre-product-full-l", "persistent"),
    ],
)
def test_phase_checkpoint_strict_deployment_round_trip(
    tmp_path, phase_mode: str, phase_placement: str, phase_scope: str
):
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
        phase_mode=phase_mode,
        phase_hidden_channels=12,
        phase_residual_scale_init=0.05,
        phase_amplitude="softplus",
        phase_placement=phase_placement,
        phase_density_rank=4,
        phase_scope=phase_scope,
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
            "ictd_fix_phase_scope": phase_scope,
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
    assert deployed.ictd_fix_phase_scope == phase_scope

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
