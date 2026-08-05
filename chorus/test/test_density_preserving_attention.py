"""Tests for the scope-matched density-preserving attention control."""

from __future__ import annotations

import torch

import chorus.models.pure_cartesian_ictd_fix as pure_cartesian_ictd_fix
from chorus.cli.train import build_baseline_model
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from chorus.synthetic import build_model, make_fixed_graph
from chorus.training.train_loop import ForceTrainer
from chorus.training.makefx_compile import trace_and_compile_force
from chorus.utils.config import ModelConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build(*, attn_heads: int, attn_mode: str, attn_scope: str):
    return build_model(
        channels=8,
        lmax=1,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-pure-u",
        dtype=torch.float64,
        device=DEVICE,
        correlation=2,
        attn_heads=attn_heads,
        ictd_fix_interaction_attn_mode=attn_mode,
        ictd_fix_interaction_attn_scope=attn_scope,
    ).to(device=DEVICE, dtype=torch.float64)


def _copy_common_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value
    target.load_state_dict(target_state, strict=True)


def _build_chorus(*, attn_heads: int):
    return build_model(
        channels=8,
        lmax=1,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-pure-u",
        dtype=torch.float64,
        device=DEVICE,
        correlation=2,
        attn_heads=attn_heads,
        ictd_fix_interaction_attn_mode="density-preserving",
        ictd_fix_interaction_attn_scope="final",
        ictd_fix_phase_mode="final-full-l-residual",
        ictd_fix_phase_amplitude="softplus",
        ictd_fix_phase_coefficient="polar",
        ictd_fix_phase_context="content",
        ictd_fix_phase_placement="pre-product-full-l",
        ictd_fix_phase_density_rank=4,
        ictd_fix_phase_density_pairs="full-nonlinear",
        ictd_fix_phase_normalization="avg-neighbors",
        ictd_fix_phase_scope="final",
    ).to(device=DEVICE, dtype=torch.float64)


def test_density_preserving_final_attention_starts_at_exact_baseline():
    torch.manual_seed(101)
    baseline = _build(
        attn_heads=0,
        attn_mode="legacy-softmax",
        attn_scope="all",
    ).eval()
    torch.manual_seed(102)
    attention = _build(
        attn_heads=2,
        attn_mode="density-preserving",
        attn_scope="final",
    ).eval()
    _copy_common_state(baseline, attention)

    assert attention.interactions[0].interaction_attn_heads == 0
    assert attention.interactions[1].interaction_attn_heads == 2
    assert attention.interactions[1].attn_z_bias_raw is None
    torch.testing.assert_close(
        attention.interactions[1].attn_logit_w,
        torch.zeros_like(attention.interactions[1].attn_logit_w),
        atol=0.0,
        rtol=0.0,
    )

    graph = make_fixed_graph(
        num_nodes=9,
        avg_degree=5,
        dtype=torch.float64,
        device=DEVICE,
        seed=103,
    )
    with torch.no_grad():
        torch.testing.assert_close(
            attention(*graph),
            baseline(*graph),
            atol=2.0e-12,
            rtol=2.0e-12,
        )


def test_density_preserving_attention_starts_at_exact_chorus():
    torch.manual_seed(107)
    chorus = _build_chorus(attn_heads=0).eval()
    torch.manual_seed(108)
    chorus_attention = _build_chorus(attn_heads=2).eval()
    _copy_common_state(chorus, chorus_attention)

    final_interaction = chorus_attention.interactions[-1]
    assert final_interaction.phase_enabled
    assert final_interaction.interaction_attn_heads == 2
    torch.testing.assert_close(
        final_interaction.attn_logit_w,
        torch.zeros_like(final_interaction.attn_logit_w),
        atol=0.0,
        rtol=0.0,
    )

    graph = make_fixed_graph(
        num_nodes=9,
        avg_degree=5,
        dtype=torch.float64,
        device=DEVICE,
        seed=109,
    )
    with torch.no_grad():
        torch.testing.assert_close(
            chorus_attention(*graph),
            chorus(*graph),
            atol=2.0e-12,
            rtol=2.0e-12,
        )


def test_chorus_density_attention_makefx_forward_force_and_gradients():
    model = _build_chorus(attn_heads=2).train()
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=110,
    )
    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    energy, forces = graph_module(*graph)
    loss = energy.square() + forces.square().mean()
    loss.backward()

    final_interaction = model.interactions[-1]
    assert torch.isfinite(energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()
    assert final_interaction.attn_logit_w.grad is not None
    assert torch.isfinite(final_interaction.attn_logit_w.grad).all()
    assert final_interaction.attn_logit_w.grad.abs().max().item() > 0.0


def test_density_preserving_gate_keeps_cutoff_weighted_density():
    torch.manual_seed(104)
    model = _build(
        attn_heads=2,
        attn_mode="density-preserving",
        attn_scope="final",
    )
    interaction = model.interactions[-1]
    num_nodes = 5
    edge_src = torch.tensor(
        [1, 2, 3, 0, 2, 4, 0, 1, 4, 1, 2],
        dtype=torch.long,
        device=DEVICE,
    )
    edge_dst = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4],
        dtype=torch.long,
        device=DEVICE,
    )
    node_scalars = torch.randn(
        num_nodes,
        interaction.channels,
        dtype=torch.float64,
        device=DEVICE,
    )
    edge_feats = torch.randn(
        edge_src.numel(),
        interaction.number_of_basis,
        dtype=torch.float64,
        device=DEVICE,
    )
    edge_env = torch.tensor(
        [1.0, 0.8, 0.3, 0.9, 0.7, 0.2, 1.0, 0.6, 0.1, 0.5, 0.4],
        dtype=torch.float64,
        device=DEVICE,
    )

    initial_gate = interaction._attention_weight(
        node_scalars,
        edge_feats,
        edge_src,
        edge_dst,
        edge_env,
        num_nodes,
    )
    torch.testing.assert_close(
        initial_gate,
        torch.ones_like(initial_gate),
        atol=2.0e-15,
        rtol=2.0e-15,
    )

    with torch.no_grad():
        interaction.attn_logit_w.normal_(mean=0.0, std=0.2)
        interaction.attn_radial_bias.weight.normal_(mean=0.0, std=0.1)
    gate = interaction._attention_weight(
        node_scalars,
        edge_feats,
        edge_src,
        edge_dst,
        edge_env,
        num_nodes,
    )
    heads = interaction.interaction_attn_heads
    weighted_gate = pure_cartesian_ictd_fix.scatter(
        edge_env[:, None] * gate,
        edge_dst,
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    )
    cutoff_density = pure_cartesian_ictd_fix.scatter(
        edge_env[:, None].expand(-1, heads),
        edge_dst,
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    )
    torch.testing.assert_close(
        weighted_gate,
        cutoff_density,
        atol=4.0e-13,
        rtol=4.0e-13,
    )
    assert torch.all(gate > 0)

    interaction.zero_grad(set_to_none=True)
    edge_value = torch.randn_like(gate)
    (gate * edge_value).sum().backward()
    assert interaction.attn_logit_w.grad is not None
    assert interaction.attn_logit_w.grad.abs().max().item() > 0.0
    assert interaction.attn_radial_bias.weight.grad is not None
    assert interaction.attn_radial_bias.weight.grad.abs().max().item() > 0.0


def test_density_preserving_attention_makefx_forward_force():
    model = _build(
        attn_heads=2,
        attn_mode="density-preserving",
        attn_scope="final",
    ).train()
    graph = make_fixed_graph(
        num_nodes=7,
        avg_degree=4,
        dtype=torch.float64,
        device=DEVICE,
        seed=105,
    )
    graph_module = trace_and_compile_force(
        model,
        graph,
        training=True,
        do_compile=False,
    )
    energy, forces = graph_module(*graph)
    assert torch.isfinite(energy)
    assert forces.shape == (7, 3)
    assert torch.isfinite(forces).all()


def test_density_preserving_attention_checkpoint_round_trip(tmp_path):
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
        attn_heads=2,
        attn_mode="density-preserving",
        attn_scope="final",
        atomic_numbers=[1, 6, 7, 8],
        ictd_save_tp_mode="fully-connected",
        invariant_channels=8,
        device=DEVICE,
        dtype=dtype,
    )
    extra_hparams = {
        "num_interaction": 2,
        "invariant_channels": 8,
        "ictd_fix_route": "baseline",
        "ictd_fix_product_backend": "ictd-pure-u",
        "ictd_fix_first_layer_self_connection": True,
        "ictd_fix_readout_hidden_channels": 16,
        "ictd_fix_edge_lmax": 1,
        "save_contraction_order": 2,
        "ictd_save_tp_mode": "fully-connected",
        "ictd_fix_interaction_attn_heads": 2,
        "ictd_fix_interaction_attn_mode": "density-preserving",
        "ictd_fix_interaction_attn_scope": "final",
        "ictd_fix_phase_mode": "none",
        "radial_sqrt_num_basis": False,
        "polynomial_cutoff_p": 5,
        "avg_num_neighbors": 4.0,
    }
    trainer = ForceTrainer(
        model,
        [],
        device=DEVICE,
        config=cfg,
        dtype=dtype,
        max_radius=5.0,
        epochs=1,
        extra_hparams=extra_hparams,
    )
    checkpoint = tmp_path / "density_attention.pth"
    trainer.save_checkpoint(str(checkpoint), epoch=0)
    deployed = LAMMPS_MLIAP_MFF.from_checkpoint(
        str(checkpoint),
        element_types=["H", "C", "N", "O"],
        device=str(DEVICE),
        atomic_energy_keys=[1, 6, 7, 8],
        atomic_energy_values=[0.0, 0.0, 0.0, 0.0],
    ).wrapper.model
    assert deployed.ictd_fix_interaction_attn_heads == 2
    assert deployed.ictd_fix_interaction_attn_mode == "density-preserving"
    assert deployed.ictd_fix_interaction_attn_scope == "final"
    assert deployed.interactions[0].interaction_attn_heads == 0
    assert deployed.interactions[1].interaction_attn_heads == 2

    graph = make_fixed_graph(
        num_nodes=8,
        avg_degree=4,
        dtype=dtype,
        device=DEVICE,
        seed=106,
    )
    model.eval()
    deployed.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(*graph),
            deployed(*graph),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
