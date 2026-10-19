"""Additional remote-only invariance, live-weight and grouped-phase checks."""
import pytest
import torch

from chorus.hp import configure_model
from chorus.synthetic import build_model, compute_energy_forces, make_fixed_graph


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("pairs,heads,normalization", [
    ("full-nonlinear", 2, "local-effective"),
    ("diagonal", 1, "avg-neighbors"),
])
def test_grouped_or_diagonal_live_weights_and_reflection(pairs, heads, normalization):
    torch.manual_seed(20261017)
    model = build_model(
        channels=8, lmax=1, num_interaction=2, route="baseline",
        product_backend="ictd-bridge-u", dtype=torch.float32, device=torch.device("cuda"),
        correlation=2, ictd_fix_freeze_conv_tp_weight=True,
        ictd_fix_phase_mode="final-full-l-residual", ictd_fix_phase_amplitude="softplus",
        ictd_fix_phase_placement="pre-product-full-l", ictd_fix_phase_density_rank=4,
        ictd_fix_phase_density_pairs=pairs, ictd_fix_phase_scope="persistent",
        ictd_fix_phase_heads=heads, ictd_fix_phase_normalization=normalization,
    )
    graph = make_fixed_graph(num_nodes=19, avg_degree=6, dtype=torch.float32,
                             device=torch.device("cuda"), seed=718)
    graph = (graph[0]*0.2,) + graph[1:]
    configure_model(model, training=True, policy="generated")
    # Update real learned weights and a fixed TP coefficient in place. The
    # optimized execution must use their current values, not its setup snapshot.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.001)
        model.interactions[0].tp.weight.add_(0.003)
    got = compute_energy_forces(model, graph, create_graph=False)
    reflection = torch.tensor([-1., 1., 1.], device="cuda")
    reflected = (graph[0]*reflection,) + graph[1:]
    mirrored = compute_energy_forces(model, reflected, create_graph=False)
    torch.testing.assert_close(mirrored[0], got[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(mirrored[1], got[1]*reflection, atol=2e-6, rtol=2e-5)
    configure_model(model, policy="reference")
    expected = compute_energy_forces(model, graph, create_graph=False)
    for actual, reference in zip(got, expected):
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=1e-4)


def test_cli_explicit_hp_flags():
    from chorus.cli.train import build_arg_parser
    args = build_arg_parser().parse_args([
        "--data-dir", "/unused", "--hp-operators", "generated", "--hp-layout", "packed",
        "--no-hp-stream-fusion", "--no-hp-density-fusion",
    ])
    assert args.hp_operators == "generated"
    assert args.hp_layout == "packed"
    assert not args.hp_stream_fusion and not args.hp_density_fusion


def test_reduced_cg_keeps_original_sc():
    from chorus.hp import check_model
    torch.manual_seed(20261017)
    model = build_model(
        channels=8, lmax=1, num_interaction=2, route="baseline",
        product_backend="ictd-bridge-u", dtype=torch.float32, device=torch.device("cuda"),
        correlation=2, ictd_fix_freeze_conv_tp_weight=True, ictd_fix_use_reduced_cg=True,
        ictd_fix_phase_mode="final-full-l-residual", ictd_fix_phase_amplitude="softplus",
        ictd_fix_phase_placement="pre-product-full-l", ictd_fix_phase_density_rank=4,
        ictd_fix_phase_density_pairs="full", ictd_fix_phase_scope="persistent",
    )
    graph = make_fixed_graph(num_nodes=19, avg_degree=6, dtype=torch.float32,
                             device=torch.device("cuda"), seed=718)
    graph = (graph[0]*0.2,) + graph[1:]
    reference = compute_energy_forces(model, graph, create_graph=False)
    assert not check_model(model)["supported"]
    assert check_model(model, symmetric_contraction=False)["supported"]
    info = configure_model(model, training=True, policy="generated", symmetric_contraction=False)
    assert info["counts"]["symmetric_contraction"] == 0
    actual = compute_energy_forces(model, graph, create_graph=False)
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, atol=3e-6, rtol=1e-4)
