"""Remote-only CUDA graph live-input and invalidation qualification."""
import pytest
import torch
from chorus.hp import capture_inference, energy_forces_virial
from chorus.synthetic import build_model, make_fixed_graph
from mace_ictc.hp import energy_and_forces


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("scope", ["final", "persistent"])
@pytest.mark.parametrize("virial", [False, True])
@pytest.mark.parametrize("radial_mode", ["off", "tiles", "jvp", "jvp_tiles", "jvp_destination", "jvp_cached"])
def test_capture_live_inputs_and_state_invalidation(scope, virial, radial_mode):
    torch.manual_seed(20261017)
    model = build_model(
        channels=8, lmax=1, num_interaction=2, route="baseline",
        product_backend="ictd-bridge-u", dtype=torch.float32, device=torch.device("cuda"),
        correlation=2, ictd_fix_freeze_conv_tp_weight=True,
        ictd_fix_phase_mode="final-full-l-residual", ictd_fix_phase_amplitude="softplus",
        ictd_fix_phase_placement="pre-product-full-l", ictd_fix_phase_density_rank=4,
        ictd_fix_phase_density_pairs="full", ictd_fix_phase_scope=scope,
    ).eval().requires_grad_(False)
    graph = make_fixed_graph(num_nodes=19, avg_degree=6, dtype=torch.float32,
                             device=torch.device("cuda"), seed=718)
    graph = (graph[0]*0.2,) + graph[1:]
    reference = energy_and_forces(model, *graph)
    if virial:
        strain = torch.zeros_like(graph[6], dtype=torch.float64, requires_grad=True)
        deformation = torch.eye(3, device="cuda", dtype=torch.float64)[None]+strain
        deformed = (torch.einsum("ni,nij->nj", graph[0].double(), deformation[graph[2]]),
                    *graph[1:6], graph[6].double()@deformation)
        atom = model(*deformed)
        reference["virial"] = -torch.autograd.grad(atom.sum(), strain)[0]
    captured = capture_inference(model, graph, clone_outputs=True, virial=virial,
                                 radial_fusion=radial_mode == "tiles", radial_tile_edges=13,
                                 radial_gate_pipeline=radial_mode in ("tiles", "jvp_tiles", "jvp_destination", "jvp_cached"),
                                 radial_jvp=radial_mode in ("jvp", "jvp_tiles", "jvp_destination", "jvp_cached"),
                                 radial_destination=radial_mode in ("jvp_destination", "jvp_cached"),
                                 radial_cache_gates=radial_mode == "jvp_cached")
    assert not getattr(model, "skip_input_validation", False)
    assert not getattr(model, "preserve_edge_order", False)
    original = captured(*graph)
    for key in reference:
        torch.testing.assert_close(original[key], reference[key], atol=4e-6, rtol=5e-4)
    changed = tuple(v.clone() for v in graph)
    changed[0].add_(torch.arange(57, device="cuda").reshape(19, 3)*1e-5)
    changed[1].copy_(changed[1].roll(1))
    changed[4].add_(1).remainder_(19)
    changed[4].copy_(torch.where(changed[4] == changed[3], (changed[4]+1) % 19, changed[4]))
    changed[6].mul_(1.01)
    got = captured(*changed)
    expected = (energy_forces_virial if virial else energy_and_forces)(model, *changed)
    for key in expected:
        torch.testing.assert_close(got[key], expected[key], atol=5e-6, rtol=5e-4)
    assert not torch.equal(got["forces"], original["forces"])
    with pytest.raises(ValueError, match="N/E/B changed"):
        captured(*(changed[:3]+(changed[3][:-1], changed[4][:-1], changed[5][:-1], changed[6])))
    layer = model.interactions[-1]
    old_control = layer.phase_intervention
    layer.phase_intervention = "common_shift"
    with pytest.raises(RuntimeError, match="Model state"):
        captured(*changed)
    layer.phase_intervention = old_control
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    with pytest.raises(RuntimeError, match="Model state"):
        captured(*changed)
    captured.close()
    with pytest.raises(RuntimeError, match="closed"):
        captured(*graph)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize('fused_geometry', [False, True])
def test_virial_multiple_structures_and_empty_edges(empty, fused_geometry):
    from chorus.test.test_hp_backend import _build
    from chorus.hp import configure_model
    torch.manual_seed(20261017)
    model = _build("persistent").eval().requires_grad_(False)
    base = make_fixed_graph(num_nodes=11, avg_degree=4, dtype=torch.float32,
                            device=torch.device("cuda"), seed=918)
    pos, z, batch, src, dst, shifts, cell = base
    if empty:
        src, dst, shifts = src[:0], dst[:0], shifts[:0]
    graph = (torch.cat((pos*0.2, pos*0.2+10), 0), z.repeat(2),
             torch.cat((batch, batch+1)), torch.cat((src, src+11)),
             torch.cat((dst, dst+11)), shifts.repeat(2, 1), cell.repeat(2, 1, 1))
    reference = energy_and_forces(model, *graph)
    strain = torch.zeros_like(graph[6], dtype=torch.float64, requires_grad=True)
    deformation = torch.eye(3, device="cuda", dtype=torch.float64)[None]+strain
    atom = model(torch.einsum("ni,nij->nj", graph[0].double(), deformation[graph[2]]),
                 *graph[1:6], graph[6].double()@deformation)
    reference["virial"] = -torch.autograd.grad(atom.sum()+strain.sum()*0.0, strain)[0]
    configure_model(model, policy="generated", topology_backend="device")
    actual = energy_forces_virial(model, *graph, fused_geometry=fused_geometry)
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], atol=7e-6, rtol=5e-4)
