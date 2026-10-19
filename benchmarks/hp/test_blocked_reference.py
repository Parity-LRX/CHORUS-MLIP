"""Dependency-halo assembly and rotation-invariant error-gate qualification."""
import pytest
import torch

from blocked_reference import blocked_reference
from profile_scaling import assert_outputs, reference_values
from chorus.synthetic import build_model, make_fixed_graph


@pytest.mark.parametrize("layers", [1, 2, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")
def test_dependency_halos_keep_every_centre_force_and_virial(layers):
    torch.manual_seed(20261017)
    model = build_model(
        channels=8, lmax=1, num_interaction=layers, route="baseline",
        product_backend="ictd-bridge-u", dtype=torch.float32, device=torch.device("cuda"),
        correlation=2, ictd_fix_phase_mode="final-full-l-residual",
        ictd_fix_phase_amplitude="softplus", ictd_fix_phase_placement="pre-product-full-l",
        ictd_fix_phase_density_pairs="full-nonlinear", ictd_fix_phase_density_rank=4,
        ictd_fix_phase_scope="final" if layers == 1 else "persistent").eval().requires_grad_(False)
    graph = make_fixed_graph(num_nodes=83, avg_degree=3, dtype=torch.float32,
                             device=torch.device("cuda"), seed=73)
    graph = (graph[0]*.3,)+graph[1:]
    expected = reference_values(model, graph, True)
    actual = blocked_reference(model, graph, virial=True, centres=7)
    for got, ref in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-4)
    assert_outputs(actual, expected, 83, virial=True)


def test_virial_error_gate_is_rotation_invariant_and_rejects_large_errors():
    v = torch.diag(torch.tensor([100., -80., 30.], dtype=torch.float64))[None]
    delta = torch.tensor([[[.0001, .0002, 0.], [.0002, -.0001, 0.], [0., 0., 0.]]], dtype=torch.float64)
    angle = torch.tensor(.37, dtype=torch.float64)
    co, si = angle.cos(), angle.sin()
    rotation = torch.stack([torch.stack([co, -si, co*0]), torch.stack([si, co, co*0]),
                            torch.tensor([0., 0., 1.], dtype=torch.float64)])
    intensive = (torch.zeros(1), torch.zeros(1, 3), torch.zeros(1, 1))
    for ref, difference in ((v, delta), (rotation.T@v@rotation, rotation.T@delta@rotation)):
        assert_outputs(intensive+(ref+difference,), intensive+(ref,), 100, virial=True)
        with pytest.raises(AssertionError, match="Frobenius"):
            assert_outputs(intensive+(ref+100*difference,), intensive+(ref,), 100, virial=True)
