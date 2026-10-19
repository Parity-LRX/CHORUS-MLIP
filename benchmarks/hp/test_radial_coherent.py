"""Remote-only radial linear/three-stream TP boundary qualification."""
import pytest
import torch
from mace_ictc.hp import CoherentGraphMessage, EdgeTensorProduct

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("channels,lmax,heads,angular,edges,k", [
    (8, 1, 1, True, 47, 16), (12, 2, 2, True, 47, 33),
    (13, 1, 1, False, 47, 64), (8, 1, 1, True, 0, 16),
    (128, 2, 1, True, 37, 64),
])
@pytest.mark.parametrize("tile_edges", [13, 4096])
@pytest.mark.parametrize("gate_pipeline", [False, True])
def test_radial_boundary_values_first_mixed_live_bank(channels, lmax, heads, angular, edges, k, tile_edges, gate_pipeline):
    torch.manual_seed(20261017)
    edge = EdgeTensorProduct.create(channels=channels, input_lmax=lmax,
                                    edge_lmax=lmax, device="cuda")
    with torch.no_grad():
        edge.tp.weight.uniform_(0.7, 1.3)
    op = CoherentGraphMessage(edge.tp, input_lmax=lmax, edge_lmax=lmax,
                              heads=heads, angular=angular)
    n = 11
    x = {l: (torch.randn(n, channels, 2*l+1, device="cuda")*0.2).requires_grad_()
         for l in range(lmax+1)}
    h = {l: (torch.randn(edges, 1, 2*l+1, device="cuda")*0.2).requires_grad_()
         for l in range(lmax+1)}
    z = (torch.randn(edges, k, device="cuda")*0.2).requires_grad_()
    # Noncontiguous frozen banks are legal and must read their live values.
    w = torch.randn(k, edge.gate_width, device="cuda").T/k**0.5
    phases = heads*(lmax+1 if angular else 1)
    r = torch.randn(edges, phases, 1, device="cuda", requires_grad=True)
    i = torch.randn(edges, phases, 1, device="cuda", requires_grad=True)
    src, dst = (torch.randint(n, (edges,), device="cuda") for _ in range(2))
    mask = (torch.arange(edges, device="cuda") % 4 != 0).float()
    probes = {l: torch.randn(n, 3, edge.tp.path_counts_by_l[l]*channels, 2*l+1,
                             device="cuda") for l in range(lmax+1)}

    def calculate(fused):
        if fused:
            return op.radial(x, h, z, w, src, dst, mask, r, i, tile_edges=tile_edges,
                              gate_pipeline=gate_pipeline)
        return op(x, h, torch.nn.functional.linear(z, w), src, dst, mask, r, i)

    def loss(value):
        return sum((value[l]*probes[l]).sum() for l in value)

    active = [*x.values(), *h.values(), z, r, i]
    expected, actual = calculate(False), calculate(True)
    for l in expected:
        torch.testing.assert_close(actual[l], expected[l], atol=2e-6, rtol=3e-5)
    ref = torch.autograd.grad(loss(expected), active, create_graph=True, retain_graph=True)
    got = torch.autograd.grad(loss(actual), active)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, atol=4e-6, rtol=1e-4)
    if edges:
        higher = torch.autograd.grad(loss(calculate(True)), active, create_graph=True)
        tangent = [torch.randn_like(v) for v in active]
        ref2 = torch.autograd.grad(sum((a*t).sum() for a, t in zip(ref, tangent)), active,
                                   allow_unused=True)
        got2 = torch.autograd.grad(sum((a*t).sum() for a, t in zip(higher, tangent)), active,
                                   allow_unused=True)
        for a, b in zip(got2, ref2):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, atol=8e-6, rtol=2e-4)
    with torch.no_grad():
        w.add_(0.021)
        edge.tp.weight.add_(0.017)
    x = {l: v.detach() for l, v in x.items()}
    active = [*h.values(), z, r, i]
    ref = torch.autograd.grad(loss(calculate(False)), active)
    got = torch.autograd.grad(loss(calculate(True)), active)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, atol=4e-6, rtol=1e-4)


def test_trainable_radial_bank_uses_original_model_path():
    from chorus.hp import configure_model
    from chorus.test.test_hp_backend import _build
    model = _build("persistent").eval()
    info = configure_model(model, policy="generated", radial_fusion=True, radial_jvp=True)
    assert info["counts"]["radial_jvp_messages"] == 0
    for layer in model.interactions:
        assert layer.fc[-1].weight.requires_grad
        features = torch.randn(9, layer.number_of_basis, device="cuda")
        value = layer._hp_message.radial_inputs(features, layer.fc)
        assert isinstance(value, torch.Tensor)
        torch.testing.assert_close(value, layer.fc(features))
