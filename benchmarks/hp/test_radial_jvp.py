"""Remote-only radial directional derivatives and scalar TP/VJP boundaries."""
import pytest
import torch
from chorus.hp.radial_jvp import RadialJVP
from chorus.models.pure_cartesian_ictd_fix import _Normalize2MomSiLU
from chorus.models.radial_basis import mace_radial_embedding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


def test_destination_configuration_requires_bounded_directional_pipeline():
    from chorus.hp import configure_model
    from chorus.test.test_hp_backend import _build
    model = _build('persistent').eval().requires_grad_(False)
    with pytest.raises(ValueError, match='requires radial_jvp and radial_gate_pipeline'):
        configure_model(model, radial_destination=True)
    with pytest.raises(TypeError, match='booleans'):
        configure_model(model, radial_destination=1)
    before = {name: id(value) for name, value in model.named_parameters()}
    keys = tuple(model.state_dict())
    plan = configure_model(model, policy='generated', radial_jvp=True,
        radial_gate_pipeline=True, radial_destination=True)
    assert plan['radial_destination'] and plan['counts']['radial_destination_messages'] > 0
    assert before == {name: id(value) for name, value in model.named_parameters()}
    assert keys == tuple(model.state_dict())
    with pytest.raises(ValueError, match='radial_cache_gates requires'):
        configure_model(model, radial_cache_gates=True)
    plan = configure_model(model, policy='generated', radial_jvp=True,
        radial_gate_pipeline=True, radial_destination=True, radial_cache_gates=True)
    assert plan['radial_cache_gates'] and plan['counts']['radial_cached_gate_messages'] == 1
    assert plan['counts']['radial_jvp_messages'] == 1
    assert not model.interactions[0]._hp_message.radial_jvp
    assert before == {name: id(value) for name, value in model.named_parameters()}
    assert keys == tuple(model.state_dict())


def _plan(width, kind="bessel", cutoff=6, normalize=False):
    layers = []
    for ci, co in ((8, 64), (64, 64), (64, 64)):
        linear = torch.nn.Linear(ci, co, bias=False, device="cuda")
        torch.nn.init.normal_(linear.weight, std=ci**-0.5)
        layers.extend((linear, _Normalize2MomSiLU()))
    layers.append(torch.nn.Linear(64, width, bias=False, device="cuda"))
    fc = torch.nn.Sequential(*layers).eval().requires_grad_(False)
    return RadialJVP(fc, r_max=5., number_of_basis=8, function_type=kind,
        polynomial_cutoff_p=cutoff, sqrt_num_basis_norm=normalize)


@pytest.mark.parametrize("kind", ["bessel", "gaussian"])
@pytest.mark.parametrize("cutoff", [None, 6])
@pytest.mark.parametrize("normalize", [False, True])
def test_radial_jvp_values_and_direction(kind, cutoff, normalize):
    torch.manual_seed(20261017)
    plan = _plan(24, kind, cutoff, normalize)
    r = torch.tensor([0.15, 0.2, 1.1, 2.3, 4.4, 4.99, 5., 5.2], device="cuda")
    expected, tangent = torch.autograd.functional.jvp(
        lambda value: plan.fc[:-1](mace_radial_embedding(value, **plan.options)), r, torch.ones_like(r))
    with torch.no_grad():
        actual, derivative = plan.evaluate(r)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(derivative, tangent, atol=3e-5, rtol=1e-4)


@pytest.mark.parametrize("channels,input_lmax,edges,heads,angular", [
    (8, 1, 47, 1, True), (12, 2, 47, 2, True), (13, 1, 47, 1, False),
    (128, 2, 37, 1, True), (128, 0, 37, 1, True), (8, 1, 0, 1, True),
    (64, 3, 47, 4, True),
])
@pytest.mark.parametrize('gate_pipeline,destination,projection', [
    (False, False, 'gemm'), (True, False, 'gemm'), (True, True, 'gemm'),
    (True, True, 'forward'), (True, True, 'both'), (True, True, 'cached')])
def test_distance_coherent_values_first_mixed(channels, input_lmax, edges, heads, angular, gate_pipeline, destination, projection):
    from mace_ictc.hp import EdgeTensorProduct, CoherentGraphMessage
    torch.manual_seed(20261017)
    lmax = max(1, input_lmax)
    edge = EdgeTensorProduct.create(channels=channels, input_lmax=input_lmax,
                                    edge_lmax=lmax, output_lmax=lmax, device="cuda")
    op = CoherentGraphMessage(edge.tp, input_lmax=input_lmax, edge_lmax=lmax,
                               heads=heads, angular=angular)
    op.distance_destination = destination
    op.distance_projection = 'gemm' if projection == 'cached' else projection
    op.distance_cached_gates = projection == 'cached'
    plan = _plan(edge.gate_width)
    n = 11
    x = {l: (torch.randn(n, channels, 2*l+1, device="cuda")*0.2).requires_grad_(input_lmax>0)
         for l in range(input_lmax+1)}
    h = {l: (torch.randn(edges, 1, 2*l+1, device="cuda")*0.2).requires_grad_()
         for l in range(lmax+1)}
    distance = (0.8+torch.rand(edges, device="cuda")*3.).requires_grad_()
    phases = heads*(lmax+1 if angular else 1)
    r = torch.randn(edges, phases, 1, device="cuda", requires_grad=True)
    i = torch.randn(edges, phases, 1, device="cuda", requires_grad=True)
    src, dst = (torch.randint(n, (edges,), device="cuda") for _ in range(2))
    mask = (torch.arange(edges, device="cuda") % 4 != 0).float()
    probes = {l: torch.randn(n, 3, edge.tp.path_counts_by_l[l]*channels, 2*l+1, device="cuda")
              for l in range(lmax+1)}
    active = [v for v in [*x.values(), *h.values(), distance, r, i] if v.requires_grad]

    def calculate(fused):
        if fused:
            return op.distance(x, h, distance, plan, src, dst, mask, r, i, tile_edges=13,
                               gate_pipeline=gate_pipeline)
        return op(x, h, plan.reference(distance), src, dst, mask, r, i)

    def loss(value):
        return sum((value[l]*probes[l]).sum() for l in value)

    expected, actual = calculate(False), calculate(True)
    for l in expected:
        torch.testing.assert_close(actual[l], expected[l], atol=3e-6, rtol=3e-5)
    ref = torch.autograd.grad(loss(expected), active, create_graph=True, retain_graph=True)
    got = torch.autograd.grad(loss(actual), active)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, atol=6e-6, rtol=2e-4)
    if edges:
        higher = torch.autograd.grad(loss(calculate(True)), active, create_graph=True)
        tangent = [torch.randn_like(v) for v in active]
        ref2 = torch.autograd.grad(sum((a*t).sum() for a, t in zip(ref, tangent)), active, allow_unused=True)
        got2 = torch.autograd.grad(sum((a*t).sum() for a, t in zip(higher, tangent)), active, allow_unused=True)
        for a, b in zip(got2, ref2):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, atol=8e-6, rtol=3e-4)
    with torch.no_grad():
        edge.tp.weight.add_(0.017)
        plan.fc[2].weight.add_(0.003)
        plan.fc[-1].weight.add_(0.007)
    ref = torch.autograd.grad(loss(calculate(False)), active)
    got = torch.autograd.grad(loss(calculate(True)), active)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, atol=7e-6, rtol=2e-4)


@pytest.mark.parametrize('input_lmax', [0, 1])
def test_cached_distance_with_fixed_phase_coefficients(input_lmax):
    """Partial AD targets retain the literal scalar-radial fallback."""
    from mace_ictc.hp import EdgeTensorProduct, CoherentGraphMessage
    torch.manual_seed(20261017)
    n, e, c = 7, 23, 13
    edge = EdgeTensorProduct.create(channels=c, input_lmax=input_lmax,
                                    edge_lmax=1, output_lmax=1, device='cuda')
    op = CoherentGraphMessage(edge.tp, input_lmax=input_lmax, edge_lmax=1)
    op.distance_destination, op.distance_cached_gates = True, True
    plan = _plan(edge.gate_width)
    x = {l: (torch.randn(n, c, 2*l+1, device='cuda')*0.2).requires_grad_(input_lmax > 0)
         for l in range(input_lmax+1)}
    h = {l: (torch.randn(e, 1, 2*l+1, device='cuda')*0.2).requires_grad_() for l in range(2)}
    distance = (1.+torch.rand(e, device='cuda')).requires_grad_()
    real, imag = (torch.randn(e, 2, 1, device='cuda') for _ in range(2))
    src, dst = (torch.randint(n, (e,), device='cuda') for _ in range(2))
    active = [value for value in (*x.values(), *h.values(), distance) if value.requires_grad]
    probes = {l: torch.randn(n, 3, edge.tp.path_counts_by_l[l]*c, 2*l+1, device='cuda')
              for l in range(2)}
    def loss(value):return sum((value[l]*probes[l]).sum() for l in value)
    expected = op(x, h, plan.reference(distance), src, dst, None, real, imag)
    actual = op.distance(x, h, distance, plan, src, dst, None, real, imag,
                         tile_edges=11, gate_pipeline=True)
    for got, ref in zip(torch.autograd.grad(loss(actual), active),
                        torch.autograd.grad(loss(expected), active)):
        torch.testing.assert_close(got, ref, atol=7e-6, rtol=2e-4)
