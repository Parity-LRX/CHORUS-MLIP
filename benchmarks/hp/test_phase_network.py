"""Remote first/mixed derivative tests of the frozen joint phase MLP."""
import pytest
import torch
from chorus.hp.phase_network import _PhaseHeads, _native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='remote CUDA required')


@pytest.mark.parametrize('phases,edges', [(1, 47), (3, 53), (4, 0)])
@pytest.mark.parametrize('targets', [(0, 1, 2), (0,), (2,)])
def test_joint_phase_mlp_heads(phases, edges, targets):
    torch.manual_seed(20261017)
    n, h, k = 17, 32, 8
    inputs = [(torch.randn(n, h, device='cuda')*0.2).requires_grad_(),
              (torch.randn(n, h, device='cuda')*0.2).requires_grad_(),
              (torch.randn(edges, k, device='cuda')*0.2).requires_grad_()]
    weights = (torch.randn(h, k, device='cuda')*0.2, torch.randn(h, device='cuda')*0.2,
               torch.randn(h, h, device='cuda')*0.1, torch.randn(h, device='cuda')*0.1,
               torch.randn(16, h, device='cuda')*0.1, torch.randn(16, device='cuda')*0.1)
    src, dst = torch.randint(n, (edges,), device='cuda'), torch.randint(n, (edges,), device='cuda')
    scales = (1.6791767923989418,)*2
    probe = (torch.randn(edges, phases, device='cuda'), torch.randn(edges, phases, device='cuda'))
    def values(fused):
        return (_PhaseHeads.apply if fused else _native)(*inputs, *weights, src, dst, *scales, phases)
    def loss(value):return sum((v*p).sum() for v, p in zip(value, probe))
    actual, expected = values(True), values(False)
    for a, b in zip(actual, expected):torch.testing.assert_close(a, b, atol=2e-6, rtol=4e-5)
    active = [inputs[i] for i in targets]
    got, ref = torch.autograd.grad(loss(actual), active), torch.autograd.grad(loss(expected), active)
    for a, b in zip(got, ref):torch.testing.assert_close(a, b, atol=3e-6, rtol=7e-5)
    got = torch.autograd.grad(loss(values(True)), active, create_graph=True)
    ref = torch.autograd.grad(loss(values(False)), active, create_graph=True)
    direction = [torch.randn_like(v) for v in active]
    if edges:
        got2 = torch.autograd.grad(sum((g*v).sum() for g, v in zip(got, direction)), inputs, allow_unused=True)
        ref2 = torch.autograd.grad(sum((g*v).sum() for g, v in zip(ref, direction)), inputs, allow_unused=True)
        for a, b in zip(got2, ref2):
            assert (a is None) == (b is None)
            if a is not None:torch.testing.assert_close(a, b, atol=6e-6, rtol=1e-4)


@pytest.mark.parametrize('preprojected', [False, True])
def test_original_bias_free_phase_head_and_live_parameters(preprojected):
    from chorus.test.test_hp_backend import _build
    from chorus.hp.phase_trunk import PhaseTrunk
    torch.manual_seed(20261017)
    layer = _build('final', channels=12, lmax=2).interactions[-1].eval().requires_grad_(False)
    plan = PhaseTrunk(layer)
    plan.network_preprojected = preprojected
    width = layer.phase_node_norm.normalized_shape[0]
    nodes = torch.randn(17, width, device='cuda', requires_grad=True)
    radial = torch.randn(53, 8, device='cuda', requires_grad=True)
    src, dst = torch.randint(17, (53,), device='cuda'), torch.randint(17, (53,), device='cuda')
    probe = tuple(torch.randn(53, layer.phase_head.out_features, device='cuda') for _ in range(2))
    for mutate in (False, True):
        if mutate:
            with torch.no_grad():layer.phase_head.weight.add_(0.017)
        plan.fused_network = False
        hidden = plan(nodes, radial, src, dst)
        expected = layer.phase_head(hidden), layer.phase_amplitude_head(hidden)
        plan.fused_network = True
        pair = plan(nodes, radial, src, dst)
        actual = pair.theta, pair.raw
        for a, b in zip(actual, expected):torch.testing.assert_close(a, b, atol=2e-6, rtol=5e-5)
        actual_grad = torch.autograd.grad(sum((v*p).sum() for v, p in zip(actual, probe)), (nodes, radial))
        expected_grad = torch.autograd.grad(sum((v*p).sum() for v, p in zip(expected, probe)), (nodes, radial))
        for a, b in zip(actual_grad, expected_grad):torch.testing.assert_close(a, b, atol=3e-6, rtol=8e-5)
