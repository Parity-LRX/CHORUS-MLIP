"""Remote tests of FP32 SiLU and its literal force-loss mixed adjoint."""
import pytest
import torch
import torch.nn.functional as F

from mace_ictc.hp import normalized_silu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("scale", [1., 1.6791767923989418])
@pytest.mark.parametrize("third", [False, True])
def test_literal_silu_values_live_cotangent_and_mixed(scale, third):
    torch.manual_seed(20261017)
    x = (torch.randn(8193, device="cuda") * 5).requires_grad_()
    g = torch.randn_like(x, requires_grad=True)
    h = torch.randn_like(x)
    ref = F.silu(x) * scale
    got = normalized_silu(x, scale)
    torch.testing.assert_close(got, ref, atol=0., rtol=0.)
    first_ref = torch.autograd.grad(ref, x, g, create_graph=True)[0]
    first_got = torch.autograd.grad(got, x, g, create_graph=True)[0]
    torch.testing.assert_close(first_got, first_ref, atol=0., rtol=0.)
    second_ref = torch.autograd.grad((first_ref*h).sum(), (x, g), create_graph=third)
    second_got = torch.autograd.grad((first_got*h).sum(), (x, g), create_graph=third)
    for actual, expected in zip(second_got, second_ref):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    if third:
        ref3 = torch.autograd.grad(sum(v.square().sum() for v in second_ref), (x, g))
        got3 = torch.autograd.grad(sum(v.square().sum() for v in second_got), (x, g))
        for actual, expected in zip(got3, ref3):
            torch.testing.assert_close(actual, expected, atol=5e-6, rtol=2e-5)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("scale", [1., 1.6791767923989418])
def test_literal_phase_gather_silu(mixed, scale):
    from chorus.hp.phase_trunk import _GatherSiLU, _GatherSiLUMixed
    torch.manual_seed(20261017)
    # Permutations give every node one incoming edge, removing scatter-order
    # ambiguity from this arithmetic test. Repeated neighbors have separate tests.
    n, width = 129, 32
    d, s, r = [(torch.randn(n, width, device="cuda")*3).requires_grad_() for _ in range(3)]
    src, dst = [torch.randperm(n, device="cuda") for _ in range(2)]
    g = torch.randn_like(r, requires_grad=True)
    inputs = (d, s, r)
    h = [torch.randn_like(value) for value in inputs]
    reference = F.silu(d[dst]+s[src]+r)*scale
    op = _GatherSiLUMixed if mixed else _GatherSiLU
    actual = op.apply(d, s, r, src, dst, scale)
    torch.testing.assert_close(actual, reference, atol=0., rtol=0.)
    ref1 = torch.autograd.grad(reference, inputs, g, create_graph=True)
    got1 = torch.autograd.grad(actual, inputs, g, create_graph=True)
    for got, ref in zip(got1, ref1):
        torch.testing.assert_close(got, ref, atol=0., rtol=0.)
    ref2 = torch.autograd.grad(sum((v*t).sum() for v, t in zip(ref1, h)), (*inputs, g))
    got2 = torch.autograd.grad(sum((v*t).sum() for v, t in zip(got1, h)), (*inputs, g))
    for got, ref in zip(got2, ref2):
        torch.testing.assert_close(got, ref, atol=2e-7, rtol=2e-6)
