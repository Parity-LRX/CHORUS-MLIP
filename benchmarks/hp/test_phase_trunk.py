"""Remote CUDA checks of target-aware phase-context adjoint pruning."""
import pytest
import torch
import torch.nn.functional as F

from chorus.hp.phase_trunk import _GatherSiLU, _GatherSiLUMixed, _FrozenContextSiLU


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("targets", [(0, 1, 2), (0,), (1,), (2,)])
@pytest.mark.parametrize("mixed", [False, True])
def test_phase_context_target_gradients_and_mixed(targets, mixed):
    torch.manual_seed(20261017)
    n, e, h = 17, 53, 32
    inputs = [torch.randn(n, h, device="cuda", requires_grad=True),
              torch.randn(n, h, device="cuda", requires_grad=True),
              torch.randn(e, h, device="cuda", requires_grad=True)]
    src, dst = torch.randint(n, (e,), device="cuda"), torch.randint(n, (e,), device="cuda")
    probe = torch.randn(e, h, device="cuda")
    scale = 1.6791767923989418

    def original():
        return F.silu(inputs[0][dst]+inputs[1][src]+inputs[2])*scale

    activation = _GatherSiLUMixed if mixed else _GatherSiLU
    actual = activation.apply(*inputs, src, dst, scale)
    expected = original()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    active = [inputs[i] for i in targets]
    got = torch.autograd.grad((actual*probe).sum(), active)
    ref = torch.autograd.grad((expected*probe).sum(), active)
    for value, reference in zip(got, ref):
        torch.testing.assert_close(value, reference, atol=3e-6, rtol=3e-5)
    actual, expected = activation.apply(*inputs, src, dst, scale), original()
    got = torch.autograd.grad((actual.square()*probe).sum(), active, create_graph=True)
    ref = torch.autograd.grad((expected.square()*probe).sum(), active, create_graph=True)
    tangent = [torch.randn_like(v) for v in got]
    got_second = torch.autograd.grad(sum((v*t).sum() for v, t in zip(got, tangent)), inputs)
    ref_second = torch.autograd.grad(sum((v*t).sum() for v, t in zip(ref, tangent)), inputs)
    for value, reference in zip(got_second, ref_second):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("targets", [(0, 1, 2), (0,), (1,), (2,)])
def test_fused_radial_context_values_and_mixed(targets):
    torch.manual_seed(20261017)
    n, e, h, k = 17, 53, 32, 8
    inputs = [torch.randn(n, h, device="cuda", requires_grad=True),
              torch.randn(n, h, device="cuda", requires_grad=True),
              torch.randn(e, k, device="cuda", requires_grad=True)]
    weight = torch.randn(h, k, device="cuda")*0.2
    bias = torch.randn(h, device="cuda")*0.2
    src, dst = torch.randint(n, (e,), device="cuda"), torch.randint(n, (e,), device="cuda")
    probe = torch.randn(e, h, device="cuda")
    scale = 1.6791767923989418

    def original():
        return F.silu(inputs[0][dst]+inputs[1][src]+F.linear(inputs[2], weight, bias))*scale

    def fused():
        return _FrozenContextSiLU.apply(*inputs, weight, bias, src, dst, scale)

    actual, expected = fused(), original()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    active = [inputs[i] for i in targets]
    got = torch.autograd.grad((actual*probe).sum(), active)
    ref = torch.autograd.grad((expected*probe).sum(), active)
    for value, reference in zip(got, ref):
        torch.testing.assert_close(value, reference, atol=3e-6, rtol=3e-5)
    got = torch.autograd.grad((fused().square()*probe).sum(), active, create_graph=True)
    ref = torch.autograd.grad((original().square()*probe).sum(), active, create_graph=True)
    tangent = [torch.randn_like(v) for v in got]
    got_second = torch.autograd.grad(sum((v*t).sum() for v, t in zip(got, tangent)), inputs)
    ref_second = torch.autograd.grad(sum((v*t).sum() for v, t in zip(ref, tangent)), inputs)
    for value, reference in zip(got_second, ref_second):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("targets", [(0, 1, 2), (0,), (1,), (2,)])
@pytest.mark.parametrize("third", [False, True])
def test_live_context_cotangent_and_higher_derivatives(targets, third):
    from mace_ictc.hp import gather_silu_adjoint
    torch.manual_seed(918)
    n, e, h = 13, 41, 19
    d = (torch.randn(n, h, device="cuda") * .3).requires_grad_()
    s = (torch.randn(n, h, device="cuda") * .3).requires_grad_()
    r = (torch.randn(e, h, device="cuda") * .3).requires_grad_()
    g = torch.randn(e, h, device="cuda", requires_grad=True)
    src, dst = (torch.randint(n, (e,), device="cuda") for _ in range(2))
    scale = 1.6791767923989418
    needed = tuple(j in targets for j in range(3))
    tangent = [torch.randn_like(v) for v in (d, s, r)]
    inputs = (d, s, r, g)

    def derivatives(native):
        if native:
            first = gather_silu_adjoint(d, s, r, g, src, dst, scale, needed)
        else:
            a = d[dst] + s[src] + r
            y = F.silu(a) * scale
            values = iter(torch.autograd.grad(y, [inputs[j] for j in targets],
                                             g, create_graph=True))
            first = tuple(next(values) if j in targets else None for j in range(3))
        loss = sum((value * t).sum() for value, t in zip(first, tangent) if value is not None)
        second = torch.autograd.grad(loss, inputs, create_graph=third)
        if not third:
            return second
        return torch.autograd.grad(sum(v.square().sum() for v in second), inputs, allow_unused=True)

    expected, actual = derivatives(False), derivatives(True)
    for got, ref in zip(actual, expected):
        assert (got is None) == (ref is None)
        if ref is not None:
            torch.testing.assert_close(got, ref, atol=2e-5, rtol=4e-4)


@pytest.mark.parametrize("n,e,h", [(7, 0, 1), (19, 67, 17), (11, 29, 64)])
def test_context_adjoint_dynamic_empty_and_live_values(n, e, h):
    from mace_ictc.hp import gather_silu_adjoint
    torch.manual_seed(n + e + h)
    inputs = [torch.randn(n, h, device="cuda", requires_grad=True),
              torch.randn(n, h, device="cuda", requires_grad=True),
              torch.randn(e, h, device="cuda", requires_grad=True)]
    src, dst = (torch.randint(n, (e,), device="cuda") for _ in range(2))
    g = torch.randn(e, h, device="cuda", requires_grad=True)
    for change in (0., .2):
        # No retained node/edge addresses or values: every call consumes its inputs.
        values = [x + change for x in inputs]
        expected = torch.autograd.grad(F.silu(values[0][dst] + values[1][src] + values[2]),
                                       values, g, create_graph=True)
        actual = gather_silu_adjoint(*values, g, src, dst)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, atol=5e-6, rtol=2e-4)
        expected_second = torch.autograd.grad(sum(v.sum() for v in expected), (*inputs, g))
        actual_second = torch.autograd.grad(sum(v.sum() for v in actual), (*inputs, g))
        for got, ref in zip(actual_second, expected_second):
            torch.testing.assert_close(got, ref, atol=2e-5, rtol=3e-4)
