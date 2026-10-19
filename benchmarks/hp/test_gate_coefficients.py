"""Remote-only value, first/mixed/third, and partial-target gate tests."""
import pytest
import torch

from mace_ictc.hp import path_weighted_gates

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("shape", [(0, 17), (37, 17), (257, 65)])
@pytest.mark.parametrize("third", [False, True])
def test_gate_live_mixed_and_higher(shape, third):
    torch.manual_seed(20261017)
    x = torch.randn(*shape, device="cuda", requires_grad=True)
    w = torch.randn(shape[1], device="cuda", requires_grad=True)
    g = torch.randn_like(x, requires_grad=True)
    hx, hw = torch.randn_like(x), torch.randn_like(w)
    ref, got = x*w, path_weighted_gates(x, w)
    torch.testing.assert_close(got, ref, atol=0., rtol=0.)
    first = [torch.autograd.grad(value, (x, w), g, create_graph=True) for value in (ref, got)]
    for actual, expected in zip(first[1], first[0]):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
    second = [torch.autograd.grad((dx*hx).sum()+(dw*hw).sum(), (x, w, g), create_graph=third)
              for dx, dw in first]
    for actual, expected in zip(second[1], second[0]):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
    if third:
        results = [torch.autograd.grad(sum(value.square().sum() for value in values), (x, w, g))
                   for values in second]
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("target", ["x", "w"])
def test_gate_partial_first_target(target):
    torch.manual_seed(20261017)
    x = torch.randn(257, 31, device="cuda", requires_grad=True)
    w = torch.randn(31, device="cuda", requires_grad=True)
    g = torch.randn_like(x, requires_grad=True)
    first_target = x if target == "x" else w
    tangent = torch.randn_like(first_target)
    result = []
    for value in (x*w, path_weighted_gates(x, w)):
        first = torch.autograd.grad(value, first_target, g, create_graph=True)[0]
        result.append(torch.autograd.grad((first*tangent).sum(), (x, w, g), allow_unused=True))
    for actual, expected in zip(result[1], result[0]):
        assert (actual is None) == (expected is None)
        if expected is not None:
            torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("zero_coefficients", [False, True])
def test_radial_head_live_banks_and_force_loss(zero_coefficients):
    from mace_ictc.hp import path_weighted_linear
    torch.manual_seed(20261017)
    x = (torch.randn(257, 31, device="cuda")*.1).requires_grad_()
    w = (torch.randn(65, 31, device="cuda")*.1).requires_grad_()
    a = (torch.randn(65, device="cuda")*.1).requires_grad_()
    if zero_coefficients:
        with torch.no_grad():
            a[::3] = 0.
    values = (torch.nn.functional.linear(x, w)*a, path_weighted_linear(x, w, a))
    force_probe = torch.randn_like(x)
    records = []
    for value in values:
        energy = value.square().sum()
        coordinate_vjp = torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy*.01+(coordinate_vjp*force_probe).square().sum()
        records.append(torch.autograd.grad(loss, (x, w, a)))
    torch.testing.assert_close(values[1], values[0], atol=2e-7, rtol=2e-6)
    for actual, expected in zip(records[1], records[0]):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
