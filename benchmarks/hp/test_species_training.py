"""Native live-species-bank values and all force-loss derivative dependencies."""
import pytest
import torch
from mace_ictc.hp import training_species_channel_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("ci,co,m,bias", [(13, 7, 1, True), (16, 19, 3, False), (128, 128, 5, False)])
@pytest.mark.parametrize("third", [False, True])
def test_live_species_banks(ci, co, m, bias, third):
    torch.manual_seed(20261017)
    n, s = 97, 4
    x = (torch.randn(n, m, ci, device="cuda")*.1).transpose(1, 2).requires_grad_()
    w = (torch.randn(s, ci, co, device="cuda")*.1).transpose(1, 2).requires_grad_()
    b = (torch.randn(s, co, device="cuda")*.1).requires_grad_() if bias else None
    index = torch.arange(n, device="cuda").roll(7)%3  # Includes an absent species.
    g = torch.randn(n, co, m, device="cuda", requires_grad=True)
    inputs = (x, w) if b is None else (x, w, b)
    tangents = [torch.randn_like(value) for value in inputs]
    reference = torch.bmm(w.index_select(0, index), x)
    if b is not None:
        reference = reference+b.index_select(0, index)[:, :, None]
    actual = training_species_channel_linear(x, w, index, b)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=4e-5)
    first = [torch.autograd.grad(value, inputs, g, create_graph=True) for value in (reference, actual)]
    for got, ref in zip(first[1], first[0]):
        torch.testing.assert_close(got, ref, atol=8e-6, rtol=5e-5)
    second = [torch.autograd.grad(sum((v*t).sum() for v, t in zip(row, tangents)),
                                  (*inputs, g), create_graph=third, allow_unused=True) for row in first]
    for got, ref in zip(second[1], second[0]):
        assert (got is None) == (ref is None)
        if ref is not None:
            torch.testing.assert_close(got, ref, atol=3e-5, rtol=8e-5)
    if third:
        values = [torch.autograd.grad(sum(v.square().mean() for v in row if v is not None),
                                     (*inputs, g), allow_unused=True) for row in second]
        for got, ref in zip(values[1], values[0]):
            assert (got is None) == (ref is None)
            if ref is not None:
                torch.testing.assert_close(got, ref, atol=3e-5, rtol=8e-5)


@pytest.mark.parametrize("target", ["features", "weights"])
def test_live_species_partial_targets(target):
    torch.manual_seed(20261017)
    x = (torch.randn(17, 7, 3, device="cuda")*.1).requires_grad_()
    w = (torch.randn(4, 11, 7, device="cuda")*.1).requires_grad_()
    index = torch.arange(17, device="cuda")%4
    g = torch.randn(17, 11, 3, device="cuda", requires_grad=True)
    first_target = x if target == "features" else w
    tangent = torch.randn_like(first_target)
    records = []
    for value in (torch.bmm(w.index_select(0, index), x), training_species_channel_linear(x, w, index)):
        first = torch.autograd.grad(value, first_target, g, create_graph=True)[0]
        records.append(torch.autograd.grad((first*tangent).sum(), (x, w, g), allow_unused=True))
    for got, ref in zip(records[1], records[0]):
        assert (got is None) == (ref is None)
        if ref is not None:
            torch.testing.assert_close(got, ref, atol=3e-6, rtol=5e-5)
