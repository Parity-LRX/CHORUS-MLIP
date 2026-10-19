"""Remote-only indexed frozen-bank GEMM and recursive adjoint checks."""
import pytest
import torch
from mace_ictc.hp import species_channel_linear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("ci,co,m,s,bias", [
    (13, 7, 1, 4, True), (16, 19, 3, 2, False),
    (128, 128, 5, 4, False), (7, 11, 3, 1, False),
])
def test_indexed_species_values_first_and_mixed(ci, co, m, s, bias):
    torch.manual_seed(207)
    x = (torch.randn(17, m, ci, device="cuda")*0.2).transpose(1, 2).requires_grad_()
    weight = torch.randn(s, ci, co, device="cuda").transpose(1, 2)*0.1
    bank_bias = torch.randn(s, co, device="cuda")*0.1 if bias else None
    species = torch.arange(17, device="cuda").roll(7) % max(s-1, 1)
    direction = torch.randn_like(x)
    for change in (False, True):
        if change:
            species.add_(1).remainder_(s)
            weight.mul_(0.9)
        reference = torch.bmm(weight.index_select(0, species), x)
        if bank_bias is not None:
            reference = reference+bank_bias.index_select(0, species)[:, :, None]
        actual = species_channel_linear(x, weight, species, bank_bias)
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=4e-5)
        ref_grad = torch.autograd.grad(reference.square().sum(), x, create_graph=True)[0]
        got_grad = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
        torch.testing.assert_close(got_grad, ref_grad, atol=3e-6, rtol=5e-5)
        ref_second = torch.autograd.grad((ref_grad*direction).sum(), x)[0]
        got_second = torch.autograd.grad((got_grad*direction).sum(), x)[0]
        torch.testing.assert_close(got_second, ref_second, atol=8e-6, rtol=7e-5)
