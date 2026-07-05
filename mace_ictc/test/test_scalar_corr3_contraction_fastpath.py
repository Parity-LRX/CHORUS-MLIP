"""Regression tests for the scalar-output correlation-3 contraction fast path."""

import torch
from e3nn import o3

from mace_ictc.models._mace_symmetric_contraction import MaceSymmetricContraction


def _make_contraction(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    torch.manual_seed(11)
    irreps_in = o3.Irreps("8x0e + 8x1o + 8x2e + 8x3o")
    irreps_out = o3.Irreps("8x0e")
    return MaceSymmetricContraction(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        correlation=3,
        num_elements=4,
    ).contractions[0]


def test_scalar_corr3_fast_path_matches_float32_tolerance():
    contraction = _make_contraction(torch.float32)
    assert contraction._use_scalar_corr3_fast

    x = torch.randn(13, 8, 16, dtype=torch.float32, requires_grad=True)
    y = torch.nn.functional.one_hot(torch.arange(13) % 4, num_classes=4).to(dtype=torch.float32)

    contraction._use_scalar_corr3_fast = False
    out_old = contraction(x, y)
    grad_old = torch.autograd.grad(out_old.sum(), x, retain_graph=True)[0]

    contraction._use_scalar_corr3_fast = True
    out_fast = contraction(x, y)
    grad_fast = torch.autograd.grad(out_fast.sum(), x)[0]

    assert torch.allclose(out_fast, out_old, rtol=2e-6, atol=2e-6)
    assert torch.allclose(grad_fast, grad_old, rtol=2e-6, atol=2e-6)


def test_scalar_corr3_fast_path_keeps_float64_on_original_graph():
    contraction = _make_contraction(torch.float64)
    assert contraction._use_scalar_corr3_fast

    x = torch.randn(13, 8, 16, dtype=torch.float64, requires_grad=True)
    y = torch.nn.functional.one_hot(torch.arange(13) % 4, num_classes=4).to(dtype=torch.float64)

    contraction._use_scalar_corr3_fast = False
    out_old = contraction(x, y)
    grad_old = torch.autograd.grad(out_old.sum(), x, retain_graph=True)[0]

    contraction._use_scalar_corr3_fast = True
    out_guarded = contraction(x, y)
    grad_guarded = torch.autograd.grad(out_guarded.sum(), x)[0]

    assert torch.equal(out_guarded, out_old)
    assert torch.equal(grad_guarded, grad_old)
