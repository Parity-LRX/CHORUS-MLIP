"""Remote CUDA first/mixed derivative checks for the CHORUS phase operator."""
import pytest
import torch
import torch.nn.functional as F
from chorus.hp.phase_coefficients import phase_coefficients


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("mode,has_raw", [
    ("polar", False), ("polar", True), ("aligned", True),
    ("positive", True), ("signed", True), ("cartesian", True),
])
@pytest.mark.parametrize("mixed_adjoint", [False, True])
def test_phase_values_and_adjoint(mode, has_raw, mixed_adjoint):
    torch.manual_seed(761)
    theta = torch.randn(67, 6, device="cuda", requires_grad=True)
    raw = torch.randn_like(theta, requires_grad=True) if has_raw else None
    inputs = [theta] if raw is None else [theta, raw]
    amplitude = torch.ones_like(theta) if raw is None else F.softplus(raw)
    if mode == "polar":
        ref = (amplitude*theta.cos(), amplitude*theta.sin())
    elif mode == "aligned":
        ref = (amplitude, theta*0.)
    elif mode == "positive":
        ref = (F.softplus(raw+theta), torch.zeros_like(theta))
    elif mode == "signed":
        ref = (amplitude*(1.+theta), torch.zeros_like(theta))
    else:
        ref = (1.+theta, raw)
    got = tuple(value.squeeze(-1) for value in phase_coefficients(theta, raw, mode, mixed_adjoint=mixed_adjoint))
    probes = [torch.randn_like(theta), torch.randn_like(theta)]
    for actual, expected in zip(got, ref):
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=3e-6)
    ref_loss = sum((value*probe).sum() for value, probe in zip(ref, probes))
    got_loss = sum((value*probe).sum() for value, probe in zip(got, probes))
    ref_grad = torch.autograd.grad(ref_loss, inputs, create_graph=True, retain_graph=True)
    got_grad = torch.autograd.grad(got_loss, inputs, create_graph=True, retain_graph=True)
    fast_grad = torch.autograd.grad(got_loss, inputs, retain_graph=True)
    for actual, fast, expected in zip(got_grad, fast_grad, ref_grad):
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=3e-6)
        torch.testing.assert_close(fast, expected, atol=5e-7, rtol=3e-6)
    tangents = [torch.randn_like(value) for value in inputs]
    zero = sum(value.sum()*0. for value in inputs)
    ref_second = torch.autograd.grad(zero+sum((g*h).sum() for g, h in zip(ref_grad, tangents)), inputs,
                                     retain_graph=True)
    got_second = torch.autograd.grad(zero+sum((g*h).sum() for g, h in zip(got_grad, tangents)), inputs)
    for actual, expected in zip(got_second, ref_second):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("has_raw", [False, True])
@pytest.mark.parametrize("third", [False, True])
def test_polar_joint_mixed_live_cotangents(has_raw, third):
    torch.manual_seed(763)
    theta = torch.randn(67, 6, device="cuda", requires_grad=True)
    raw = torch.randn_like(theta, requires_grad=True) if has_raw else None
    gc, gs = [torch.randn_like(theta, requires_grad=True) for _ in range(2)]
    inputs = (theta,) if raw is None else (theta, raw)
    tangents = [torch.randn_like(value) for value in inputs]
    amp = F.softplus(raw) if has_raw else 1.
    reference = (amp*theta.cos(), amp*theta.sin())
    actual = tuple(value.squeeze(-1) for value in phase_coefficients(
        theta, raw, "polar", mixed_adjoint=True))
    records = []
    for values in (reference, actual):
        first = torch.autograd.grad(values, inputs, (gc, gs), create_graph=True)
        records.append(torch.autograd.grad(sum((g*h).sum() for g, h in zip(first, tangents)),
                                           (*inputs, gc, gs), create_graph=third))
    for got, ref in zip(records[1], records[0]):
        torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-5)
    if third:
        values = [torch.autograd.grad(sum(v.square().sum() for v in record), (*inputs, gc, gs))
                  for record in records]
        for got, ref in zip(values[1], values[0]):
            torch.testing.assert_close(got, ref, atol=2e-5, rtol=5e-5)
