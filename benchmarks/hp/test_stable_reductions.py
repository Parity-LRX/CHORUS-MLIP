"""Remote CUDA qualification for stable topology and input-owned density VJP."""
import pytest
import torch
from chorus.hp.density import DensityPlan, _Density
from chorus.models.pure_cartesian_ictd_fix import PhaseHermitianFullLResidual
from mace_ictc.models.message_topology import stable_topology

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='remote CUDA required')


@pytest.mark.parametrize('edges', [0, 117, 1024])
def test_stable_topology_live_capture(edges):
    torch.manual_seed(20261017)
    n = 19
    keys = torch.randint(n, (edges,), device='cuda', dtype=torch.long)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):stable_topology(keys, n, True)
    torch.cuda.current_stream().wait_stream(stream)
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):ptr, order, hint = stable_topology(keys, n, True)
    for changed in (False, True):
        if changed:keys.add_(3).remainder_(n)
        cg.replay()
        expected_order = torch.argsort(keys, stable=True)
        counts = torch.bincount(keys, minlength=n)
        expected_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        torch.testing.assert_close(ptr, expected_ptr, atol=0, rtol=0)
        torch.testing.assert_close(order, expected_order, atol=0, rtol=0)
        assert bool(hint[0]) == bool((keys[1:] >= keys[:-1]).all())


@pytest.mark.parametrize('form', ['hermitian', 'charge2'])
@pytest.mark.parametrize('lmax', [0, 1, 2, 3])
def test_density_owned_vjp_and_mixed(form, lmax, monkeypatch):
    torch.manual_seed(20261017)
    adapter = PhaseHermitianFullLResidual(num_elements=2, channels=13, lmax=lmax,
        density_rank=5, quadratic_form=form).cuda().float()
    plan = DensityPlan(adapter)
    x = (torch.randn(11, 2*plan.xhalf, device='cuda')*0.2).requires_grad_()
    grad = torch.randn(11, plan.width, device='cuda', requires_grad=True)
    expected = plan.native_vjp(x, grad)
    actual = plan.gather_vjp(x, grad)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=6e-5)
    assert torch.equal(actual, plan.gather_vjp(x, grad))
    monkeypatch.setattr(DensityPlan, 'vjp', DensityPlan.gather_vjp)
    output = _Density.apply(x, plan)
    got_first = torch.autograd.grad((output*grad).sum(), x, create_graph=True)[0]
    direction = torch.randn_like(x)
    got_mixed = torch.autograd.grad((got_first*direction).sum(), (x, grad), create_graph=True)
    expected_mixed = torch.autograd.grad((expected*direction).sum(), (x, grad), create_graph=True)
    for got, ref in zip(got_mixed, expected_mixed):
        torch.testing.assert_close(got, ref, atol=3e-6, rtol=8e-5)
