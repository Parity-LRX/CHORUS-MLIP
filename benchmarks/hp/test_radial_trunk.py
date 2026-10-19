"""Remote CUDA tests of the original frozen radial trunk's full derivatives."""
import pytest
import torch
from chorus.hp.radial_trunk import _Trunk, _native, FrozenRadialTrunk
from chorus.models.pure_cartesian_ictd_fix import _Normalize2MomSiLU

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='remote CUDA required')


@pytest.mark.parametrize('edges', [0, 47, 512])
@pytest.mark.parametrize('tile', [16, 32, 64, 128])
def test_radial_trunk_values_first_mixed(edges, tile):
    torch.manual_seed(20261017)
    x = (torch.randn(edges, 8, device='cuda')*0.2).requires_grad_()
    weights = [torch.randn(64, 8, device='cuda')*0.1,
               torch.randn(64, 64, device='cuda')*0.1,
               torch.randn(64, 64, device='cuda')*0.1]
    scales = (1.6791767923989418,)*3
    probe = torch.randn(edges, 64, device='cuda')
    def value(fused):
        return _Trunk.apply(x, *weights, (*scales, tile)) if fused else _native(x, *weights, scales)
    got, ref = value(True), value(False)
    torch.testing.assert_close(got, ref, atol=2e-6, rtol=4e-5)
    dg, = torch.autograd.grad((got*probe).sum(), x)
    dr, = torch.autograd.grad((ref*probe).sum(), x)
    torch.testing.assert_close(dg, dr, atol=3e-6, rtol=8e-5)
    if edges:
        dg, = torch.autograd.grad((value(True)*probe).sum(), x, create_graph=True)
        dr, = torch.autograd.grad((value(False)*probe).sum(), x, create_graph=True)
        direction = torch.randn_like(x)
        gg, = torch.autograd.grad((dg*direction).sum(), x)
        gr, = torch.autograd.grad((dr*direction).sum(), x)
        torch.testing.assert_close(gg, gr, atol=5e-6, rtol=1e-4)


def test_radial_trunk_original_parameter_updates_and_guard():
    torch.manual_seed(20261017)
    fc = torch.nn.Sequential(torch.nn.Linear(8, 64, bias=False), _Normalize2MomSiLU(),
        torch.nn.Linear(64, 64, bias=False), _Normalize2MomSiLU(),
        torch.nn.Linear(64, 64, bias=False), _Normalize2MomSiLU(),
        torch.nn.Linear(64, 384, bias=False)).cuda().eval().requires_grad_(False)
    plan = FrozenRadialTrunk()
    x = torch.randn(47, 8, device='cuda', requires_grad=True)
    for mutate in (False, True):
        if mutate:
            with torch.no_grad():fc[2].weight.add_(0.007)
        got, ref = plan(fc, x), fc[:-1](x)
        torch.testing.assert_close(got, ref, atol=2e-6, rtol=5e-5)
    fc[0].weight.requires_grad_(True)
    assert not plan.eligible(fc, x)
    with pytest.raises(ValueError, match='ineligible'):
        plan(fc, x)
