"""Remote exact-function and mixed-gradient checks for frozen affine SiLU."""
import pytest
import torch
import torch.nn.functional as F
from chorus.hp.affine_activation import FrozenAffineSiLU

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='remote CUDA required')


@pytest.mark.parametrize('width', [32, 64])
@pytest.mark.parametrize('edges', [0, 47, 256])
def test_frozen_affine_silu(width, edges):
    torch.manual_seed(20261017)
    x = torch.randn(edges, width, device='cuda', requires_grad=True)*0.2
    w = torch.randn(width, width, device='cuda')/width**0.5
    b = torch.randn(width, device='cuda')*0.1
    scale = 1.6791767923989418
    probe = torch.randn(edges, width, device='cuda')
    ref = F.silu(F.linear(x, w, b))*scale
    got = FrozenAffineSiLU.apply(x, w, b, scale)
    torch.testing.assert_close(got, ref, atol=2e-6, rtol=4e-5)
    expected_first = torch.autograd.grad((ref*probe).sum(), x, create_graph=True)[0]
    actual_first = torch.autograd.grad((got*probe).sum(), x, create_graph=True)[0]
    torch.testing.assert_close(actual_first, expected_first, atol=3e-6, rtol=5e-5)
    direction = torch.randn_like(x)
    if edges:
        actual_second = torch.autograd.grad((actual_first*direction).sum(), x)[0]
        expected_second = torch.autograd.grad((expected_first*direction).sum(), x)[0]
        torch.testing.assert_close(actual_second, expected_second, atol=4e-6, rtol=8e-5)
    got = FrozenAffineSiLU.apply(x, w, b, scale)
    actual_first = torch.autograd.grad((got*probe).sum(), x)[0]
    torch.testing.assert_close(actual_first, expected_first, atol=3e-6, rtol=5e-5)
