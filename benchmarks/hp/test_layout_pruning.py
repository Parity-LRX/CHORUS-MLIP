"""Same views and exact derivatives, with one irreps-split adjoint assembly."""
import pytest
import torch

from chorus.models.pure_cartesian_ictd_layers import _split_irreps


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("tail,strided,selected", [
    (0, False, (0, 1, 2)), (7, False, (0, 1, 2)),
    (0, True, (0, 1, 2)), (7, True, (0, 2)),
    (0, False, (0,)), (7, True, (1,)),
])
def test_irreps_split_values_first_and_second(tail, strided, selected):
    torch.manual_seed(20261017)
    c, lmax = 13, 2
    width = c*9+tail
    base = torch.randn(3, 2, width*(2 if strided else 1), device="cuda", requires_grad=True)
    x = base[..., ::2] if strided else base
    actual = _split_irreps(x, c, lmax)
    expected, offset = {}, 0
    for l in range(lmax+1):
        size = c*(2*l+1)
        expected[l] = x[..., offset:offset+size].view(*x.shape[:-1], c, 2*l+1)
        offset += size
        assert actual[l].stride() == expected[l].stride()
        assert actual[l].data_ptr() == expected[l].data_ptr()
        torch.testing.assert_close(actual[l], expected[l], atol=0, rtol=0)
    probe = {l: torch.randn_like(actual[l]) for l in selected}
    got = torch.autograd.grad(sum((actual[l].pow(3)*probe[l]).sum() for l in selected), base,
                              create_graph=True)[0]
    ref = torch.autograd.grad(sum((expected[l].pow(3)*probe[l]).sum() for l in selected), base,
                              create_graph=True)[0]
    torch.testing.assert_close(got, ref, atol=0, rtol=0)
    tangent = torch.randn_like(base)
    got_second = torch.autograd.grad((got*tangent).sum(), base)[0]
    ref_second = torch.autograd.grad((ref*tangent).sum(), base)[0]
    torch.testing.assert_close(got_second, ref_second, atol=0, rtol=0)
