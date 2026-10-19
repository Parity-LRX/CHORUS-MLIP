"""Remote CUDA qualification of the native checkpoint-U path; no foreign TP."""
import pytest
import torch
from chorus.hp import configure_model
from chorus.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("correlation", [2, 3])
def test_checkpoint_u_without_basis_regeneration(monkeypatch, correlation):
    import chorus.models._mace_symmetric_contraction as sc
    from chorus.hp.runtime import _borrow_contraction
    torch.manual_seed(818)
    original = ICTDBridgeUSymmetricContractionSO3(
        num_elements=3, channels=8, lmax=2, target_lmax=2,
        correlation=correlation).cuda().float()
    constants = {f"{i}.U_matrix_{order}": c.U_tensors(order).detach().clone()
                 for i, c in enumerate(original.symmetric_contractions.contractions)
                 for order in range(1, correlation+1)}

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint reconstruction must not regenerate U")

    monkeypatch.setattr(sc, "U_matrix_real", forbidden)
    restored = ICTDBridgeUSymmetricContractionSO3(
        num_elements=3, channels=8, lmax=2, target_lmax=2,
        correlation=correlation, use_reduced_cg=True,
        ictc_u_tensor_state=constants).cuda().float()
    restored.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(11, 8*9, device="cuda", requires_grad=True)*0.2
    species = torch.arange(11, device="cuda") % 3
    attrs = torch.nn.functional.one_hot(species, 3).float()
    expected = original(x, attrs)
    torch.testing.assert_close(restored(x, attrs), expected, atol=0, rtol=0)
    borrowed = _borrow_contraction(restored, training=True, checkpoint_contraction=True)
    value = borrowed(x, attrs, species)
    torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-4)
    parameters = list(restored.parameters())
    ref_grad = torch.autograd.grad(expected.square().sum(), [x, *original.parameters()],
                                   retain_graph=True, create_graph=True)
    got_grad = torch.autograd.grad(value.square().sum(), [x, *parameters],
                                   retain_graph=True, create_graph=True)
    for got, ref in zip(got_grad, ref_grad):
        torch.testing.assert_close(got, ref, atol=3e-5, rtol=5e-4)
    ref_second = torch.autograd.grad(ref_grad[0].sum(), x)[0]
    got_second = torch.autograd.grad(got_grad[0].sum(), x)[0]
    torch.testing.assert_close(got_second, ref_second, atol=5e-5, rtol=1e-3)
