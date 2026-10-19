"""Confirm original-formula edge checkpointing preserves force-training VJPs."""
import pytest
import torch
from checkpoint_reference_tp import checkpoint_original_tp
from chorus.synthetic import make_fixed_graph
from chorus.test.test_hp_backend import _build, _gradient_record

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("chunk_nodes", [0, 5])
@pytest.mark.parametrize("chunk_channels", [0, 2])
def test_original_tp_checkpoint_values_and_mixed_gradients(chunk_nodes, chunk_channels):
    torch.manual_seed(20261017)
    model = _build("persistent")
    for layer in model.interactions:
        layer.tp.weight.requires_grad_(True)
    graph = make_fixed_graph(num_nodes=17, avg_degree=5, dtype=torch.float32,
                             device=torch.device("cuda"), seed=91)
    original = [layer.tp.forward for layer in model.interactions]
    contractions = [module for module in model.modules()
        if type(module).__name__ == "MaceSymmetricContraction"]
    original_sc = [module.forward for module in contractions]
    reference = _gradient_record(model, graph)
    with torch.autograd.graph.save_on_cpu(), checkpoint_original_tp(model, 13, chunk_nodes, chunk_channels):
        actual = _gradient_record(model, graph)
    assert all(layer.tp.forward == before for layer, before in zip(model.interactions, original))
    assert all(module.forward == before for module, before in zip(contractions, original_sc))
    for got, ref in zip(actual[:3], reference[:3]):
        torch.testing.assert_close(got, ref, atol=3e-6, rtol=5e-5)
    for name, ref in reference[3].items():
        got = actual[3][name]
        assert (got is None) == (ref is None), name
        if ref is not None:
            torch.testing.assert_close(got, ref, atol=3e-5, rtol=1e-3, msg=name)


@pytest.mark.parametrize('correlation', [2, 3])
def test_original_high_l_sc_ragged_channel_chunks(correlation):
    from chorus.models._mace_symmetric_contraction import MaceSymmetricContraction
    torch.manual_seed(20261017)
    model = torch.nn.Module()
    model.interactions = torch.nn.ModuleList()
    model.contraction = MaceSymmetricContraction('5x0e+5x1o+5x2e', '5x0e+5x1o+5x2e',
        correlation=correlation, num_elements=2, use_reduced_cg=False).cuda().float()
    x = (torch.randn(7, 5, 9, device='cuda')*.2).requires_grad_()
    y = torch.randn(7, 2, device='cuda', requires_grad=True)
    inputs = (x, y, *(p for p in model.parameters() if p.requires_grad))
    tangents = tuple(torch.randn_like(value) for value in inputs)

    def record():
        value = model.contraction(x, y)
        first = torch.autograd.grad(value.square().sum(), inputs, create_graph=True)
        mixed = torch.autograd.grad(sum((a*b).sum() for a, b in zip(first, tangents)), inputs)
        return value, first, mixed

    expected = record()
    with torch.autograd.graph.save_on_cpu(), checkpoint_original_tp(model, 0, 3, 2):
        actual = record()
    torch.testing.assert_close(actual[0], expected[0], atol=3e-6, rtol=5e-5)
    for got, ref in zip((*actual[1], *actual[2]), (*expected[1], *expected[2])):
        torch.testing.assert_close(got, ref, atol=3e-5, rtol=1e-3)
