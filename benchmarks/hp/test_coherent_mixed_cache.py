"""CUDA-only checks of cached high-l live mixed adjoints and all targets."""
import pytest
import torch

from mace_ictc.hp import CoherentGraphMessage, EdgeTensorProduct

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("budget", [8, 16, 32, 64])
@pytest.mark.parametrize("third", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_cached_mixed_high_l_and_cotangents(budget, third, first, monkeypatch):
    _check_cached(budget, third, first, False, monkeypatch)


@pytest.mark.parametrize("budget", [8, 16, 32])
@pytest.mark.parametrize("third", [False, True])
def test_cached_mixed_fused_jvp(budget, third, monkeypatch):
    _check_cached(budget, third, False, True, monkeypatch)


@pytest.mark.parametrize("budget", [16, 32])
@pytest.mark.parametrize("fused_jvp", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_cached_mixed_group_specialization(budget, fused_jvp, first, monkeypatch):
    _check_cached(budget, False, first, fused_jvp, monkeypatch, grouped=True)


@pytest.mark.parametrize("schedule", ["pruned", "grouped", "compact", "compact_grouped"])
@pytest.mark.parametrize("prune_h", [False, True])
def test_first_adjoint_schedules(schedule, prune_h, monkeypatch):
    _check_cached(16, False, False, True, monkeypatch, grouped=True,
        first_schedule=schedule, prune_h=prune_h)


@pytest.mark.parametrize("schedule", ["pruned", "grouped", "compact", "compact_grouped"])
@pytest.mark.parametrize("freeze_x", [False, True])
def test_first_schedule_plain_backward_dispatch(schedule, freeze_x, monkeypatch):
    _check_cached(16, False, False, True, monkeypatch, grouped=True,
        first_schedule=schedule, prune_h=True, plain_first=True, freeze_x=freeze_x)


def _check_cached(budget, third, first, fused_jvp, monkeypatch, grouped=False,
                  first_schedule="legacy", prune_h=False, plain_first=False, freeze_x=False):
    torch.manual_seed(719)
    c, n, e, lmax = 32, 7, 19, 3
    edge = EdgeTensorProduct.create(channels=c, input_lmax=lmax, edge_lmax=lmax, device="cuda")
    op = CoherentGraphMessage(edge.tp, input_lmax=lmax, edge_lmax=lmax,
        mixed_cache=budget, mixed_destination=True, training_destination_first=0 if plain_first else 1, first_cache=first,
        mixed_fuse_jvp=fused_jvp, mixed_grouped=grouped, first_schedule=first_schedule)
    x = {l: (torch.randn(n, c, 2*l+1, device="cuda")*.2).requires_grad_(not freeze_x) for l in range(lmax+1)}
    h = {l: (torch.randn(e, 1, 2*l+1, device="cuda")*.2).requires_grad_() for l in range(lmax+1)}
    gates = (torch.randn(e, edge.gate_width, device="cuda")*.2).requires_grad_()
    real, imag = (torch.randn(e, lmax+1, 1, device="cuda", requires_grad=True) for _ in range(2))
    src, dst = (torch.randint(n, (e,), device="cuda") for _ in range(2))
    mask = (torch.arange(e, device="cuda")%3 != 0).float()
    probes = {l: torch.randn(n, 3, edge.tp.path_counts_by_l[l]*c, 2*l+1,
                            device="cuda", requires_grad=True) for l in range(lmax+1)}
    inputs = [*([] if freeze_x else x.values()), *([] if prune_h else h.values()), gates, real, imag]
    tangents = [torch.randn_like(value) for value in inputs]
    all_inputs = [*inputs, *probes.values()]

    def reference():
        degrees = torch.tensor([out for _, _, out in edge.tp.paths], device="cuda")
        weighted = [gates]+[(gates.reshape(e, edge.tp.num_paths, c)*phase.squeeze(-1).index_select(
            1, degrees)[:, :, None]).reshape_as(gates) for phase in (real, imag)]
        streams = []
        for gate in weighted:
            messages = edge.tp({l: value.index_select(0, src) for l, value in x.items()}, h, gate)
            streams.append({l: value.new_zeros(n, *value.shape[1:]).index_add(
                0, dst, value*mask[:, None, None]) for l, value in messages.items()})
        return {l: torch.stack([value[l] for value in streams], 1) for l in range(lmax+1)}

    def derivatives(values):
        loss = sum((value*probes[l]).square().sum() for l, value in values.items())
        if plain_first:
            return torch.autograd.grad(loss, inputs)
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        second = torch.autograd.grad(sum((value*tangent).sum() for value, tangent in zip(first, tangents)),
            all_inputs, create_graph=third, allow_unused=True)
        if third:
            return torch.autograd.grad(sum(value.sum() for value in second if value is not None),
                all_inputs, allow_unused=True)
        return second

    expected = derivatives(reference())
    first_calls = []
    if plain_first:
        import mace_ictc.hp._coherent_training as training
        original_first = training._first
        def record_first(*args, **kwargs):
            first_calls.append(True)
            return original_first(*args, **kwargs)
        monkeypatch.setattr(training, "_first", record_first)
    if not third:
        import mace_ictc.hp.coherent as coherent
        def no_reference(*args, **kwargs):
            raise AssertionError("Cached force mixed adjoint must stay native")
        monkeypatch.setattr(coherent, "_reference", no_reference)
    actual = derivatives(op(x, h, gates, src, dst, mask, real, imag))
    if plain_first:
        assert first_calls, "Explicit first_schedule must also apply without create_graph or live X"
    for got, ref in zip(actual, expected):
        assert (got is None) == (ref is None)
        if got is not None:
            torch.testing.assert_close(got, ref, atol=2e-5, rtol=5e-4)
