"""Remote qualification of mixed and third derivatives of the joint operator."""
import pytest
import torch
from mace_ictc.hp import CoherentGraphMessage, EdgeTensorProduct


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("third", [False, True])
@pytest.mark.parametrize("schedule", [
    (0, False, False, 0), (32, False, False, 0), (64, False, False, 0),
    (0, True, False, 0), (32, True, False, 0), (64, True, False, 0),
    (0, False, True, 0), (64, False, True, 0),
    (0, False, False, 2), (0, False, False, 4), (0, False, True, 4),
    (0, False, False, 0, True), (64, False, True, 0, True),
    (0, False, False, 0, True, 1), (0, False, False, 0, True, 2),
    (32, False, False, 0, True, 1, True), (64, False, True, 0, True, 1, True),
])
def test_coherent_live_cotangents_and_higher_derivatives(third, schedule, monkeypatch):
    mixed_threads, split_first, mixed_grouped, csr_splits, *extra = schedule
    torch.manual_seed(719)
    c, n, e = 8, 7, 19
    edge = EdgeTensorProduct.create(channels=c, input_lmax=1, edge_lmax=1, device="cuda")
    op = CoherentGraphMessage(edge.tp, input_lmax=1, edge_lmax=1, heads=1, angular=True,
                             mixed_threads=mixed_threads, training_split_first=split_first,
                             mixed_grouped=mixed_grouped, training_csr_splits=csr_splits,
                             mixed_destination=bool(extra and extra[0]),
                             training_destination_first=extra[1] if len(extra) > 1 else 0,
                             mixed_compact=extra[2] if len(extra) > 2 else False)
    x = {l: (torch.randn(n, c, 2*l+1, device="cuda")*.2).requires_grad_() for l in range(2)}
    h = {l: (torch.randn(e, 1, 2*l+1, device="cuda")*.2).requires_grad_() for l in range(2)}
    g = (torch.randn(e, edge.gate_width, device="cuda")*.2).requires_grad_()
    r = torch.randn(e, 2, 1, device="cuda", requires_grad=True)
    i = torch.randn(e, 2, 1, device="cuda", requires_grad=True)
    src, dst = (torch.randint(n, (e,), device="cuda") for _ in range(2))
    mask = (torch.arange(e, device="cuda") % 3 != 0).float()
    probes = {l: torch.randn(n, 3, edge.tp.path_counts_by_l[l]*c, 2*l+1,
                             device="cuda", requires_grad=True) for l in range(2)}
    inputs = [*x.values(), *h.values(), g, r, i]
    all_inputs = inputs+list(probes.values())
    tangents = [torch.randn_like(v) for v in inputs]

    def reference():
        ids = torch.tensor([out_l for _, _, out_l in edge.tp.paths], device="cuda")
        gates = [g]+[(g.reshape(e, edge.tp.num_paths, c)*co.squeeze(-1).index_select(
            1, ids)[:, :, None]).reshape_as(g) for co in (r, i)]
        streams = []
        for gate in gates:
            messages = edge.tp({l: v.index_select(0, src) for l, v in x.items()}, h, gate)
            streams.append({l: v.new_zeros(n, *v.shape[1:]).index_add(0, dst, v*mask[:, None, None])
                            for l, v in messages.items()})
        return {l: torch.stack([v[l] for v in streams], 1) for l in range(2)}

    def derivatives(values):
        # Nonlinear output loss supplies a live, input-dependent cotangent.
        loss = sum((v*probes[l]).square().sum() for l, v in values.items())
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        second = torch.autograd.grad(sum((v*t).sum() for v, t in zip(first, tangents)),
                                     all_inputs, create_graph=third, allow_unused=True)
        if not third:
            return second
        return torch.autograd.grad(sum(v.sum() for v in second if v is not None),
                                   all_inputs, allow_unused=True)

    expected = derivatives(reference())
    if not third:
        import mace_ictc.hp.coherent as coherent
        def no_reference(*args, **kwargs):
            raise AssertionError("Force double backward must use the joint native mixed adjoint")
        monkeypatch.setattr(coherent, "_reference", no_reference)
    actual = derivatives(op(x, h, g, src, dst, mask, r, i))
    for got, ref in zip(actual, expected):
        assert (got is None) == (ref is None)
        if ref is not None:
            torch.testing.assert_close(got, ref, atol=1e-5, rtol=4e-4)
