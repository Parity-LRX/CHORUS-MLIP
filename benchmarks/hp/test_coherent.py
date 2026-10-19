"""Remote-only qualification against the original Cartesian TP expression."""
import pytest
import torch
from mace_ictc.hp import CoherentGraphMessage, EdgeTensorProduct


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("channels,lmax,heads,angular,edges", [
    (8, 1, 1, True, 47), (12, 2, 2, True, 47),
    (13, 1, 1, False, 47), (8, 1, 1, True, 0),
])
def test_coherent_values_first_and_mixed(channels, lmax, heads, angular, edges):
    torch.manual_seed(20261017)
    edge = EdgeTensorProduct.create(channels=channels, input_lmax=lmax,
                                    edge_lmax=lmax, device="cuda")
    with torch.no_grad():
        edge.tp.weight.uniform_(0.7, 1.3)
    op = CoherentGraphMessage(edge.tp, input_lmax=lmax, edge_lmax=lmax,
                               heads=heads, angular=angular)
    n = 11
    x = {l: (torch.randn(n, channels, 2*l+1, device="cuda")*0.2).requires_grad_()
         for l in range(lmax+1)}
    h = {l: (torch.randn(edges, 1, 2*l+1, device="cuda")*0.2).requires_grad_()
         for l in range(lmax+1)}
    g = (torch.randn(edges, edge.gate_width, device="cuda")*0.2).requires_grad_()
    phases = heads*(lmax+1 if angular else 1)
    r = torch.randn(edges, phases, 1, device="cuda", requires_grad=True)
    i = torch.randn(edges, phases, 1, device="cuda", requires_grad=True)
    src = torch.randint(n, (edges,), device="cuda")
    dst = torch.randint(n, (edges,), device="cuda")
    mask = (torch.arange(edges, device="cuda") % 4 != 0).float()
    probes = {l: torch.randn(n, 3, edge.tp.path_counts_by_l[l]*channels, 2*l+1,
                             device="cuda") for l in range(lmax+1)}
    inputs = [*x.values(), *h.values(), g, r, i]

    def reference():
        phase_ids = torch.tensor([
            (out_l*heads if angular else 0)+head
            for _, _, out_l in edge.tp.paths for head in range(heads)], device="cuda")
        gates = [g]
        for coeff in (r, i):
            factors = coeff.squeeze(-1).index_select(1, phase_ids).reshape(
                edges, edge.tp.num_paths, heads, 1)
            gates.append((g.reshape(edges, edge.tp.num_paths, heads, channels//heads)*factors).reshape_as(g))
        values = []
        for gate in gates:
            messages = edge.tp({l: v.index_select(0, src) for l, v in x.items()}, h, gate)
            values.append({l: v.new_zeros(n, *v.shape[1:]).index_add(
                0, dst, v*mask[:, None, None]) for l, v in messages.items()})
        return {l: torch.stack([v[l] for v in values], 1) for l in range(lmax+1)}

    def loss(values):
        return sum((values[l]*probes[l]).sum() for l in values)

    expected = reference()
    actual = op(x, h, g, src, dst, mask, r, i)
    for l in expected:
        torch.testing.assert_close(actual[l], expected[l], atol=2e-6, rtol=3e-5)
        assert actual[l].movedim(-2, -1).is_contiguous()
    ref_grad = torch.autograd.grad(loss(expected), inputs, create_graph=True, retain_graph=True)
    fast_grad = torch.autograd.grad(loss(actual), inputs)
    for got, ref in zip(fast_grad, ref_grad):
        torch.testing.assert_close(got, ref, atol=3e-6, rtol=1e-4)
    actual = op(x, h, g, src, dst, mask, r, i)
    got_grad = torch.autograd.grad(loss(actual), inputs, create_graph=True)
    tangent = [torch.randn_like(v) for v in inputs]
    # The empty graph has exact zero gradients and no mixed dependencies.
    if edges:
        ref_second = torch.autograd.grad(sum((v*t).sum() for v, t in zip(ref_grad, tangent)), inputs,
                                         allow_unused=True)
        got_second = torch.autograd.grad(sum((v*t).sum() for v, t in zip(got_grad, tangent)), inputs,
                                         allow_unused=True)
        for got, ref in zip(got_second, ref_second):
            assert (got is None) == (ref is None)
            if got is not None:
                torch.testing.assert_close(got, ref, atol=5e-6, rtol=1e-4)
    with torch.no_grad():
        edge.tp.weight.add_(0.037)
    # Recheck live fixed coefficients with the original nontrivial mask.
    actual, expected = op(x, h, g, src, dst, mask, r, i), reference()
    for l in expected:
        torch.testing.assert_close(actual[l], expected[l], atol=2e-6, rtol=3e-5)
    # The first frozen inference layer does not need a charged-input VJP.
    # Exercise the edge-owned geometric/phase reduction independently of DX.
    x = {l: value.detach() for l, value in x.items()}
    actual, expected = op(x, h, g, src, dst, mask, r, i), reference()
    active = [*h.values(), g, r, i]
    got_grad = torch.autograd.grad(loss(actual), active)
    ref_grad = torch.autograd.grad(loss(expected), active)
    for got, ref in zip(got_grad, ref_grad):
        torch.testing.assert_close(got, ref, atol=3e-6, rtol=1e-4)


@pytest.mark.parametrize("schedule", ['split', 'owned', 'destination', 'destination_coalesced',
                                      'destination_groups1', 'destination_groups2', 'destination_groups4',
                                      'destination_launch1', 'destination_launch2', 'destination_launch8',
                                      'destination_reg128'])
@pytest.mark.parametrize("channels,lmax,heads,angular", [(12, 2, 2, True), (13, 1, 1, False)])
def test_split_csr_adjoint_matches_joint(channels, lmax, heads, angular, schedule, monkeypatch):
    torch.manual_seed(20261017)
    edge = EdgeTensorProduct.create(channels=channels, input_lmax=lmax,
                                    edge_lmax=lmax, device="cuda")
    op = CoherentGraphMessage(edge.tp, input_lmax=lmax, edge_lmax=lmax,
                               heads=heads, angular=angular, adjoint_splits=1)
    n, e = 11, 47
    x = {l: (torch.randn(n, channels, 2*l+1, device="cuda")*0.2).requires_grad_()
         for l in range(lmax+1)}
    h = {l: (torch.randn(e, 1, 2*l+1, device="cuda")*0.2).requires_grad_()
         for l in range(lmax+1)}
    g = (torch.randn(e, edge.gate_width, device="cuda")*0.2).requires_grad_()
    phases = heads*(lmax+1 if angular else 1)
    r = torch.randn(e, phases, 1, device="cuda", requires_grad=True)
    i = torch.randn(e, phases, 1, device="cuda", requires_grad=True)
    src, dst = torch.randint(n, (e,), device="cuda"), torch.randint(n, (e,), device="cuda")
    mask = (torch.arange(e, device="cuda") % 4 != 0).float()
    probes = {l: torch.randn(n, 3, edge.tp.path_counts_by_l[l]*channels, 2*l+1,
                             device="cuda") for l in range(lmax+1)}
    inputs = [*x.values(), *h.values(), g, r, i]
    def gradients():
        values = op(x, h, g, src, dst, mask, r, i)
        return torch.autograd.grad(sum((values[l]*probes[l]).sum() for l in values), inputs)
    expected = gradients()
    if schedule != 'split':
        import mace_ictc.hp._coherent_csr_split as split_module
        if schedule == 'owned':
            from mace_ictc.hp._coherent_csr_owned import adjoint
        else:
            from mace_ictc.hp._coherent_destination import adjoint
            if (schedule == 'destination_coalesced' or schedule.startswith('destination_groups')
                    or schedule.startswith('destination_launch') or schedule == 'destination_reg128'):
                from functools import partial
                adjoint = partial(adjoint, atomic_x=True, factored=True, coalesced=True,
                    group_paths=int(schedule[-1]) if schedule.startswith('destination_groups') else 0,
                    block_warps=int(schedule[-1]) if schedule.startswith('destination_launch') else 4,
                    register_limit=128 if schedule == 'destination_reg128' else 0)
        monkeypatch.setattr(split_module, 'adjoint', adjoint)
    op.adjoint_splits = 4
    actual = gradients()
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, atol=3e-6, rtol=1e-4)


@pytest.mark.parametrize('channels,heads,edges', [(12, 2, 47), (13, 1, 47), (12, 2, 0)])
def test_destination_frozen_input_first_layer(channels, heads, edges, monkeypatch):
    """No DX scratch when the original frozen X has no gradient consumer."""
    import mace_ictc.hp.coherent as coherent
    from mace_ictc.hp._coherent_destination import adjoint
    torch.manual_seed(20261017)
    edge = EdgeTensorProduct.create(channels=channels, input_lmax=0,
                                    edge_lmax=2, output_lmax=2, device='cuda')
    op = CoherentGraphMessage(edge.tp, input_lmax=0, edge_lmax=2, heads=heads,
                             angular=True, adjoint_splits=1)
    n = 11
    x = {0: torch.randn(n, channels, 1, device='cuda')*0.2}
    h = {l: (torch.randn(edges, 1, 2*l+1, device='cuda')*0.2).requires_grad_()
         for l in range(3)}
    g = (torch.randn(edges, edge.gate_width, device='cuda')*0.2).requires_grad_()
    r = torch.randn(edges, heads*3, 1, device='cuda', requires_grad=True)
    i = torch.randn_like(r, requires_grad=True)
    src, dst = torch.randint(n, (edges,), device='cuda'), torch.randint(n, (edges,), device='cuda')
    mask = (torch.arange(edges, device='cuda') % 4 != 0).float()
    probes = {l: torch.randn(n, 3, edge.tp.path_counts_by_l[l]*channels, 2*l+1,
                             device='cuda') for l in range(3)}
    active = [*h.values(), g, r, i]
    def gradients():
        values = op(x, h, g, src, dst, mask, r, i)
        # Also exercise an unconsumed degree/cotangent.
        return torch.autograd.grad(sum((values[l]*probes[l]).sum() for l in (0, 2)), active)
    expected = gradients()
    original = coherent._Coherent.backward
    def destination(ctx, *grads):
        if torch.is_grad_enabled():
            return original(ctx, *grads)
        xx, sh, w, gates, real, imag, ss, dd, mm = ctx.saved_tensors
        needed = tuple(ctx.needs_input_grad[k] for k in (0, 1, 3, 4, 5))
        assert needed == (False, True, True, True, True)
        dx, dh, dg, dr, di = adjoint(xx, sh, w, gates, real, imag, ss, dd, mm,
            ctx.p, ctx.projection, needed, grads, atomic_x=True, factored=True, coalesced=True)
        assert dx is None
        return dx, dh, None, dg, dr, di, None, None, None, None, None
    monkeypatch.setattr(coherent._Coherent, 'backward', staticmethod(destination))
    for got, ref in zip(gradients(), expected):
        torch.testing.assert_close(got, ref, atol=3e-6, rtol=1e-4)
