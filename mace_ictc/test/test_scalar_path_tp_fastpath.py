"""Regression tests for the scalar-input path-preserving TP fast path."""

import torch

from mace_ictc.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct


def test_scalar_path_fast_path_matches_index_add_forward_and_grad():
    torch.manual_seed(7)
    allowed_paths = [(0, l, l) for l in range(4)]
    fast = EdgeWeightedPathPreservingTensorProduct(
        channels=8,
        lmax=3,
        allowed_paths=allowed_paths,
        internal_compute_dtype=torch.float64,
    ).double()
    old = EdgeWeightedPathPreservingTensorProduct(
        channels=8,
        lmax=3,
        allowed_paths=allowed_paths,
        internal_compute_dtype=torch.float64,
    ).double()
    old.load_state_dict(fast.state_dict())
    old._use_scalar_direct_fast_path = False

    assert fast._use_scalar_direct_fast_path
    assert fast.path_counts_by_l == {0: 1, 1: 1, 2: 1, 3: 1}

    x1 = {0: torch.randn(17, 8, 1, dtype=torch.float64, requires_grad=True)}
    x2 = {
        l: torch.randn(17, 1, 2 * l + 1, dtype=torch.float64, requires_grad=True)
        for l in range(4)
    }
    gates = torch.randn(17, fast.num_paths * fast.channels, dtype=torch.float64, requires_grad=True)

    out_fast = fast(x1, x2, gates)
    out_old = old(x1, x2, gates)
    for l in range(4):
        assert torch.equal(out_fast[l], out_old[l])

    loss_fast = sum(v.sum() for v in out_fast.values())
    loss_old = sum(v.sum() for v in out_old.values())
    inputs = [x1[0], *(x2[l] for l in range(4)), gates]
    grads_fast = torch.autograd.grad(loss_fast, inputs, retain_graph=True)
    grads_old = torch.autograd.grad(loss_old, inputs)
    for grad_fast, grad_old in zip(grads_fast, grads_old):
        assert torch.equal(grad_fast, grad_old)


def test_scalar_path_fast_path_does_not_trigger_for_general_paths():
    tp = EdgeWeightedPathPreservingTensorProduct(
        channels=4,
        lmax=3,
        internal_compute_dtype=torch.float64,
    )
    assert not tp._use_scalar_direct_fast_path


def test_scalar_identity_projector_fast_path_matches_matmul_path_float32():
    torch.manual_seed(17)
    allowed_paths = [(0, l, l) for l in range(4)]
    fast = EdgeWeightedPathPreservingTensorProduct(
        channels=8,
        lmax=3,
        allowed_paths=allowed_paths,
        internal_compute_dtype=torch.float32,
    )
    matmul = EdgeWeightedPathPreservingTensorProduct(
        channels=8,
        lmax=3,
        allowed_paths=allowed_paths,
        internal_compute_dtype=torch.float32,
    )
    matmul.load_state_dict(fast.state_dict())
    matmul._use_scalar_identity_projector_fast_path = False

    assert fast._use_scalar_identity_projector_fast_path
    assert fast._scalar_direct_group_scales == [1.0, 1.0, 1.0, 1.0]

    x1 = {0: torch.randn(19, 8, 1, dtype=torch.float32, requires_grad=True)}
    x2 = {
        l: torch.randn(19, 1, 2 * l + 1, dtype=torch.float32, requires_grad=True)
        for l in range(4)
    }
    gates = torch.randn(19, fast.num_paths * fast.channels, dtype=torch.float32, requires_grad=True)

    out_fast = fast(x1, x2, gates)
    out_matmul = matmul(x1, x2, gates)
    for l in range(4):
        assert torch.allclose(out_fast[l], out_matmul[l], rtol=1e-6, atol=1e-6)

    loss_fast = sum(v.sum() for v in out_fast.values())
    loss_matmul = sum(v.sum() for v in out_matmul.values())
    inputs = [x1[0], *(x2[l] for l in range(4)), gates]
    grads_fast = torch.autograd.grad(loss_fast, inputs, retain_graph=True)
    grads_matmul = torch.autograd.grad(loss_matmul, inputs)
    for grad_fast, grad_matmul in zip(grads_fast, grads_matmul):
        assert torch.allclose(grad_fast, grad_matmul, rtol=1e-6, atol=1e-6)
