"""Physics test for the learned pairwise C6 dispersion term."""

from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from chorus.models.dispersion import (
    LongRangeDispersion,
    ManyBodyDispersion,
    ManyBodyDispersionSLQ,
    PairwiseDispersion,
    _estimate_dispersion_max_neighbors,
    dispersion_cutoff_is_single_image_exact,
    dispersion_deployment_graph_rule,
    dispersion_neighbor_list,
    dispersion_mode_needs_deployment_edges,
    dispersion_train_deploy_graph_compatibility,
    dispersion_mode_uses_canonical_edges,
    dispersion_mode_uses_cutoff_edges,
    dispersion_training_graph_rule,
    normalize_dispersion_edges,
)
from chorus.models.long_range import (
    _build_assignment_offsets,
    _prepare_frac_for_boundary,
    _prepare_frac_for_boundary_batched,
    apply_periodic_dipole_pme_field,
    apply_periodic_dipole_pme_field_batched,
    build_periodic_dipole_pme_kernel,
    build_periodic_dipole_pme_kernel_batched,
    periodic_dipole_pme_field,
    periodic_dipole_pme_field_batched,
)
from chorus.test.test_multipole_long_range import _build_model, _neighbor_list, _random_rotation
from chorus.utils.checkpoint_metadata import (
    resolve_model_architecture,
    validate_dispersion_deployment_graph_rule,
    validate_dispersion_train_deploy_graph_compatibility,
    validate_dispersion_training_graph_rule,
)


def test_dispersion_physics():
    dtype = torch.float64
    torch.manual_seed(0)
    c = 8
    disp = PairwiseDispersion(feature_dim=c).to(dtype)
    feats = torch.randn(2, c, dtype=dtype)
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 0])

    def energy(r):
        lengths = torch.tensor([r, r], dtype=dtype)
        return disp(feats, src, dst, lengths).sum()

    e_close, e_mid, e_far = energy(2.0), energy(4.0), energy(8.0)
    # attractive and monotonically decaying toward 0 with separation
    assert e_close < 0 and e_mid < 0, "dispersion must be attractive (negative)"
    assert e_close < e_mid < e_far <= 1e-12, "dispersion must decay toward 0 with distance"
    # Becke-Johnson damping keeps it finite as r -> 0 (short-range network owns contact)
    assert torch.isfinite(energy(0.05)), "dispersion not finite at small r"

    # forces flow and are attractive (atoms pulled together)
    pos = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]], dtype=dtype, requires_grad=True)
    lengths = (pos[dst] - pos[src]).norm(dim=-1)
    e = disp(feats, src, dst, lengths).sum()
    (grad,) = torch.autograd.grad(e, pos)
    force = -grad
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "bad force"
    assert force[0, 0] > 0 and force[1, 0] < 0, "dispersion force is not attractive"

    # rotation/translation invariance is by construction (depends only on |r_ij|): spot-check
    R = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))[0]
    pos_r = pos.detach() @ R.T + torch.tensor([1.3, -0.7, 0.2], dtype=dtype)
    lengths_r = (pos_r[dst] - pos_r[src]).norm(dim=-1)
    e_r = disp(feats, src, dst, lengths_r).sum()
    assert torch.allclose(e.detach(), e_r, atol=1e-10), "not rotation/translation invariant"


def test_long_range_dispersion_wrapper_matches_pairwise_edge_mode():
    dtype = torch.float64
    torch.manual_seed(4)
    c = 8
    pair = PairwiseDispersion(feature_dim=c).to(dtype)
    wrapped = LongRangeDispersion(feature_dim=c, mode="pairwise-c6", cutoff=0.0, pbc=False).to(dtype)
    wrapped.term.load_state_dict(pair.state_dict())

    feats = torch.randn(4, c, dtype=dtype)
    pos = torch.randn(4, 3, dtype=dtype)
    batch = torch.zeros(4, dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 20.0
    src = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    dst = torch.tensor([1, 0, 3, 2, 2, 0], dtype=torch.long)
    lengths = (pos[dst] - pos[src]).norm(dim=-1)

    e_pair = pair(feats, src, dst, lengths)
    e_wrapped = wrapped(
        feats,
        pos,
        batch,
        cell,
        edge_src=src,
        edge_dst=dst,
        edge_lengths=lengths,
        cutoff=0.0,
        pbc=False,
    )
    assert torch.allclose(e_pair, e_wrapped, atol=0.0, rtol=0.0), "wrapper changed pairwise C6 numerics"


def test_pairwise_c6_outer_cutoff_switch_has_zero_boundary_force():
    dtype = torch.float64
    torch.manual_seed(5)
    model = LongRangeDispersion(
        feature_dim=6,
        mode="pairwise-c6",
        cutoff=5.0,
        pbc=False,
        switch_width=0.5,
        neighbor_method="bruteforce",
    ).to(dtype)
    feats = torch.randn(2, 6, dtype=dtype)
    batch = torch.zeros(2, dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 20.0

    def energy_and_force(r: float):
        pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=dtype, requires_grad=True)
        e = model(
            feats,
            pos,
            batch,
            cell,
            edge_src=torch.zeros(0, dtype=torch.long),
            edge_dst=torch.zeros(0, dtype=torch.long),
            edge_lengths=pos.new_zeros(0),
            cutoff=5.0,
            pbc=False,
        ).sum()
        (grad,) = torch.autograd.grad(e, pos)
        return e.detach(), (-grad).detach()

    e_inside, f_inside = energy_and_force(4.75)
    e_cut, f_cut = energy_and_force(5.0)
    e_out, f_out = energy_and_force(5.25)

    assert e_inside < 0.0
    assert f_inside.abs().sum() > 0.0
    assert torch.allclose(e_cut, e_cut.new_zeros(()), atol=1e-12, rtol=0.0)
    assert torch.allclose(f_cut, torch.zeros_like(f_cut), atol=1e-10, rtol=0.0)
    assert torch.allclose(e_out, e_out.new_zeros(()), atol=1e-12, rtol=0.0)
    assert torch.allclose(f_out, torch.zeros_like(f_out), atol=1e-10, rtol=0.0)


def test_pairwise_c6_min_cutoff_switch_has_zero_boundary_force():
    dtype = torch.float64
    torch.manual_seed(6)
    model = LongRangeDispersion(
        feature_dim=6,
        mode="pairwise-c6",
        cutoff=6.0,
        pbc=False,
        min_cutoff=2.0,
        switch_width=0.5,
        neighbor_method="bruteforce",
    ).to(dtype)
    feats = torch.randn(2, 6, dtype=dtype)
    batch = torch.zeros(2, dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 20.0

    def energy_and_force(r: float):
        pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=dtype, requires_grad=True)
        e = model(
            feats,
            pos,
            batch,
            cell,
            edge_src=torch.zeros(0, dtype=torch.long),
            edge_dst=torch.zeros(0, dtype=torch.long),
            edge_lengths=pos.new_zeros(0),
            cutoff=6.0,
            pbc=False,
        ).sum()
        (grad,) = torch.autograd.grad(e, pos)
        return e.detach(), (-grad).detach()

    e_below, f_below = energy_and_force(1.75)
    e_min, f_min = energy_and_force(2.0)
    e_ramp, f_ramp = energy_and_force(2.25)

    assert torch.allclose(e_below, e_below.new_zeros(()), atol=1e-12, rtol=0.0)
    assert torch.allclose(f_below, torch.zeros_like(f_below), atol=1e-10, rtol=0.0)
    assert torch.allclose(e_min, e_min.new_zeros(()), atol=1e-12, rtol=0.0)
    assert torch.allclose(f_min, torch.zeros_like(f_min), atol=1e-10, rtol=0.0)
    assert e_ramp < 0.0
    assert f_ramp.abs().sum() > 0.0


def test_pairwise_c6_explicit_edges_use_module_cutoff_window_when_cutoff_arg_is_zero():
    dtype = torch.float64
    torch.manual_seed(7)
    model = LongRangeDispersion(
        feature_dim=3,
        mode="pairwise-c6",
        cutoff=5.0,
        pbc=False,
        switch_width=0.5,
    ).to(dtype)
    feats = torch.randn(2, 3, dtype=dtype)
    batch = torch.zeros(2, dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 20.0
    edge_src = torch.tensor([0], dtype=torch.long)
    edge_dst = torch.tensor([1], dtype=torch.long)

    def explicit_energy(r_value: float) -> torch.Tensor:
        pos = torch.tensor([[0.0, 0.0, 0.0], [r_value, 0.0, 0.0]], dtype=dtype)
        edge_vec = torch.tensor([[r_value, 0.0, 0.0]], dtype=dtype)
        edge_len = torch.tensor([r_value], dtype=dtype)
        return model(
            feats,
            pos,
            batch,
            cell,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_lengths=edge_len,
            edge_vec=edge_vec,
            cutoff=0.0,
            pbc=False,
        ).sum()

    assert explicit_energy(4.75).abs() > 0.0
    assert torch.allclose(explicit_energy(5.0), torch.zeros((), dtype=dtype), atol=1e-12, rtol=0.0)


def test_long_range_dispersion_wrapper_forwards_neighbor_cap(monkeypatch):
    dtype = torch.float64
    captured = {}

    def fake_dispersion_neighbor_list(*args, **kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([1, 0], dtype=torch.long),
            torch.zeros(2, 3, dtype=torch.long),
        )

    monkeypatch.setattr("chorus.models.dispersion.dispersion_neighbor_list", fake_dispersion_neighbor_list)
    wrapped = LongRangeDispersion(
        feature_dim=4,
        mode="pairwise-c6",
        cutoff=5.0,
        pbc=True,
        max_num_neighbors=7,
        neighbor_method="cell",
        bruteforce_threshold=321,
        allow_large_bruteforce_fallback=True,
    ).to(dtype)
    feats = torch.randn(2, 4, dtype=dtype)
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype)
    batch = torch.zeros(2, dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 12.0
    fallback_src = torch.tensor([0], dtype=torch.long)
    fallback_dst = torch.tensor([1], dtype=torch.long)
    fallback_len = torch.tensor([2.0], dtype=dtype)

    out = wrapped(
        feats,
        pos,
        batch,
        cell,
        edge_src=fallback_src,
        edge_dst=fallback_dst,
        edge_lengths=fallback_len,
        cutoff=5.0,
        pbc=True,
    )
    assert out.shape == (2, 1)
    assert captured["max_num_neighbors"] == 7
    assert captured["canonical_undirected"] is False
    assert captured["method"] == "cell"
    assert captured["bruteforce_threshold"] == 321
    assert captured["allow_large_bruteforce_fallback"] is True


def test_many_body_dispersion_is_finite_invariant_and_nonadditive():
    dtype = torch.float64
    c = 4
    mbd = ManyBodyDispersion(feature_dim=c).to(dtype)

    def inv_softplus(x):
        return torch.log(torch.expm1(torch.as_tensor(x, dtype=dtype)))

    with torch.no_grad():
        for p in mbd.parameters():
            p.zero_()
        mbd.alpha_head[-1].bias.fill_(inv_softplus(1.2))
        mbd.omega_head[-1].bias.fill_(inv_softplus(0.7))
        mbd.coupling_scale.fill_(0.2)
        mbd.beta_raw.fill_(inv_softplus(1.1))

    def complete_directed_edges(pos):
        n = pos.shape[0]
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(j)
                    dst.append(i)
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        return src, dst, pos[dst] - pos[src]

    pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.7, 2.6, 0.0]], dtype=dtype, requires_grad=True)
    feats = torch.zeros(3, c, dtype=dtype)
    batch = torch.zeros(3, dtype=torch.long)
    src, dst, edge_vec = complete_directed_edges(pos)
    e3 = mbd(feats, batch, src, dst, edge_vec).sum()
    assert torch.isfinite(e3), "MBD energy is not finite"
    (grad,) = torch.autograd.grad(e3, pos, create_graph=True)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0, "MBD force gradient did not flow"

    R = _random_rotation(dtype)
    pos_r = (pos.detach() @ R.T).requires_grad_(True)
    src_r, dst_r, edge_vec_r = complete_directed_edges(pos_r)
    e3_r = mbd(feats, batch, src_r, dst_r, edge_vec_r).sum()
    assert torch.allclose(e3.detach(), e3_r.detach(), atol=1e-10), "MBD energy is not rotation invariant"

    pair_sum = pos.new_tensor(0.0)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        pp = pos.detach()[torch.tensor([a, b])].clone().requires_grad_(True)
        pf = feats[:2]
        pb = torch.zeros(2, dtype=torch.long)
        ps, pd, pev = complete_directed_edges(pp)
        pair_sum = pair_sum + mbd(pf, pb, ps, pd, pev).sum()
    assert (e3.detach() - pair_sum.detach()).abs() > 1e-10, "MBD collapsed to pairwise-additive energy"


def test_many_body_dispersion_slq_basis_matches_dense_oracle():
    dtype = torch.float64
    c = 4
    torch.manual_seed(17)
    dense = ManyBodyDispersion(feature_dim=c).to(dtype)
    slq = ManyBodyDispersionSLQ(feature_dim=c, probe_mode="basis", lanczos_steps=32).to(dtype)
    slq.load_state_dict(dense.state_dict(), strict=False)
    slq.eval()

    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.7, 0.2, 0.0], [0.4, 2.4, 0.3], [1.5, 1.2, 2.1]],
        dtype=dtype,
        requires_grad=True,
    )
    feats = torch.randn(4, c, dtype=dtype)
    batch = torch.zeros(4, dtype=torch.long)
    src, dst = [], []
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i != j:
                src.append(j)
                dst.append(i)
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    edge_vec = pos[dst] - pos[src]

    e_dense = dense(feats, batch, src, dst, edge_vec).sum()
    e_slq = slq(feats, batch, src, dst, edge_vec).sum()
    assert torch.allclose(e_dense, e_slq, atol=2e-8, rtol=2e-8), (
        f"basis SLQ does not match dense MBD: dense={e_dense.item():.8e}, slq={e_slq.item():.8e}"
    )

    (g_dense,) = torch.autograd.grad(e_dense, pos, retain_graph=True)
    (g_slq,) = torch.autograd.grad(e_slq, pos)
    assert torch.allclose(g_dense, g_slq, atol=2e-7, rtol=2e-7), "basis SLQ force gradient != dense MBD"


def test_mbd_training_ramp_count_persists_in_state_dict():
    slq = ManyBodyDispersionSLQ(feature_dim=4)
    slq._mbd_ramp_count = 17

    state = slq.state_dict()
    assert "_mbd_ramp_count_state" in state
    assert int(state["_mbd_ramp_count_state"].item()) == 17

    restored = ManyBodyDispersionSLQ(feature_dim=4)
    restored.load_state_dict(state, strict=True)

    assert restored._mbd_ramp_count == 17


def test_many_body_dispersion_slq_operator_backend_contract():
    dtype = torch.float64
    c = 3
    torch.manual_seed(19)
    default = ManyBodyDispersionSLQ(feature_dim=c, probe_mode="basis", lanczos_steps=12).to(dtype)
    edge_sparse = ManyBodyDispersionSLQ(
        feature_dim=c,
        probe_mode="basis",
        lanczos_steps=12,
        operator_backend="edge_sparse",
    ).to(dtype)
    edge_sparse.load_state_dict(default.state_dict())

    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.1, 0.4, 0.0], [0.2, 2.5, 0.3]],
        dtype=dtype,
        requires_grad=True,
    )
    feats = torch.randn(3, c, dtype=dtype)
    batch = torch.zeros(3, dtype=torch.long)
    src = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    dst = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    edge_vec = pos[dst] - pos[src]

    e_default = default(feats, batch, src, dst, edge_vec).sum()
    e_edge = edge_sparse(feats, batch, src, dst, edge_vec).sum()
    assert torch.allclose(e_default, e_edge, atol=0.0, rtol=0.0)

    pme = ManyBodyDispersionSLQ(
        feature_dim=c,
        probe_mode="basis",
        lanczos_steps=12,
        operator_backend="pme_fft",
        pme_mesh_size=8,
    ).to(dtype)
    assert pme.operator_backend == "pme_fft"
    with pytest.raises(ValueError, match="Unsupported SLQ-MBD operator backend"):
        ManyBodyDispersionSLQ(feature_dim=c, operator_backend="not-a-backend")


def test_periodic_dipole_pme_field_kernel_matches_one_shot():
    dtype = torch.float64
    torch.manual_seed(20)
    cell = torch.tensor(
        [[8.0, 0.2, 0.0], [0.1, 8.5, 0.0], [0.0, 0.0, 9.0]],
        dtype=dtype,
    )
    pos = torch.tensor(
        [[0.7, 0.4, 0.3], [2.2, 1.1, 0.8], [1.4, 2.6, 1.7], [3.0, 2.5, 2.2]],
        dtype=dtype,
    )
    frac = _prepare_frac_for_boundary(pos, cell, boundary="periodic", slab_padding_factor=1)
    assignment = "cic"
    offsets = _build_assignment_offsets(assignment)
    dipoles = torch.randn(pos.size(0), 3, 3, dtype=dtype)

    direct = periodic_dipole_pme_field(
        frac,
        dipoles,
        cell=cell,
        mesh_size=8,
        assignment=assignment,
        assignment_offsets=offsets,
    )
    k_cart, k2, spectral = build_periodic_dipole_pme_kernel(
        cell=cell,
        mesh_size=8,
        assignment=assignment,
        device=dipoles.device,
        dtype=dipoles.dtype,
    )
    planned = apply_periodic_dipole_pme_field(
        frac,
        dipoles,
        mesh_size=8,
        assignment=assignment,
        assignment_offsets=offsets,
        k_cart=k_cart,
        k2=k2,
        spectral=spectral,
    )
    assert torch.allclose(planned, direct, atol=0.0, rtol=0.0)


def test_periodic_dipole_pme_field_matches_edge_sparse_sign():
    dtype = torch.float64
    box = 24.0
    r = 6.0
    cell = torch.eye(3, dtype=dtype) * box
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=dtype)
    frac = _prepare_frac_for_boundary(pos, cell, boundary="periodic", slab_padding_factor=1)
    dipoles = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]], dtype=dtype)
    assignment = "pcs"
    field = periodic_dipole_pme_field(
        frac,
        dipoles,
        cell=cell,
        mesh_size=48,
        assignment=assignment,
        assignment_offsets=_build_assignment_offsets(assignment),
    )[:, 0, :]

    rvec = pos[1] - pos[0]
    rhat = rvec / rvec.norm()
    bare = ((3.0 * torch.outer(rhat, rhat) - torch.eye(3, dtype=dtype)) / rvec.norm().pow(3)) @ dipoles[0, 0]
    ratio = field[1, 0] / bare[0]
    assert ratio > 0.9, f"PME dipole tensor sign flipped relative to edge_sparse: ratio={ratio.item():.3e}"


def test_periodic_dipole_pme_field_batched_matches_per_graph():
    dtype = torch.float64
    torch.manual_seed(22)
    cell = torch.stack(
        [
            torch.tensor([[8.0, 0.2, 0.0], [0.1, 8.5, 0.0], [0.0, 0.0, 9.0]], dtype=dtype),
            torch.tensor([[9.0, 0.1, 0.0], [0.3, 8.2, 0.2], [0.0, 0.1, 9.5]], dtype=dtype),
        ],
        dim=0,
    )
    pos = torch.tensor(
        [
            [0.7, 0.4, 0.3],
            [2.2, 1.1, 0.8],
            [1.4, 2.6, 1.7],
            [0.6, 0.7, 0.8],
            [2.1, 1.5, 1.2],
        ],
        dtype=dtype,
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    assignment = "cic"
    offsets = _build_assignment_offsets(assignment)
    dipoles = torch.randn(pos.size(0), 2, 3, dtype=dtype)
    frac = _prepare_frac_for_boundary_batched(
        pos,
        batch,
        cell,
        boundary="periodic",
        slab_padding_factor=1,
    )

    direct_batched = periodic_dipole_pme_field_batched(
        frac,
        batch,
        dipoles,
        cell=cell,
        mesh_size=8,
        assignment=assignment,
        assignment_offsets=offsets,
    )
    k_cart, k2, spectral = build_periodic_dipole_pme_kernel_batched(
        cell=cell,
        mesh_size=8,
        assignment=assignment,
        device=dipoles.device,
        dtype=dipoles.dtype,
    )
    planned_batched = apply_periodic_dipole_pme_field_batched(
        frac,
        batch,
        dipoles,
        mesh_size=8,
        assignment=assignment,
        assignment_offsets=offsets,
        k_cart=k_cart,
        k2=k2,
        spectral=spectral,
    )

    per_graph: list[torch.Tensor] = []
    for g in range(cell.size(0)):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        per_graph.append(
            periodic_dipole_pme_field(
                frac.index_select(0, idx),
                dipoles.index_select(0, idx),
                cell=cell[g],
                mesh_size=8,
                assignment=assignment,
                assignment_offsets=offsets,
            )
        )
    expected = torch.cat(per_graph, dim=0)
    assert torch.allclose(planned_batched, direct_batched, atol=0.0, rtol=0.0)
    assert torch.allclose(direct_batched, expected, atol=1e-10, rtol=1e-10)


def test_periodic_dipole_pme_field_removes_discrete_self_action():
    dtype = torch.float64
    torch.manual_seed(23)
    assignment = "pcs"
    offsets = _build_assignment_offsets(assignment)
    cell = torch.tensor(
        [[8.0, 0.2, 0.1], [0.1, 8.5, 0.3], [0.2, 0.0, 9.0]],
        dtype=dtype,
    )
    pos = torch.tensor([[1.37, 2.11, 3.42]], dtype=dtype)
    frac = _prepare_frac_for_boundary(pos, cell, boundary="periodic", slab_padding_factor=1)
    dipoles = torch.randn(1, 3, 3, dtype=dtype)

    field = periodic_dipole_pme_field(
        frac,
        dipoles,
        cell=cell,
        mesh_size=8,
        assignment=assignment,
        assignment_offsets=offsets,
    )
    assert torch.allclose(field, torch.zeros_like(field), atol=1e-10, rtol=0.0)

    cell_b = torch.stack([cell, cell * 1.1], dim=0)
    pos_b = torch.cat([pos, pos + torch.tensor([[0.2, 0.3, 0.4]], dtype=dtype)], dim=0)
    batch = torch.tensor([0, 1], dtype=torch.long)
    frac_b = _prepare_frac_for_boundary_batched(
        pos_b,
        batch,
        cell_b,
        boundary="periodic",
        slab_padding_factor=1,
    )
    dipoles_b = torch.randn(2, 2, 3, dtype=dtype)
    field_b = periodic_dipole_pme_field_batched(
        frac_b,
        batch,
        dipoles_b,
        cell=cell_b,
        mesh_size=8,
        assignment=assignment,
        assignment_offsets=offsets,
    )
    assert torch.allclose(field_b, torch.zeros_like(field_b), atol=1e-10, rtol=0.0)


def test_many_body_dispersion_slq_pme_fft_smoke_and_lattice_shift():
    dtype = torch.float64
    c = 3
    torch.manual_seed(21)
    model = LongRangeDispersion(
        feature_dim=c,
        mode="mbd-slq",
        cutoff=0.0,
        slq_num_probes=2,
        slq_lanczos_steps=5,
        mbd_operator_backend="pme_fft",
    ).to(dtype)
    model.term.probe_mode = "basis"
    model.term.pme_mesh_size = 8
    model.eval()

    cell = (torch.eye(3, dtype=dtype) * 8.0).reshape(1, 3, 3)
    pos = torch.tensor(
        [[0.7, 0.4, 0.3], [2.2, 1.1, 0.8], [1.4, 2.6, 1.7]],
        dtype=dtype,
        requires_grad=True,
    )
    feats = torch.randn(3, c, dtype=dtype)
    batch = torch.zeros(3, dtype=torch.long)
    empty = torch.zeros(0, dtype=torch.long)
    empty_shift = torch.zeros(0, 3, dtype=dtype)

    e = model(
        feats,
        pos,
        batch,
        cell,
        edge_src=empty,
        edge_dst=empty,
        edge_lengths=torch.zeros(0, dtype=dtype),
        edge_vec=empty_shift,
    ).sum()
    assert torch.isfinite(e)
    (grad,) = torch.autograd.grad(e, pos, retain_graph=True)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    shifted = (pos.detach() + cell[0, 0]).requires_grad_(True)
    e_shift = model(
        feats,
        shifted,
        batch,
        cell,
        edge_src=empty,
        edge_dst=empty,
        edge_lengths=torch.zeros(0, dtype=dtype),
        edge_vec=empty_shift,
    ).sum()
    assert torch.allclose(e.detach(), e_shift.detach(), atol=1e-8, rtol=1e-8)


def test_many_body_dispersion_slq_pme_fft_skips_dispersion_neighbor_list(monkeypatch):
    dtype = torch.float64

    def fail_neighbor_list(*args, **kwargs):
        raise AssertionError("pme_fft MBD should not build a dispersion neighbor list")

    monkeypatch.setattr("chorus.models.dispersion.dispersion_neighbor_list", fail_neighbor_list)
    model = LongRangeDispersion(
        feature_dim=2,
        mode="mbd-slq",
        cutoff=6.0,
        slq_num_probes=1,
        slq_lanczos_steps=4,
        mbd_operator_backend="pme_fft",
    ).to(dtype)
    model.term.probe_mode = "basis"
    model.eval()
    model.term.pme_mesh_size = 8

    cell = torch.stack(
        [
            torch.eye(3, dtype=dtype) * 9.0,
            torch.eye(3, dtype=dtype) * 10.0,
            torch.eye(3, dtype=dtype) * 11.0,
        ],
        dim=0,
    )
    pos = torch.tensor(
        [
            [0.4, 0.3, 0.2],
            [1.7, 1.1, 0.9],
            [0.6, 0.7, 0.8],
            [2.1, 1.5, 1.2],
            [3.4, 2.8, 1.9],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    feats = torch.randn(5, 2, dtype=dtype)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    bogus_src = torch.tensor([999], dtype=torch.long)
    bogus_dst = torch.tensor([998], dtype=torch.long)
    energy = model(
        feats,
        pos,
        batch,
        cell,
        edge_src=bogus_src,
        edge_dst=bogus_dst,
        edge_lengths=torch.ones(1, dtype=dtype),
        edge_vec=None,
    ).sum()
    assert torch.isfinite(energy)
    (grad,) = torch.autograd.grad(energy, pos)
    assert torch.isfinite(grad).all()


def test_many_body_dispersion_slq_pme_fft_batch_matches_separate_graphs():
    dtype = torch.float64
    c = 3
    torch.manual_seed(23)
    model = LongRangeDispersion(
        feature_dim=c,
        mode="mbd-slq",
        cutoff=0.0,
        slq_num_probes=2,
        slq_lanczos_steps=5,
        mbd_operator_backend="pme_fft",
        mbd_pme_mesh_size=8,
    ).to(dtype)
    model.term.probe_mode = "basis"
    model.eval()

    cell = torch.stack(
        [
            torch.eye(3, dtype=dtype) * 8.0,
            torch.tensor([[9.0, 0.2, 0.0], [0.1, 8.5, 0.0], [0.0, 0.0, 9.5]], dtype=dtype),
        ],
        dim=0,
    )
    pos0 = torch.tensor([[0.7, 0.4, 0.3], [2.2, 1.1, 0.8]], dtype=dtype)
    pos1 = torch.tensor([[0.6, 0.7, 0.8], [2.1, 1.5, 1.2], [3.4, 2.8, 1.9]], dtype=dtype)
    feats0 = torch.randn(pos0.size(0), c, dtype=dtype)
    feats1 = torch.randn(pos1.size(0), c, dtype=dtype)
    pos = torch.cat([pos0, pos1], dim=0).requires_grad_(True)
    feats = torch.cat([feats0, feats1], dim=0)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    empty = torch.zeros(0, dtype=torch.long)
    empty_vec = torch.zeros(0, 3, dtype=dtype)

    batched = model(
        feats,
        pos,
        batch,
        cell,
        edge_src=empty,
        edge_dst=empty,
        edge_lengths=torch.zeros(0, dtype=dtype),
        edge_vec=empty_vec,
    )
    (grad_batched,) = torch.autograd.grad(batched.sum(), pos, retain_graph=True)

    separate_energy = []
    separate_grad = []
    for g, (p0, f0) in enumerate(((pos0, feats0), (pos1, feats1))):
        p = p0.clone().requires_grad_(True)
        b = torch.zeros(p.size(0), dtype=torch.long)
        out = model(
            f0,
            p,
            b,
            cell[g : g + 1],
            edge_src=empty,
            edge_dst=empty,
            edge_lengths=torch.zeros(0, dtype=dtype),
            edge_vec=empty_vec,
        )
        (grad,) = torch.autograd.grad(out.sum(), p)
        separate_energy.append(out.detach())
        separate_grad.append(grad.detach())

    expected_energy = torch.cat(separate_energy, dim=0)
    expected_grad = torch.cat(separate_grad, dim=0)
    assert torch.allclose(batched.detach(), expected_energy, atol=1e-8, rtol=1e-8)
    assert torch.allclose(grad_batched.detach(), expected_grad, atol=1e-8, rtol=1e-8)


def test_many_body_dispersion_slq_pme_fft_builds_batched_kernel_once(monkeypatch):
    dtype = torch.float64
    c = 2
    calls = {"batched": 0}
    import chorus.models.dispersion as dispersion_mod

    real_build = dispersion_mod.build_periodic_dipole_pme_kernel_batched

    def counted_build(*args, **kwargs):
        calls["batched"] += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(dispersion_mod, "build_periodic_dipole_pme_kernel_batched", counted_build)
    torch.manual_seed(24)
    model = LongRangeDispersion(
        feature_dim=c,
        mode="mbd-slq",
        cutoff=0.0,
        slq_num_probes=1,
        slq_lanczos_steps=4,
        mbd_operator_backend="pme_fft",
        mbd_pme_mesh_size=8,
    ).to(dtype)
    model.term.probe_mode = "basis"

    cell = torch.stack([torch.eye(3, dtype=dtype) * 8.0, torch.eye(3, dtype=dtype) * 9.0], dim=0)
    pos = torch.tensor(
        [[0.7, 0.4, 0.3], [2.2, 1.1, 0.8], [0.6, 0.7, 0.8], [2.1, 1.5, 1.2]],
        dtype=dtype,
        requires_grad=True,
    )
    feats = torch.randn(pos.size(0), c, dtype=dtype)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    empty = torch.zeros(0, dtype=torch.long)
    empty_vec = torch.zeros(0, 3, dtype=dtype)

    out = model(
        feats,
        pos,
        batch,
        cell,
        edge_src=empty,
        edge_dst=empty,
        edge_lengths=torch.zeros(0, dtype=dtype),
        edge_vec=empty_vec,
    )
    assert torch.isfinite(out).all()
    assert calls["batched"] == 1


def test_mbd_operator_backend_metadata_contract():
    wrapped = LongRangeDispersion(
        feature_dim=4,
        mode="mbd-slq",
        mbd_operator_backend="edge_sparse",
    )
    assert wrapped.mbd_operator_backend == "edge_sparse"
    assert isinstance(wrapped.term, ManyBodyDispersionSLQ)
    assert wrapped.term.operator_backend == "edge_sparse"

    resolved = resolve_model_architecture({"model_hyperparameters": {}}, overrides={})
    assert resolved["mbd_operator_backend"] == "edge_sparse"
    assert resolved["dispersion_training_graph_rule"] == "none"
    assert resolved["dispersion_deployment_graph_rule"] == "none"
    assert resolved["dispersion_train_deploy_graph_compatibility"] == "none"
    resolved = resolve_model_architecture(
        {
            "model_hyperparameters": {
                "long_range_dispersion_mode": "mbd-slq",
                "mbd_operator_backend": "edge_sparse",
            }
        },
        overrides={},
    )
    assert resolved["mbd_operator_backend"] == "edge_sparse"
    assert resolved["dispersion_training_graph_rule"] == "explicit_or_built_canonical_cutoff_edge_sparse"
    assert resolved["dispersion_deployment_graph_rule"] == "explicit_canonical_single_image_edge_sparse"
    assert resolved["dispersion_train_deploy_graph_compatibility"] == "conditional_on_single_image_cutoff"
    resolved = resolve_model_architecture(
        {
            "model_hyperparameters": {
                "long_range_dispersion_mode": "mbd-slq",
                "mbd_operator_backend": "edge_sparse",
                "dispersion_training_graph_rule": "explicit_or_built_canonical_cutoff_edge_sparse",
                "dispersion_deployment_graph_rule": "explicit_canonical_single_image_edge_sparse",
                "dispersion_train_deploy_graph_compatibility": "conditional_on_single_image_cutoff",
            }
        },
        overrides={},
    )
    assert resolved["dispersion_training_graph_rule"] == "explicit_or_built_canonical_cutoff_edge_sparse"
    assert resolved["dispersion_deployment_graph_rule"] == "explicit_canonical_single_image_edge_sparse"
    assert resolved["dispersion_train_deploy_graph_compatibility"] == "conditional_on_single_image_cutoff"
    with pytest.raises(ValueError, match="dispersion_training_graph_rule"):
        resolve_model_architecture(
            {
                "model_hyperparameters": {
                    "long_range_dispersion_mode": "mbd-slq",
                    "mbd_operator_backend": "edge_sparse",
                    "dispersion_training_graph_rule": "pme_fft_matvec_no_cutoff_edges",
                }
            },
            overrides={},
        )
    with pytest.raises(ValueError, match="dispersion_deployment_graph_rule"):
        resolve_model_architecture(
            {
                "model_hyperparameters": {
                    "long_range_dispersion_mode": "mbd-slq",
                    "mbd_operator_backend": "edge_sparse",
                    "dispersion_deployment_graph_rule": "main_neighbor_graph",
                }
            },
            overrides={},
        )
    with pytest.raises(ValueError, match="dispersion_train_deploy_graph_compatibility"):
        resolve_model_architecture(
            {
                "model_hyperparameters": {
                    "long_range_dispersion_mode": "mbd-slq",
                    "mbd_operator_backend": "edge_sparse",
                    "dispersion_train_deploy_graph_compatibility": "shared_main_neighbor_graph",
                }
            },
            overrides={},
        )
    resolved = resolve_model_architecture(
        {
            "model_hyperparameters": {
                "dispersion_neighbor_method": "cell",
                "dispersion_bruteforce_threshold": 123,
                "dispersion_allow_large_bruteforce_fallback": True,
            }
        },
        overrides={},
    )
    assert resolved["dispersion_neighbor_method"] == "cell"
    assert resolved["dispersion_bruteforce_threshold"] == 123
    assert resolved["dispersion_allow_large_bruteforce_fallback"] is True
    resolved = resolve_model_architecture(
        {"model_hyperparameters": {"dispersion_max_num_neighbors": 0}},
        overrides={},
    )
    assert resolved["dispersion_max_num_neighbors"] is None
    with pytest.raises(ValueError, match="dispersion_max_num_neighbors must be >= 0"):
        resolve_model_architecture(
            {"model_hyperparameters": {"dispersion_max_num_neighbors": -1}},
            overrides={},
        )
    with pytest.raises(ValueError, match="Unsupported dispersion_neighbor_method"):
        resolve_model_architecture(
            {"model_hyperparameters": {"dispersion_neighbor_method": "not-a-method"}},
            overrides={},
        )
    with pytest.raises(ValueError, match="Unsupported dispersion neighbor-list method"):
        LongRangeDispersion(feature_dim=4, mode="mbd-slq", neighbor_method="not-a-method")
    with pytest.raises(ValueError, match="bruteforce_threshold must be >= 0"):
        LongRangeDispersion(feature_dim=4, mode="mbd-slq", bruteforce_threshold=-1)
    assert LongRangeDispersion(feature_dim=4, mode="mbd-slq", max_num_neighbors=0).max_num_neighbors is None
    with pytest.raises(ValueError, match="max_num_neighbors must be >= 0"):
        LongRangeDispersion(feature_dim=4, mode="mbd-slq", max_num_neighbors=-1)

    assert (
        validate_dispersion_training_graph_rule(
            long_range_dispersion_mode="mbd-slq",
            mbd_operator_backend="edge_sparse",
            raw_rule="explicit_or_built_canonical_cutoff_edge_sparse",
        )
        == "explicit_or_built_canonical_cutoff_edge_sparse"
    )
    assert (
        validate_dispersion_deployment_graph_rule(
            long_range_dispersion_mode="mbd-slq",
            mbd_operator_backend="edge_sparse",
            raw_rule="explicit_canonical_single_image_edge_sparse",
        )
        == "explicit_canonical_single_image_edge_sparse"
    )
    assert (
        validate_dispersion_deployment_graph_rule(
            long_range_dispersion_mode="mbd-slq",
            mbd_operator_backend="pme_fft",
            raw_rule=None,
        )
        == "pme_fft_matvec_prototype"
    )
    assert (
        validate_dispersion_train_deploy_graph_compatibility(
            long_range_dispersion_mode="mbd-slq",
            mbd_operator_backend="edge_sparse",
            raw_value="conditional_on_single_image_cutoff",
        )
        == "conditional_on_single_image_cutoff"
    )
    with pytest.raises(ValueError, match="checkpoint dispersion_deployment_graph_rule"):
        validate_dispersion_deployment_graph_rule(
            long_range_dispersion_mode="mbd-slq",
            mbd_operator_backend="edge_sparse",
            raw_rule="main_neighbor_graph",
            source_label="checkpoint dispersion_deployment_graph_rule",
        )
    with pytest.raises(ValueError, match="checkpoint dispersion_train_deploy_graph_compatibility"):
        validate_dispersion_train_deploy_graph_compatibility(
            long_range_dispersion_mode="mbd-slq",
            mbd_operator_backend="edge_sparse",
            raw_value="shared_main_neighbor_graph",
            source_label="checkpoint dispersion_train_deploy_graph_compatibility",
        )

    pme = LongRangeDispersion(
        feature_dim=4,
        mode="mbd-slq",
        neighbor_method="cell",
        bruteforce_threshold=456,
        allow_large_bruteforce_fallback=True,
        mbd_operator_backend="pme_fft",
        mbd_pme_mesh_size=10,
        mbd_pme_assignment="pcs",
        mbd_pme_k_norm_floor=2.0e-6,
        mbd_pme_assignment_window_floor=3.0e-6,
        mbd_pme_ewald_alpha_prefactor=6.5,
    )
    assert pme.mbd_operator_backend == "pme_fft"
    assert pme.neighbor_method == "cell"
    assert pme.bruteforce_threshold == 456
    assert pme.allow_large_bruteforce_fallback is True
    assert pme.mbd_pme_mesh_size == 10
    assert pme.mbd_pme_assignment == "pcs"
    assert isinstance(pme.term, ManyBodyDispersionSLQ)
    assert pme.term.operator_backend == "pme_fft"
    assert pme.term.trace_estimator == "hutchinson"
    assert pme.term.pme_mesh_size == 10
    assert pme.term.pme_assignment == "pcs"
    assert pme.term.pme_k_norm_floor == 2.0e-6
    assert pme.term.pme_assignment_window_floor == 3.0e-6
    assert pme.term.pme_ewald_alpha_prefactor == 6.5

    resolved = resolve_model_architecture(
        {
            "model_hyperparameters": {
                "long_range_dispersion_mode": "mbd-slq",
                "mbd_operator_backend": "pme_fft",
                "mbd_pme_mesh_size": 12,
                "mbd_pme_assignment": "pcs",
                "mbd_pme_k_norm_floor": 4.0e-6,
                "mbd_pme_assignment_window_floor": 5.0e-6,
                "mbd_pme_ewald_alpha_prefactor": 7.0,
            }
        },
        overrides={},
    )
    assert resolved["mbd_operator_backend"] == "pme_fft"
    assert resolved["dispersion_training_graph_rule"] == "pme_fft_matvec_no_cutoff_edges"
    assert resolved["dispersion_deployment_graph_rule"] == "pme_fft_matvec_prototype"
    assert (
        resolved["dispersion_train_deploy_graph_compatibility"]
        == "matched_pme_fft_matvec"
    )
    assert resolved["mbd_pme_mesh_size"] == 12
    assert resolved["mbd_pme_assignment"] == "pcs"
    assert resolved["mbd_pme_k_norm_floor"] == 4.0e-6
    assert resolved["mbd_pme_assignment_window_floor"] == 5.0e-6
    assert resolved["mbd_pme_ewald_alpha_prefactor"] == 7.0


def test_aoti_pure_mbd_slq_metadata_keeps_dispersion_without_reciprocal_source():
    from chorus.cli.export_aoti_core import _long_range_deploy_metadata

    model = SimpleNamespace(
        long_range_dispersion=True,
        long_range_dispersion_mode="mbd-slq",
        dispersion_cutoff=9.5,
        dispersion_max_num_neighbors=None,
        dispersion_neighbor_method="auto",
        dispersion_bruteforce_threshold=384,
        dispersion_allow_large_bruteforce_fallback=False,
        dispersion_slq_num_probes=5,
        dispersion_slq_lanczos_steps=11,
        mbd_operator_backend="edge_sparse",
        mbd_pme_mesh_size=24,
        mbd_pme_assignment="pcs",
        mbd_pme_k_norm_floor=2.0e-6,
        mbd_pme_assignment_window_floor=3.0e-6,
        mbd_pme_ewald_alpha_prefactor=6.0,
    )

    meta = _long_range_deploy_metadata(
        model,
        export_reciprocal_source=False,
        use_explicit_dispersion_edges=True,
    )

    assert meta["export_reciprocal_source"] is False
    assert meta["aoti_dispersion_edges"] is True
    assert meta["long_range_dispersion_mode"] == "mbd-slq"
    assert meta["dispersion_cutoff"] == 9.5
    assert meta["dispersion_neighbor_method"] == "auto"
    assert meta["dispersion_bruteforce_threshold"] == 384
    assert meta["dispersion_training_graph_rule"] == "explicit_or_built_canonical_cutoff_edge_sparse"
    assert meta["dispersion_deployment_graph_rule"] == "explicit_canonical_single_image_edge_sparse"
    assert meta["dispersion_train_deploy_graph_compatibility"] == "conditional_on_single_image_cutoff"
    assert meta["mbd_operator_backend"] == "edge_sparse"
    assert meta["mbd_pme_mesh_size"] == 24
    assert meta["mbd_pme_assignment"] == "pcs"
    assert meta["long_range_mbd_energy_scale"] == 1.0


def test_aoti_mbd_energy_scale_metadata_uses_live_term_value():
    from chorus.cli.export_aoti_core import _long_range_deploy_metadata

    disp = LongRangeDispersion(
        feature_dim=3,
        mode="mbd-slq",
        cutoff=5.0,
        mbd_operator_backend="edge_sparse",
    ).double()
    with torch.no_grad():
        disp.term.energy_scale.fill_(0.056)
    model = SimpleNamespace(
        long_range_dispersion=True,
        long_range_dispersion_mode="mbd-slq",
        long_range_mbd_source_enabled=True,
        long_range_mbd_source_offset=0,
        long_range_mbd_source_channels=2,
        dispersion=disp,
        dispersion_cutoff=5.0,
        mbd_operator_backend="edge_sparse",
    )

    meta = _long_range_deploy_metadata(
        model,
        export_reciprocal_source=True,
        use_explicit_dispersion_edges=True,
    )

    assert meta["long_range_mbd_energy_scale"] == pytest.approx(0.056)
    assert meta["long_range_mbd_beta"] == pytest.approx(disp.mbd_beta())
    assert meta["long_range_mbd_coupling_scale"] == pytest.approx(disp.mbd_coupling_scale())


def test_pure_cartesian_ictd_fix_dispersion_metadata_contract():
    from chorus.models.pure_cartesian_ictd_fix import PureCartesianICTDFix

    core = PureCartesianICTDFix(
        max_embed_radius=3.0,
        main_max_radius=3.0,
        main_number_of_basis=4,
        hidden_dim_conv=4,
        hidden_dim_sh=4,
        hidden_dim=4,
        channel_in2=4,
        embedding_dim=4,
        max_atomvalue=10,
        atomic_numbers=[1, 6],
        num_interaction=2,
        function_type_main="bessel",
        lmax=1,
        ictd_fix_edge_lmax=1,
        ictd_fix_route="baseline",
        ictd_fix_product_backend="ictd-bridge-u",
        save_contraction_order=2,
        avg_num_neighbors=4.0,
        angular_basis="ictd",
        internal_compute_dtype=torch.float64,
        long_range_dispersion_mode="mbd-slq",
        dispersion_max_num_neighbors=0,
        dispersion_neighbor_method="cell",
        dispersion_bruteforce_threshold=222,
        dispersion_allow_large_bruteforce_fallback=True,
        mbd_operator_backend="pme_fft",
        mbd_pme_mesh_size=14,
        mbd_pme_assignment="pcs",
        mbd_pme_k_norm_floor=8.0e-6,
        mbd_pme_assignment_window_floor=9.0e-6,
        mbd_pme_ewald_alpha_prefactor=8.5,
    )
    assert core.dispersion_max_num_neighbors is None
    assert core.dispersion_neighbor_method == "cell"
    assert core.dispersion_bruteforce_threshold == 222
    assert core.dispersion_allow_large_bruteforce_fallback is True
    assert core.mbd_pme_mesh_size == 14
    assert core.mbd_pme_assignment == "pcs"
    assert core.dispersion is not None
    assert core.dispersion.neighbor_method == "cell"
    assert core.dispersion.bruteforce_threshold == 222
    assert core.dispersion.allow_large_bruteforce_fallback is True
    assert isinstance(core.dispersion.term, ManyBodyDispersionSLQ)
    assert core.dispersion.term.operator_backend == "pme_fft"
    assert core.dispersion.term.pme_mesh_size == 14
    assert core.dispersion.term.pme_assignment == "pcs"
    assert core.dispersion.term.pme_k_norm_floor == 8.0e-6
    assert core.dispersion.term.pme_assignment_window_floor == 9.0e-6
    assert core.dispersion.term.pme_ewald_alpha_prefactor == 8.5


def test_dispersion_mode_edge_policy_contract():
    assert dispersion_deployment_graph_rule("none") == "none"
    assert dispersion_training_graph_rule("none") == "none"
    assert dispersion_train_deploy_graph_compatibility("none") == "none"
    assert dispersion_deployment_graph_rule("pairwise-c6") == "main_neighbor_graph"
    assert dispersion_training_graph_rule("pairwise-c6") == "directed_cutoff_or_main_neighbor_graph"
    assert dispersion_train_deploy_graph_compatibility("pairwise-c6") == "shared_main_neighbor_graph"
    assert dispersion_mode_uses_cutoff_edges("pairwise-c6")
    assert not dispersion_mode_uses_canonical_edges("pairwise-c6")
    assert not dispersion_mode_needs_deployment_edges("pairwise-c6")

    for mode in ("mbd", "mbd-slq"):
        assert dispersion_mode_uses_cutoff_edges(mode, mbd_operator_backend="edge_sparse")
        assert dispersion_mode_uses_canonical_edges(mode)
        assert dispersion_mode_needs_deployment_edges(mode, mbd_operator_backend="edge_sparse")
        assert (
            dispersion_deployment_graph_rule(mode, mbd_operator_backend="edge_sparse")
            == "explicit_canonical_single_image_edge_sparse"
        )
        assert (
            dispersion_training_graph_rule(mode, mbd_operator_backend="edge_sparse")
            == "explicit_or_built_canonical_cutoff_edge_sparse"
        )
        assert (
            dispersion_train_deploy_graph_compatibility(mode, mbd_operator_backend="edge_sparse")
            == "conditional_on_single_image_cutoff"
        )

    assert not dispersion_mode_uses_cutoff_edges("mbd-slq", mbd_operator_backend="pme_fft")
    assert dispersion_mode_uses_canonical_edges("mbd-slq")
    assert not dispersion_mode_needs_deployment_edges("mbd-slq", mbd_operator_backend="pme_fft")
    assert (
        dispersion_deployment_graph_rule("mbd-slq", mbd_operator_backend="pme_fft")
        == "pme_fft_matvec_prototype"
    )
    assert (
        dispersion_training_graph_rule("mbd-slq", mbd_operator_backend="pme_fft")
        == "pme_fft_matvec_no_cutoff_edges"
    )
    assert (
        dispersion_train_deploy_graph_compatibility("mbd-slq", mbd_operator_backend="pme_fft")
        == "matched_pme_fft_matvec"
    )


def test_model_accepts_explicit_dispersion_neighbor_list():
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0, dispersion=True).double().eval()
    model.dispersion_cutoff = 5.0
    if model.dispersion is not None:
        model.dispersion.cutoff = 5.0

    torch.manual_seed(5)
    L = 12.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6], dtype=torch.long)
    pos = torch.rand(A.numel(), 3, dtype=torch.float64) * L
    batch = torch.zeros(A.numel(), dtype=torch.long)
    main_src, main_dst, main_shift = _neighbor_list(pos, cell[0], r_max=3.0)
    disp_src, disp_dst, disp_shift = _neighbor_list(pos, cell[0], r_max=5.0)

    with torch.no_grad():
        e_internal = model(pos, A, batch, main_src, main_dst, main_shift, cell)
        e_explicit = model(
            pos,
            A,
            batch,
            main_src,
            main_dst,
            main_shift,
            cell,
            dispersion_edge_src=disp_src,
            dispersion_edge_dst=disp_dst,
            dispersion_edge_shifts=disp_shift,
        )
    assert torch.allclose(e_internal, e_explicit, atol=1e-10, rtol=1e-10), (
        "explicit dispersion neighbor list changed the cutoff-based dispersion result"
    )


def test_model_mbd_dispersion_smoke():
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)

    torch.manual_seed(6)
    L = 10.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1], dtype=torch.long)
    pos = (torch.rand(A.numel(), 3, dtype=torch.float64) * L).requires_grad_(True)
    batch = torch.zeros(A.numel(), dtype=torch.long)
    main_src, main_dst, main_shift = _neighbor_list(pos.detach(), cell[0], r_max=4.0)
    disp_src, disp_dst, disp_shift = _neighbor_list(pos.detach(), cell[0], r_max=6.0)

    e = model(
        pos,
        A,
        batch,
        main_src,
        main_dst,
        main_shift,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shift,
    ).sum()
    assert torch.isfinite(e), "model+MBD energy is not finite"
    (force,) = torch.autograd.grad(e, pos, create_graph=True)
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "model+MBD force gradient did not flow"
    force.pow(2).mean().backward()
    grads = [p.grad for p in model.dispersion.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "MBD dispersion parameters got no gradient"


def test_model_mbd_slq_dispersion_smoke():
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)
    model.dispersion.term.num_probes = 4
    model.dispersion.term.lanczos_steps = 8

    torch.manual_seed(18)
    L = 10.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6], dtype=torch.long)
    pos = (torch.rand(A.numel(), 3, dtype=torch.float64) * L).requires_grad_(True)
    batch = torch.zeros(A.numel(), dtype=torch.long)
    main_src, main_dst, main_shift = _neighbor_list(pos.detach(), cell[0], r_max=4.0)
    disp_src, disp_dst, disp_shift = _neighbor_list(pos.detach(), cell[0], r_max=6.0)

    e = model(
        pos,
        A,
        batch,
        main_src,
        main_dst,
        main_shift,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shift,
    ).sum()
    assert torch.isfinite(e), "model+SLQ-MBD energy is not finite"
    (force,) = torch.autograd.grad(e, pos, create_graph=True)
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "model+SLQ-MBD force gradient did not flow"
    force.pow(2).mean().backward()
    grads = [p.grad for p in model.dispersion.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "SLQ-MBD dispersion parameters got no gradient"


def test_model_mbd_slq_dispersion_batched_variable_n_smoke():
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)
    model.dispersion.term.num_probes = 3
    model.dispersion.term.lanczos_steps = 6

    torch.manual_seed(19)
    L = 10.0
    elems = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    pos_parts, atom_parts, batch_parts = [], [], []
    main_src_parts, main_dst_parts, main_shift_parts = [], [], []
    disp_src_parts, disp_dst_parts, disp_shift_parts = [], [], []
    cells = []
    offset = 0
    for graph_idx, n_atoms in enumerate((4, 7)):
        cell = torch.eye(3, dtype=torch.float64) * L
        pos_g = torch.rand(n_atoms, 3, dtype=torch.float64) * L
        A_g = elems[torch.arange(n_atoms) % elems.numel()]
        main_src, main_dst, main_shift = _neighbor_list(pos_g, cell, r_max=4.0)
        disp_src, disp_dst, disp_shift = _neighbor_list(pos_g, cell, r_max=6.0)

        pos_parts.append(pos_g)
        atom_parts.append(A_g)
        batch_parts.append(torch.full((n_atoms,), graph_idx, dtype=torch.long))
        main_src_parts.append(main_src + offset)
        main_dst_parts.append(main_dst + offset)
        main_shift_parts.append(main_shift)
        disp_src_parts.append(disp_src + offset)
        disp_dst_parts.append(disp_dst + offset)
        disp_shift_parts.append(disp_shift)
        cells.append(cell)
        offset += n_atoms

    pos = torch.cat(pos_parts).requires_grad_(True)
    A = torch.cat(atom_parts)
    batch = torch.cat(batch_parts)
    cell = torch.stack(cells)
    main_src = torch.cat(main_src_parts)
    main_dst = torch.cat(main_dst_parts)
    main_shift = torch.cat(main_shift_parts)
    disp_src = torch.cat(disp_src_parts)
    disp_dst = torch.cat(disp_dst_parts)
    disp_shift = torch.cat(disp_shift_parts)

    e = model(
        pos,
        A,
        batch,
        main_src,
        main_dst,
        main_shift,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shift,
    ).sum()
    assert torch.isfinite(e), "batched variable-N SLQ-MBD energy is not finite"
    (force,) = torch.autograd.grad(e, pos, create_graph=True)
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "batched variable-N SLQ-MBD force did not flow"
    force.pow(2).mean().backward()
    grads = [p.grad for p in model.dispersion.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "batched variable-N SLQ-MBD got no gradient"


def test_mbd_torchscript_core_accepts_variable_atom_and_edge_counts():
    """The LibTorch deployment core must not bake the traced MBD matrix size."""
    torch.set_default_dtype(torch.float64)
    from chorus.interfaces.lammps_mliap import _TorchScriptEdgeVecCore

    model = _build_model(max_multipole_l=0, dispersion=True).double().eval()
    model.long_range_dispersion_mode = "mbd"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)

    def make_inputs(n: int):
        torch.manual_seed(n)
        box = 14.0
        cell = (torch.eye(3, dtype=torch.float64) * box).reshape(1, 3, 3)
        pos = torch.rand(n, 3, dtype=torch.float64) * box
        elements = torch.tensor([1, 6, 7, 8], dtype=torch.long)
        A = elements[torch.arange(n) % elements.numel()]
        batch = torch.zeros(n, dtype=torch.long)
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(j)
                    dst.append(i)
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        shifts = torch.zeros(src.numel(), 3, dtype=torch.float64)
        edge_vec = pos[dst] - pos[src]
        external = torch.empty(0, dtype=torch.float64)
        return (pos, A, batch, src, dst, shifts, cell, edge_vec, src, dst, shifts, edge_vec, external)

    core = _TorchScriptEdgeVecCore(model).eval()
    traced = torch.jit.trace(core, make_inputs(5), check_trace=False, strict=False)
    for n in (4, 7):
        out = traced(*make_inputs(n))
        assert isinstance(out, tuple) and len(out) == 6
        assert out[0].shape == (n, 1)
        assert torch.isfinite(out[0]).all(), f"non-finite traced MBD energy at N={n}"


def test_mbd_slq_torchscript_core_accepts_variable_atom_and_edge_counts():
    """The matrix-free MBD core must keep the deployment ABI and avoid fixed edge counts."""
    torch.set_default_dtype(torch.float64)
    from chorus.interfaces.lammps_mliap import _TorchScriptEdgeVecCore

    model = _build_model(max_multipole_l=0, dispersion=True).double().eval()
    model.long_range_dispersion_mode = "mbd-slq"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)
    model.dispersion.term.num_probes = 3
    model.dispersion.term.lanczos_steps = 6

    def make_inputs(n: int):
        torch.manual_seed(n + 100)
        box = 14.0
        cell = (torch.eye(3, dtype=torch.float64) * box).reshape(1, 3, 3)
        pos = torch.rand(n, 3, dtype=torch.float64) * box
        elements = torch.tensor([1, 6, 7, 8], dtype=torch.long)
        A = elements[torch.arange(n) % elements.numel()]
        batch = torch.zeros(n, dtype=torch.long)
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(j)
                    dst.append(i)
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        shifts = torch.zeros(src.numel(), 3, dtype=torch.float64)
        edge_vec = pos[dst] - pos[src]
        external = torch.empty(0, dtype=torch.float64)
        return (pos, A, batch, src, dst, shifts, cell, edge_vec, src, dst, shifts, edge_vec, external)

    core = _TorchScriptEdgeVecCore(model).eval()
    traced = torch.jit.trace(core, make_inputs(6), check_trace=False, strict=False)
    for n in (4, 8):
        out = traced(*make_inputs(n))
        assert isinstance(out, tuple) and len(out) == 6
        assert out[0].shape == (n, 1)
        assert torch.isfinite(out[0]).all(), f"non-finite traced SLQ-MBD energy at N={n}"


def test_model_complete_long_range_smoke():
    """Model with BOTH multipole electrostatics AND C6 dispersion (the complete long-range):
    runs, finite energy + forces, rotation-invariant total energy, and both new heads train."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2, dispersion=True).double().train()
    for m in model.modules():
        if getattr(m, "energy_scale", None) is not None:
            with torch.no_grad():
                m.energy_scale.fill_(0.1)  # activate the reciprocal term (inits to 0)
        complete = getattr(m, "complete_training_ramp", None)
        if callable(complete):
            complete()

    torch.manual_seed(1)
    L = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8])
    n = A.numel()
    pos0 = torch.rand(n, 3, dtype=torch.float64) * L
    batch = torch.zeros(n, dtype=torch.long)
    es, ed, sh = _neighbor_list(pos0, cell[0], r_max=4.5)

    pos = pos0.clone().requires_grad_(True)
    e = model(pos, A, batch, es, ed, sh, cell).sum()
    assert torch.isfinite(e), "energy not finite"
    (force,) = torch.autograd.grad(e, pos, create_graph=True)
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "bad force"

    R = _random_rotation(torch.float64)
    e_r = model(pos0 @ R.T, A, batch, es, ed, sh, (cell[0] @ R.T).reshape(1, 3, 3)).sum()
    assert torch.allclose(e.detach(), e_r.detach(), atol=1e-6), (
        f"complete long-range not rotation-invariant: {(e - e_r).abs().item():.2e}"
    )

    (force ** 2).mean().backward()
    mp = [p.grad for p in model.multipole_readout.parameters() if p.grad is not None]
    dp = [p.grad for p in model.dispersion.parameters() if p.grad is not None]
    assert mp and any(g.abs().sum() > 0 for g in mp), "multipole readout got no gradient"
    assert dp and any(g.abs().sum() > 0 for g in dp), "dispersion got no gradient"


def test_dispersion_neighbor_list_matches_bruteforce():
    """The longer-cutoff dispersion neighbor list matches a brute-force periodic search
    (cutoff < box so a single image shell is complete), removing the short-range truncation."""
    from chorus.test.test_multipole_long_range import _neighbor_list

    dtype = torch.float64
    torch.manual_seed(2)
    box, cutoff, n = 12.0, 5.0, 6
    cell = torch.eye(3, dtype=dtype) * box
    pos = torch.rand(n, 3, dtype=dtype) * box
    batch = torch.zeros(n, dtype=torch.long)

    src, dst, shifts = dispersion_neighbor_list(pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True)
    bsrc, bdst, bsh = _neighbor_list(pos, cell, cutoff)

    def keyset(s, d, sh):
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s, d, sh)}

    assert keyset(src, dst, shifts) == keyset(bsrc, bdst, bsh), "dispersion list != brute force"
    dlen = (pos[dst] - pos[src] + shifts.to(dtype) @ cell).norm(dim=1)
    assert (dlen > 1e-8).all() and (dlen <= cutoff + 1e-9).all(), "pairs outside cutoff"


def test_normalize_dispersion_edges_canonicalizes_reverse_mbd_edges():
    src = torch.tensor([1, 0, 0, 2, 1], dtype=torch.long)
    dst = torch.tensor([0, 0, 0, 1, 2], dtype=torch.long)
    shifts = torch.tensor(
        [
            [0, 0, 0],   # reverse of pair (0, 1)
            [-1, 0, 0],  # reverse of positive self-image
            [0, 0, 0],   # nonphysical self-zero edge, dropped for MBD
            [0, -1, 0],  # reverse of pair (1, 2) with shifted image
            [0, 1, 0],   # duplicate of the previous edge after canonicalization
        ],
        dtype=torch.float64,
    )

    mbd_src, mbd_dst, mbd_shifts = normalize_dispersion_edges(
        src,
        dst,
        shifts,
        canonical_undirected=True,
    )
    assert mbd_src.tolist() == [0, 0, 1]
    assert mbd_dst.tolist() == [0, 1, 2]
    assert mbd_shifts.tolist() == [[1, 0, 0], [0, 0, 0], [0, 1, 0]]

    directed_src, directed_dst, directed_shifts = normalize_dispersion_edges(
        src,
        dst,
        shifts,
        canonical_undirected=False,
    )
    assert directed_src.tolist() == [0, 0, 1, 2, 1]
    assert directed_dst.tolist() == [0, 0, 0, 1, 2]
    assert directed_shifts.tolist() == [[-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, -1, 0], [0, 1, 0]]


def test_dispersion_neighbor_list_canonical_matches_directed_subset():
    dtype = torch.float64
    torch.manual_seed(22)
    box, cutoff, n = 14.0, 5.0, 16
    cell = torch.eye(3, dtype=dtype) * box
    pos = torch.rand(n, 3, dtype=dtype) * box
    batch = torch.zeros(n, dtype=torch.long)

    directed = dispersion_neighbor_list(
        pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True, method="cell"
    )
    brute_directed = dispersion_neighbor_list(
        pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True, method="bruteforce"
    )
    canonical = dispersion_neighbor_list(
        pos,
        batch,
        cell.reshape(1, 3, 3),
        cutoff,
        pbc=True,
        canonical_undirected=True,
        method="cell",
    )
    brute_canonical = dispersion_neighbor_list(
        pos,
        batch,
        cell.reshape(1, 3, 3),
        cutoff,
        pbc=True,
        canonical_undirected=True,
        method="bruteforce",
    )

    def keyset(edges):
        s, d, sh = edges
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s, d, sh)}

    assert keyset(directed) == keyset(brute_directed)
    assert keyset(canonical) == keyset(brute_canonical)
    assert keyset(canonical) == {(s, d, sh) for (s, d, sh) in keyset(directed) if s < d}
    assert bool((canonical[0] < canonical[1]).all())


def test_dispersion_neighbor_list_outputs_deterministic_sorted_edges():
    dtype = torch.float64
    box, cutoff = 4.0, 5.5
    cell = (torch.eye(3, dtype=dtype) * box).reshape(1, 3, 3)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]], dtype=dtype)
    batch = torch.zeros(pos.shape[0], dtype=torch.long)

    brute = dispersion_neighbor_list(
        pos, batch, cell, cutoff, pbc=True, canonical_undirected=True, method="bruteforce"
    )
    cell_edges = dispersion_neighbor_list(
        pos, batch, cell, cutoff, pbc=True, canonical_undirected=True, method="cell",
        allow_large_bruteforce_fallback=True,
    )
    def rows(edges):
        s, d, sh = edges
        return [(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s, d, sh)]

    assert rows(brute) == sorted(rows(brute), key=lambda r: (r[1], r[0], r[2]))
    assert rows(cell_edges) == rows(brute)


def test_dispersion_neighbor_cap_is_density_based_not_system_size():
    dtype = torch.float64
    cutoff = 5.0
    density = 0.02
    caps = []
    for n in (2048, 16384):
        box = (n / density) ** (1.0 / 3.0)
        cell = torch.eye(3, dtype=dtype) * box
        caps.append(_estimate_dispersion_max_neighbors(n, cell, cutoff, pbc=True))
    assert caps[0] == caps[1]
    assert 64 <= caps[0] < 512


def test_dispersion_neighbor_list_rejects_invalid_strategy_limits():
    dtype = torch.float64
    cell = (torch.eye(3, dtype=dtype) * 8.0).reshape(1, 3, 3)
    pos = torch.rand(4, 3, dtype=dtype)
    batch = torch.zeros(pos.size(0), dtype=torch.long)

    with pytest.raises(ValueError, match="Unsupported dispersion neighbor-list method"):
        dispersion_neighbor_list(pos, batch, cell, 4.0, method="not-a-method")
    with pytest.raises(ValueError, match="bruteforce_threshold must be >= 0"):
        dispersion_neighbor_list(pos, batch, cell, 4.0, bruteforce_threshold=-1)
    dispersion_neighbor_list(pos, batch, cell, 4.0, max_num_neighbors=0)
    with pytest.raises(ValueError, match="max_num_neighbors must be >= 0"):
        dispersion_neighbor_list(pos, batch, cell, 4.0, max_num_neighbors=-1)


def test_dispersion_cutoff_single_image_exact_matches_deployment_guard():
    dtype = torch.float64
    orthogonal = (torch.eye(3, dtype=dtype) * 10.0).reshape(1, 3, 3)
    assert dispersion_cutoff_is_single_image_exact(orthogonal, 5.0, pbc=True)
    assert not dispersion_cutoff_is_single_image_exact(orthogonal, 5.0 + 1.0e-6, pbc=True)
    assert dispersion_cutoff_is_single_image_exact(orthogonal, 12.0, pbc=False)

    skewed = torch.tensor(
        [[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=dtype,
    )
    assert not dispersion_cutoff_is_single_image_exact(skewed, 2.5, pbc=True)
    assert dispersion_cutoff_is_single_image_exact(skewed, 0.5, pbc=True)


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or importlib.util.find_spec("torch_cluster") is None,
    reason="adaptive cap retry needs CUDA torch_cluster",
)
def test_dispersion_neighbor_list_adaptive_low_cap_matches_bruteforce():
    torch.manual_seed(34)
    n, cutoff, density = 256, 5.0, 0.10
    box = (n / density) ** (1.0 / 3.0)
    cell = torch.eye(3, device="cuda") * box
    pos = torch.rand(n, 3, device="cuda") * box
    batch = torch.zeros(n, device="cuda", dtype=torch.long)

    brute = dispersion_neighbor_list(
        pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True,
        canonical_undirected=True, method="bruteforce",
    )
    adaptive = dispersion_neighbor_list(
        pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True,
        canonical_undirected=True, method="auto", bruteforce_threshold=0,
        max_num_neighbors=4,
    )

    def keyset(edges):
        s, d, sh = edges
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s.cpu(), d.cpu(), sh.cpu())}

    assert keyset(adaptive) == keyset(brute)


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or importlib.util.find_spec("torch_cluster") is None,
    reason="multi-image PBC radius search needs CUDA torch_cluster",
)
def test_dispersion_neighbor_list_multi_image_auto_matches_bruteforce():
    torch.manual_seed(35)
    n, box, cutoff = 32, 4.0, 5.5
    cell = torch.eye(3, device="cuda") * box
    pos = torch.rand(n, 3, device="cuda") * box
    batch = torch.zeros(n, device="cuda", dtype=torch.long)

    brute = dispersion_neighbor_list(
        pos,
        batch,
        cell.reshape(1, 3, 3),
        cutoff,
        pbc=True,
        canonical_undirected=True,
        method="bruteforce",
    )
    auto = dispersion_neighbor_list(
        pos,
        batch,
        cell.reshape(1, 3, 3),
        cutoff,
        pbc=True,
        canonical_undirected=True,
        method="auto",
        bruteforce_threshold=0,
    )

    def keyset(edges):
        s, d, sh = edges
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s.cpu(), d.cpu(), sh.cpu())}

    assert keyset(auto) == keyset(brute)
    assert list(zip(auto[0].cpu().tolist(), auto[1].cpu().tolist(), auto[2].cpu().tolist())) == list(
        zip(brute[0].cpu().tolist(), brute[1].cpu().tolist(), brute[2].cpu().tolist())
    )
    assert any(any(abs(v) > 1 for v in shift) for _, _, shift in keyset(auto))


def test_dispersion_neighbor_list_canonical_keeps_self_image_half():
    dtype = torch.float64
    box, cutoff = 4.0, 5.5
    cell = (torch.eye(3, dtype=dtype) * box).reshape(1, 3, 3)
    pos = torch.zeros(1, 3, dtype=dtype)
    batch = torch.zeros(1, dtype=torch.long)

    directed = dispersion_neighbor_list(
        pos, batch, cell, cutoff, pbc=True, canonical_undirected=False, method="bruteforce"
    )
    canonical = dispersion_neighbor_list(
        pos, batch, cell, cutoff, pbc=True, canonical_undirected=True, method="bruteforce"
    )
    cell_list = dispersion_neighbor_list(
        pos, batch, cell, cutoff, pbc=True, canonical_undirected=True, method="cell",
        allow_large_bruteforce_fallback=True,
    )

    def keyset(edges):
        s, d, sh = edges
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s, d, sh)}

    assert keyset(directed) == {
        (0, 0, (-1, 0, 0)),
        (0, 0, (1, 0, 0)),
        (0, 0, (0, -1, 0)),
        (0, 0, (0, 1, 0)),
        (0, 0, (0, 0, -1)),
        (0, 0, (0, 0, 1)),
    }
    assert keyset(canonical) == {
        (0, 0, (1, 0, 0)),
        (0, 0, (0, 1, 0)),
        (0, 0, (0, 0, 1)),
    }
    assert keyset(cell_list) == keyset(canonical)

    mbd = ManyBodyDispersion(feature_dim=2).to(dtype)
    feats = torch.zeros(1, 2, dtype=dtype)
    edge_vec_directed = pos[directed[1]] - pos[directed[0]] + directed[2].to(dtype) @ cell[0]
    edge_vec_canonical = pos[canonical[1]] - pos[canonical[0]] + canonical[2].to(dtype) @ cell[0]
    e_directed = mbd(feats, batch, directed[0], directed[1], edge_vec_directed).sum()
    e_canonical = mbd(feats, batch, canonical[0], canonical[1], edge_vec_canonical).sum()
    assert torch.allclose(e_directed, e_canonical, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or importlib.util.find_spec("torch_cluster") is None,
    reason="single-atom self-image auto path needs CUDA torch_cluster",
)
def test_dispersion_neighbor_list_single_atom_auto_keeps_self_images():
    dtype = torch.float32
    box, cutoff = 4.0, 5.5
    cell = (torch.eye(3, device="cuda", dtype=dtype) * box).reshape(1, 3, 3)
    pos = torch.zeros(1, 3, device="cuda", dtype=dtype)
    batch = torch.zeros(1, device="cuda", dtype=torch.long)

    brute = dispersion_neighbor_list(
        pos, batch, cell, cutoff, pbc=True, canonical_undirected=True, method="bruteforce"
    )
    auto = dispersion_neighbor_list(
        pos, batch, cell, cutoff, pbc=True, canonical_undirected=True, method="auto", bruteforce_threshold=0
    )

    def keyset(edges):
        s, d, sh = edges
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s.cpu(), d.cpu(), sh.cpu())}

    assert keyset(auto) == keyset(brute)
    assert list(zip(auto[0].cpu().tolist(), auto[1].cpu().tolist(), auto[2].cpu().tolist())) == list(
        zip(brute[0].cpu().tolist(), brute[1].cpu().tolist(), brute[2].cpu().tolist())
    )
    assert keyset(auto) == {
        (0, 0, (1, 0, 0)),
        (0, 0, (0, 1, 0)),
        (0, 0, (0, 0, 1)),
    }


def test_dispersion_neighbor_list_triclinic_uses_face_height_image_bound():
    dtype = torch.float64
    cutoff = 2.5
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=dtype,
    )
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.05, 0.0]], dtype=dtype)
    batch = torch.zeros(pos.shape[0], dtype=torch.long)

    brute = dispersion_neighbor_list(
        pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True, method="bruteforce"
    )
    cell_list = dispersion_neighbor_list(
        pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True, method="cell",
        allow_large_bruteforce_fallback=True,
    )

    def keyset(edges):
        s, d, sh = edges
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s, d, sh)}

    assert keyset(cell_list) == keyset(brute)
    assert any(any(abs(v) > 1 for v in shift) for _, _, shift in keyset(brute))
    dlen = (pos[brute[1]] - pos[brute[0]] + brute[2].to(dtype) @ cell).norm(dim=1)
    assert (dlen > 1e-8).all() and (dlen <= cutoff + 1e-9).all()


def test_dispersion_neighbor_list_auto_accounts_for_image_count(monkeypatch):
    dtype = torch.float64
    cutoff = 2.5
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=dtype,
    )
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.05, 0.05, 0.0], [0.3, 0.1, 0.0], [0.4, 0.2, 0.0]],
        dtype=dtype,
    )
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    called = {"bruteforce": 0, "radius": 0}

    def fake_bruteforce(*args, **kwargs):
        called["bruteforce"] += 1
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, 3, dtype=torch.long)

    def fake_radius(*args, **kwargs):
        called["radius"] += 1
        return torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long), torch.zeros(1, 3, dtype=torch.long)

    monkeypatch.setattr("chorus.models.dispersion._dispersion_neighbor_list_bruteforce", fake_bruteforce)
    monkeypatch.setattr("chorus.models.dispersion._dispersion_neighbor_list_torch_cluster", fake_radius)

    src, dst, shifts, info = dispersion_neighbor_list(
        pos,
        batch,
        cell.reshape(1, 3, 3),
        cutoff,
        pbc=True,
        method="auto",
        bruteforce_threshold=8,
        return_info=True,
    )

    assert called == {"bruteforce": 0, "radius": 1}
    assert src.tolist() == [0]
    assert dst.tolist() == [1]
    assert shifts.shape == (1, 3)
    assert info["selected_method"] == "auto_torch_cluster"
    assert info["dense_work"] > info["dense_work_limit"]


def test_dispersion_neighbor_list_auto_falls_back_to_cell_without_torch_cluster(monkeypatch):
    dtype = torch.float64
    cutoff = 2.5
    pos = torch.rand(16, 3, dtype=dtype)
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    cell = (torch.eye(3, dtype=dtype) * 20.0).reshape(1, 3, 3)
    called = {"cell": 0, "bruteforce": 0}

    def fake_radius(*args, **kwargs):
        raise ImportError("torch_cluster unavailable")

    def fake_cell(*args, **kwargs):
        called["cell"] += 1
        assert kwargs["allow_bruteforce_fallback"] is False
        return torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long), torch.zeros(1, 3, dtype=torch.long)

    def fake_bruteforce(*args, **kwargs):
        called["bruteforce"] += 1
        raise AssertionError("auto should try cell-list before large brute-force fallback")

    monkeypatch.setattr("chorus.models.dispersion._dispersion_neighbor_list_torch_cluster", fake_radius)
    monkeypatch.setattr("chorus.models.dispersion._dispersion_neighbor_list_cell", fake_cell)
    monkeypatch.setattr("chorus.models.dispersion._dispersion_neighbor_list_bruteforce", fake_bruteforce)

    src, dst, shifts, info = dispersion_neighbor_list(
        pos,
        batch,
        cell,
        cutoff,
        pbc=True,
        method="auto",
        bruteforce_threshold=0,
        allow_large_bruteforce_fallback=False,
        return_info=True,
    )

    assert called == {"cell": 1, "bruteforce": 0}
    assert src.tolist() == [0]
    assert dst.tolist() == [1]
    assert shifts.shape == (1, 3)
    assert info["selected_method"] == "auto_cell"


def test_dispersion_neighbor_list_auto_refuses_large_multi_image_bruteforce_without_opt_in(monkeypatch):
    dtype = torch.float64
    cutoff = 5.5
    pos = torch.zeros(1, 3, dtype=dtype)
    batch = torch.zeros(1, dtype=torch.long)
    cell = (torch.eye(3, dtype=dtype) * 4.0).reshape(1, 3, 3)

    def fake_radius(*args, **kwargs):
        raise ImportError("torch_cluster unavailable")

    def fail_bruteforce(*args, **kwargs):
        raise AssertionError("large auto should not silently use exact O(N^2 * images) brute force")

    monkeypatch.setattr("chorus.models.dispersion._dispersion_neighbor_list_torch_cluster", fake_radius)
    monkeypatch.setattr("chorus.models.dispersion._dispersion_neighbor_list_bruteforce", fail_bruteforce)

    with pytest.raises(ImportError, match="large exact multi-image brute-force fallback") as exc_info:
        dispersion_neighbor_list(
            pos,
            batch,
            cell,
            cutoff,
            pbc=True,
            method="auto",
            bruteforce_threshold=0,
            allow_large_bruteforce_fallback=False,
        )
    message = str(exc_info.value)
    assert "Complexity context:" in message
    assert "max_graph_atoms=1" in message
    assert "bruteforce_threshold=0" in message
    assert "dense_work=" in message
    assert "dense_work_limit=0" in message


def test_dispersion_neighbor_list_cell_refuses_multi_image_bruteforce_without_opt_in(monkeypatch):
    dtype = torch.float64
    cutoff = 5.5
    pos = torch.zeros(1, 3, dtype=dtype)
    batch = torch.zeros(1, dtype=torch.long)
    cell = (torch.eye(3, dtype=dtype) * 4.0).reshape(1, 3, 3)

    def fail_bruteforce(*args, **kwargs):
        raise AssertionError("cell method should not silently use exact O(N^2 * images) brute force")

    monkeypatch.setattr("chorus.models.dispersion._dispersion_neighbor_list_bruteforce", fail_bruteforce)

    with pytest.raises(ImportError, match="exact multi-image brute-force fallback"):
        dispersion_neighbor_list(
            pos,
            batch,
            cell,
            cutoff,
            pbc=True,
            method="cell",
            allow_large_bruteforce_fallback=False,
        )


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or importlib.util.find_spec("torch_cluster") is None,
    reason="triclinic auto PBC radius search needs CUDA torch_cluster",
)
def test_dispersion_neighbor_list_triclinic_auto_matches_bruteforce():
    dtype = torch.float32
    cutoff = 2.5
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [0.0, 0.0, 10.0]],
        device="cuda",
        dtype=dtype,
    )
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.05, 0.0]], device="cuda", dtype=dtype)
    batch = torch.zeros(pos.shape[0], device="cuda", dtype=torch.long)

    brute = dispersion_neighbor_list(
        pos,
        batch,
        cell.reshape(1, 3, 3),
        cutoff,
        pbc=True,
        canonical_undirected=True,
        method="bruteforce",
    )
    auto = dispersion_neighbor_list(
        pos,
        batch,
        cell.reshape(1, 3, 3),
        cutoff,
        pbc=True,
        canonical_undirected=True,
        method="auto",
        bruteforce_threshold=0,
    )

    def keyset(edges):
        s, d, sh = edges
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s.cpu(), d.cpu(), sh.cpu())}

    assert keyset(auto) == keyset(brute)
    assert list(zip(auto[0].cpu().tolist(), auto[1].cpu().tolist(), auto[2].cpu().tolist())) == list(
        zip(brute[0].cpu().tolist(), brute[1].cpu().tolist(), brute[2].cpu().tolist())
    )
    assert any(any(abs(v) > 1 for v in shift) for _, _, shift in keyset(auto))


def test_force_trainer_mbd_slq_uses_canonical_dispersion_edges():
    from chorus.training.train_loop import ForceTrainer

    dtype = torch.float64
    torch.manual_seed(23)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        slq_num_probes=2,
        slq_lanczos_steps=4,
    ).double()
    model.dispersion_cutoff = 5.0
    model.dispersion_pbc = True

    cell = (torch.eye(3, dtype=dtype) * 12.0).reshape(1, 3, 3)
    pos = torch.rand(14, 3, dtype=dtype) * 12.0
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    trainer = ForceTrainer(model, train_loader=[], device="cpu", dtype=dtype, lr_scheduler="none")
    src, dst, shifts = trainer._dispersion_edges_for_batch(pos, batch, cell, extras={})
    assert src.numel() > 0
    assert bool((src < dst).all())
    assert trainer._last_dispersion_neighbor_build_info["selected_method"] == "auto_bruteforce"
    trainer._record_dispersion_graph_observation(batch, cell, (src, dst, shifts))
    summary = trainer._dispersion_graph_observation_summary()
    assert summary["builder_auto_bruteforce_batches"] == 1
    assert summary["builder_unknown_batches"] == 0
    assert summary["neighbor_builder_status"] == "observed_dense_builder"
    brute = dispersion_neighbor_list(
        pos,
        batch,
        cell,
        5.0,
        pbc=True,
        canonical_undirected=True,
        method="bruteforce",
    )

    def keyset(s, d, sh):
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s, d, sh)}

    assert keyset(src, dst, shifts) == keyset(*brute)


def test_force_trainer_forwards_dispersion_neighbor_cap(monkeypatch):
    from chorus.training.train_loop import ForceTrainer

    dtype = torch.float64
    captured = {}

    def fake_dispersion_neighbor_list(*args, **kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
        )

    monkeypatch.setattr("chorus.models.dispersion.dispersion_neighbor_list", fake_dispersion_neighbor_list)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        max_num_neighbors=11,
    ).double()
    model.dispersion_cutoff = 5.0
    model.dispersion_pbc = True
    model.dispersion_max_num_neighbors = 11

    cell = (torch.eye(3, dtype=dtype) * 12.0).reshape(1, 3, 3)
    pos = torch.rand(4, 3, dtype=dtype) * 12.0
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    trainer = ForceTrainer(model, train_loader=[], device="cpu", dtype=dtype, lr_scheduler="none")
    src, dst, shifts = trainer._dispersion_edges_for_batch(pos, batch, cell, extras={})

    assert src.tolist() == [0]
    assert dst.tolist() == [1]
    assert shifts.shape == (1, 3)
    assert captured["max_num_neighbors"] == 11
    assert captured["canonical_undirected"] is True


def test_force_trainer_records_dispersion_graph_deployability_observation():
    from chorus.training.train_loop import ForceTrainer

    dtype = torch.float64
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "edge_sparse"
    model.dispersion_cutoff = 5.0
    model.dispersion_pbc = True
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        slq_num_probes=2,
        slq_lanczos_steps=4,
        mbd_operator_backend="edge_sparse",
    ).double()

    trainer = ForceTrainer(model, train_loader=[], device="cpu", dtype=dtype, lr_scheduler="none")
    deployable_edges = (
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.zeros(1, 3, dtype=dtype),
    )
    trainer._record_dispersion_graph_observation(
        torch.zeros(2, dtype=torch.long),
        (torch.eye(3, dtype=dtype) * 12.0).reshape(1, 3, 3),
        deployable_edges,
    )
    summary = trainer._dispersion_graph_observation_summary()
    assert summary["status"] == "observed_single_image_deployable"
    assert summary["scope"] == "local"
    assert summary["observed_batches"] == 1
    assert summary["observed_edges"] == 1
    assert summary["single_image_cell_batches"] == 1
    assert summary["non_single_image_cell_batches"] == 0
    assert summary["self_image_edges"] == 0
    assert summary["multi_image_shift_edges"] == 0
    assert summary["builder_unknown_batches"] == 1
    assert summary["neighbor_builder_status"] == "observed_unknown_builder"

    model.dispersion_cutoff = 5.5
    non_deployable_edges = (
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([[1, 0, 0], [2, 0, 0]], dtype=dtype),
    )
    trainer._record_dispersion_graph_observation(
        torch.zeros(2, dtype=torch.long),
        (torch.eye(3, dtype=dtype) * 4.0).reshape(1, 3, 3),
        non_deployable_edges,
    )
    summary = trainer._dispersion_graph_observation_summary()
    assert summary["status"] == "observed_not_single_image_deployable"
    assert summary["scope"] == "local"
    assert summary["observed_batches"] == 2
    assert summary["observed_edges"] == 3
    assert summary["non_single_image_cell_batches"] == 1
    assert summary["self_image_edges"] == 1
    assert summary["multi_image_shift_edges"] == 1
    assert summary["builder_unknown_batches"] == 2
    assert summary["neighbor_builder_status"] == "observed_unknown_builder"
    assert trainer._training_metadata()["dispersion_training_graph_observation"] == summary


def test_force_trainer_syncs_dispersion_graph_observation_across_ranks(monkeypatch):
    from chorus.training.train_loop import ForceTrainer
    import chorus.training.train_loop as train_loop

    dtype = torch.float64
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "edge_sparse"
    model.dispersion_cutoff = 5.0
    model.dispersion_pbc = True
    trainer = ForceTrainer(model, train_loader=[], device="cpu", dtype=dtype, lr_scheduler="none")
    trainer._dispersion_graph_observation.update(
        {
            "observed_batches": 1,
            "observed_edges": 3,
            "single_image_cell_batches": 1,
            "non_single_image_cell_batches": 0,
            "self_image_edges": 0,
            "multi_image_shift_edges": 0,
            "builder_auto_torch_cluster_batches": 1,
        }
    )

    monkeypatch.setattr(trainer, "_dist_ready", lambda: True)

    class FakeReduceOp:
        SUM = object()

    def fake_all_reduce(tensor, op=None):
        del op
        remote = torch.zeros_like(tensor)
        remote[:6] = torch.tensor([2, 5, 0, 2, 1, 1], dtype=tensor.dtype, device=tensor.device)
        remote[9] = 2
        tensor.add_(remote)

    monkeypatch.setattr(train_loop.dist, "ReduceOp", FakeReduceOp)
    monkeypatch.setattr(train_loop.dist, "all_reduce", fake_all_reduce)

    trainer._sync_dispersion_graph_observation()
    summary = trainer._dispersion_graph_observation_summary()
    assert summary["scope"] == "distributed"
    assert summary["status"] == "observed_not_single_image_deployable"
    assert summary["observed_batches"] == 3
    assert summary["observed_edges"] == 8
    assert summary["single_image_cell_batches"] == 1
    assert summary["non_single_image_cell_batches"] == 2
    assert summary["self_image_edges"] == 1
    assert summary["multi_image_shift_edges"] == 1
    assert summary["builder_auto_cell_batches"] == 2
    assert summary["neighbor_builder_status"] == "observed_sparse_or_explicit_builder"

    trainer._record_dispersion_graph_observation(
        torch.zeros(2, dtype=torch.long),
        (torch.eye(3, dtype=dtype) * 12.0).reshape(1, 3, 3),
        (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.zeros(1, 3, dtype=dtype),
        ),
    )
    assert trainer._dispersion_graph_observation_summary()["scope"] == "local"


def test_force_trainer_restores_dispersion_graph_observation_as_resume_base(monkeypatch):
    from chorus.training.train_loop import ForceTrainer
    import chorus.training.train_loop as train_loop

    dtype = torch.float64
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "edge_sparse"
    model.dispersion_cutoff = 5.0
    model.dispersion_pbc = True
    trainer = ForceTrainer(model, train_loader=[], device="cpu", dtype=dtype, lr_scheduler="none")
    trainer._restore_dispersion_graph_observation(
        {
            "observed_batches": 4,
            "observed_edges": 10,
            "single_image_cell_batches": 3,
            "non_single_image_cell_batches": 1,
            "self_image_edges": 2,
            "multi_image_shift_edges": 1,
            "status": "observed_not_single_image_deployable",
            "scope": "distributed",
        }
    )
    summary = trainer._dispersion_graph_observation_summary()
    assert summary["scope"] == "local"
    assert summary["status"] == "observed_not_single_image_deployable"
    assert summary["observed_batches"] == 4
    assert summary["observed_edges"] == 10
    assert summary["neighbor_builder_status"] == "observed_unknown_builder"

    trainer._record_dispersion_graph_observation(
        torch.zeros(2, dtype=torch.long),
        (torch.eye(3, dtype=dtype) * 12.0).reshape(1, 3, 3),
        (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.zeros(1, 3, dtype=dtype),
        ),
    )
    summary = trainer._dispersion_graph_observation_summary()
    assert summary["observed_batches"] == 5
    assert summary["observed_edges"] == 11
    assert summary["single_image_cell_batches"] == 4
    assert summary["non_single_image_cell_batches"] == 1
    assert summary["self_image_edges"] == 2
    assert summary["multi_image_shift_edges"] == 1
    assert summary["neighbor_builder_status"] == "observed_unknown_builder"

    monkeypatch.setattr(trainer, "_dist_ready", lambda: True)

    class FakeReduceOp:
        SUM = object()

    def fake_all_reduce(tensor, op=None):
        del op
        remote_new = torch.zeros_like(tensor)
        remote_new[:6] = torch.tensor([2, 5, 0, 2, 1, 1], dtype=tensor.dtype, device=tensor.device)
        remote_new[9] = 2
        tensor.add_(remote_new)

    monkeypatch.setattr(train_loop.dist, "ReduceOp", FakeReduceOp)
    monkeypatch.setattr(train_loop.dist, "all_reduce", fake_all_reduce)
    trainer._sync_dispersion_graph_observation()
    summary = trainer._dispersion_graph_observation_summary()
    assert summary["scope"] == "distributed"
    assert summary["status"] == "observed_not_single_image_deployable"
    assert summary["observed_batches"] == 7
    assert summary["observed_edges"] == 16
    assert summary["single_image_cell_batches"] == 4
    assert summary["non_single_image_cell_batches"] == 3
    assert summary["self_image_edges"] == 3
    assert summary["multi_image_shift_edges"] == 2
    assert summary["builder_auto_cell_batches"] == 2
    assert summary["neighbor_builder_status"] == "observed_unknown_builder"


def test_force_trainer_normalizes_explicit_mbd_dispersion_edges():
    from chorus.training.train_loop import ForceTrainer

    dtype = torch.float64
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "edge_sparse"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        slq_num_probes=2,
        slq_lanczos_steps=4,
        mbd_operator_backend="edge_sparse",
    ).double()

    cell = (torch.eye(3, dtype=dtype) * 8.0).reshape(1, 3, 3)
    pos = torch.rand(2, 3, dtype=dtype)
    batch = torch.zeros(2, dtype=torch.long)
    extras = {
        "dispersion_edge_src": torch.tensor([1, 0, 0, 0], dtype=torch.long),
        "dispersion_edge_dst": torch.tensor([0, 0, 0, 0], dtype=torch.long),
        "dispersion_edge_shifts": torch.tensor(
            [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]],
            dtype=dtype,
        ),
    }
    trainer = ForceTrainer(model, train_loader=[], device="cpu", dtype=dtype, lr_scheduler="none")
    src, dst, shifts = trainer._dispersion_edges_for_batch(pos, batch, cell, extras=extras)

    assert src.tolist() == [0, 0]
    assert dst.tolist() == [0, 1]
    assert shifts.to(torch.long).tolist() == [[1, 0, 0], [0, 0, 0]]


def test_force_trainer_rejects_invalid_explicit_dispersion_edges():
    from chorus.training.train_loop import ForceTrainer

    dtype = torch.float64
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "edge_sparse"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        mbd_operator_backend="edge_sparse",
    ).double()

    cell = torch.stack([torch.eye(3, dtype=dtype) * 8.0, torch.eye(3, dtype=dtype) * 9.0], dim=0)
    pos = torch.rand(4, 3, dtype=dtype)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    trainer = ForceTrainer(model, train_loader=[], device="cpu", dtype=dtype, lr_scheduler="none")

    with pytest.raises(ValueError, match="outside the current batch"):
        trainer._dispersion_edges_for_batch(
            pos,
            batch,
            cell,
            extras={
                "dispersion_edge_src": torch.tensor([0], dtype=torch.long),
                "dispersion_edge_dst": torch.tensor([4], dtype=torch.long),
                "dispersion_edge_shifts": torch.zeros(1, 3, dtype=dtype),
            },
        )

    with pytest.raises(ValueError, match="different batch graphs"):
        trainer._dispersion_edges_for_batch(
            pos,
            batch,
            cell,
            extras={
                "dispersion_edge_src": torch.tensor([1], dtype=torch.long),
                "dispersion_edge_dst": torch.tensor([2], dtype=torch.long),
                "dispersion_edge_shifts": torch.zeros(1, 3, dtype=dtype),
            },
        )


def test_force_trainer_pme_fft_mbd_skips_dispersion_neighbor_list(monkeypatch):
    from chorus.training.train_loop import ForceTrainer

    def fail_neighbor_list(*args, **kwargs):
        raise AssertionError("pme_fft MBD should not build training dispersion edges")

    monkeypatch.setattr("chorus.models.dispersion.dispersion_neighbor_list", fail_neighbor_list)
    dtype = torch.float64
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "pme_fft"
    model.dispersion_cutoff = 5.0
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        mbd_operator_backend="pme_fft",
    ).double()

    cell = (torch.eye(3, dtype=dtype) * 8.0).reshape(1, 3, 3)
    pos = torch.rand(4, 3, dtype=dtype)
    batch = torch.zeros(4, dtype=torch.long)
    trainer = ForceTrainer(model, train_loader=[], device="cpu", dtype=dtype, lr_scheduler="none")
    assert trainer._dispersion_edges_for_batch(pos, batch, cell, extras={}) is None


def test_model_forward_explicit_mbd_edges_override_internal_dispersion_neighbor_list(monkeypatch):
    dtype = torch.float64
    torch.manual_seed(37)
    model = _build_model(max_multipole_l=0, dispersion=True).double().eval()
    model.long_range_module = None
    model.long_range_exports_reciprocal_source = False
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "edge_sparse"
    model.dispersion_cutoff = 5.0
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        slq_num_probes=2,
        slq_lanczos_steps=4,
        mbd_operator_backend="edge_sparse",
    ).double()

    def fail_neighbor_list(*args, **kwargs):
        raise AssertionError("explicit MBD dispersion edges must bypass internal neighbor-list construction")

    monkeypatch.setattr("chorus.models.dispersion.dispersion_neighbor_list", fail_neighbor_list)

    cell = (torch.eye(3, dtype=dtype) * 8.0).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7], dtype=torch.long)
    pos = torch.tensor(
        [[0.4, 0.3, 0.2], [1.7, 1.1, 0.8], [2.8, 1.9, 1.3]],
        dtype=dtype,
        requires_grad=True,
    )
    batch = torch.zeros(A.numel(), dtype=torch.long)
    edge_src, edge_dst, edge_shifts = _neighbor_list(pos.detach(), cell[0], r_max=4.5)
    disp_src = torch.tensor([0, 0, 1], dtype=torch.long)
    disp_dst = torch.tensor([1, 2, 2], dtype=torch.long)
    disp_shifts = torch.zeros(3, 3, dtype=dtype)

    energy = model(
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shifts,
    ).sum()
    assert torch.isfinite(energy)
    (grad,) = torch.autograd.grad(energy, pos)
    assert torch.isfinite(grad).all()


def test_force_trainer_makefx_with_explicit_mbd_dispersion_edges(monkeypatch):
    from chorus.training.train_loop import ForceTrainer
    import chorus.training.makefx_compile as makefx_compile

    real_trace = makefx_compile.trace_and_compile_force

    def trace_without_inductor(*args, **kwargs):
        kwargs["do_compile"] = False
        return real_trace(*args, **kwargs)

    monkeypatch.setattr(makefx_compile, "trace_and_compile_force", trace_without_inductor)
    dtype = torch.float64
    torch.manual_seed(41)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_module = None
    model.long_range_exports_reciprocal_source = False
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "edge_sparse"
    model.dispersion_cutoff = 5.0
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        slq_num_probes=2,
        slq_lanczos_steps=4,
        mbd_operator_backend="edge_sparse",
    ).double()

    cell = (torch.eye(3, dtype=dtype) * 8.0).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1], dtype=torch.long)
    pos = torch.rand(A.numel(), 3, dtype=dtype) * 3.0
    batch = torch.zeros(A.numel(), dtype=torch.long)
    edge_src, edge_dst, edge_shifts = _neighbor_list(pos, cell[0], r_max=4.5)
    disp_src, disp_dst, disp_shifts = dispersion_neighbor_list(
        pos,
        batch,
        cell,
        5.0,
        pbc=True,
        canonical_undirected=True,
        method="bruteforce",
    )
    trainer = ForceTrainer(
        model,
        train_loader=[],
        device="cpu",
        dtype=dtype,
        lr_scheduler="none",
        train_makefx_compile=True,
        require_train_makefx_compile=True,
    )

    e_fx, grad_fx = trainer._makefx_forward(
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shifts.to(dtype),
    )
    assert model.dispersion.term._mbd_ramp_count == model.dispersion.term._mbd_ramp_steps

    pos_leaf = pos.detach().requires_grad_(True)
    e_eager = model(
        pos_leaf,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shifts.to(dtype),
    )
    grad_eager = torch.autograd.grad(e_eager.sum(), pos_leaf)[0]
    assert torch.allclose(e_fx, e_eager, atol=1e-8, rtol=1e-8)
    assert torch.allclose(grad_fx, grad_eager, atol=1e-8, rtol=1e-8)


def test_force_trainer_compute_makefx_normalizes_explicit_mbd_edges_multi_graph(monkeypatch):
    from chorus.training.train_loop import ForceTrainer
    import chorus.training.makefx_compile as makefx_compile

    real_trace = makefx_compile.trace_and_compile_force

    def trace_without_inductor(*args, **kwargs):
        kwargs["do_compile"] = False
        return real_trace(*args, **kwargs)

    monkeypatch.setattr(makefx_compile, "trace_and_compile_force", trace_without_inductor)
    dtype = torch.float64
    torch.manual_seed(43)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_module = None
    model.long_range_exports_reciprocal_source = False
    model.long_range_dispersion_mode = "mbd-slq"
    model.mbd_operator_backend = "edge_sparse"
    model.dispersion_cutoff = 5.0
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=5.0,
        pbc=True,
        slq_num_probes=2,
        slq_lanczos_steps=4,
        mbd_operator_backend="edge_sparse",
    ).double()

    cell = torch.stack(
        [torch.eye(3, dtype=dtype) * 8.0, torch.eye(3, dtype=dtype) * 9.0],
        dim=0,
    )
    A = torch.tensor([1, 6, 1, 7, 8], dtype=torch.long)
    batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.4, 0.3, 0.2],
            [1.6, 0.9, 0.7],
            [0.5, 0.4, 0.3],
            [1.7, 1.0, 0.8],
            [2.6, 1.8, 1.1],
        ],
        dtype=dtype,
    )

    edge_parts = []
    for graph_id in range(cell.size(0)):
        atom_ids = torch.nonzero(batch_idx == graph_id, as_tuple=False).view(-1)
        src_g, dst_g, shift_g = _neighbor_list(pos.index_select(0, atom_ids), cell[graph_id], r_max=4.5)
        edge_parts.append((atom_ids.index_select(0, src_g), atom_ids.index_select(0, dst_g), shift_g))
    edge_src = torch.cat([part[0] for part in edge_parts], dim=0)
    edge_dst = torch.cat([part[1] for part in edge_parts], dim=0)
    edge_shifts = torch.cat([part[2] for part in edge_parts], dim=0).to(dtype)

    extras = {
        "dispersion_edge_src": torch.tensor([1, 0, 3, 2, 4, 2, 2, 3, 2], dtype=torch.long),
        "dispersion_edge_dst": torch.tensor([0, 1, 2, 3, 2, 4, 2, 4, 2], dtype=torch.long),
        "dispersion_edge_shifts": torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [-1, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=dtype,
        ),
    }
    expected_src = torch.tensor([0, 2, 2, 2, 3], dtype=torch.long)
    expected_dst = torch.tensor([1, 2, 3, 4, 4], dtype=torch.long)
    expected_shifts = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=dtype,
    )

    force_ref = torch.zeros_like(pos)
    target_energies = torch.zeros(cell.size(0), dtype=dtype)
    stress_ref = torch.zeros(cell.size(0), 3, 3, dtype=dtype)
    batch_tuple = (
        pos,
        A,
        batch_idx,
        force_ref,
        target_energies,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        stress_ref,
        extras,
    )

    trainer_fx = ForceTrainer(
        model,
        train_loader=[],
        device="cpu",
        dtype=dtype,
        lr_scheduler="none",
        train_makefx_compile=True,
        require_train_makefx_compile=True,
    )
    captured = {}
    real_makefx_forward = trainer_fx._makefx_forward

    def checked_makefx_forward(*args, **kwargs):
        captured["src"] = kwargs["dispersion_edge_src"].detach().cpu().clone()
        captured["dst"] = kwargs["dispersion_edge_dst"].detach().cpu().clone()
        captured["shifts"] = kwargs["dispersion_edge_shifts"].detach().cpu().clone()
        return real_makefx_forward(*args, **kwargs)

    monkeypatch.setattr(trainer_fx, "_makefx_forward", checked_makefx_forward)
    out_fx = trainer_fx._compute(batch_tuple, training=True)

    assert torch.equal(captured["src"], expected_src)
    assert torch.equal(captured["dst"], expected_dst)
    assert torch.equal(captured["shifts"], expected_shifts)
    assert not trainer_fx._makefx_disabled
    assert trainer_fx._makefx_cache is not None and len(trainer_fx._makefx_cache._cache) == 1

    trainer_eager = ForceTrainer(
        model,
        train_loader=[],
        device="cpu",
        dtype=dtype,
        lr_scheduler="none",
        train_makefx_compile=False,
    )
    out_eager = trainer_eager._compute(batch_tuple, training=True)
    for key in ("total_loss", "energy_loss", "force_loss", "force_rmse", "energy_rmse_avg"):
        assert torch.allclose(out_fx[key], out_eager[key], atol=1e-8, rtol=1e-8), key


def test_mfftorch_requires_explicit_mbd_dispersion_edges():
    repo = Path(__file__).resolve().parents[2]
    engine = repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_torch_engine.cpp"
    engine_h = repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_torch_engine.h"
    pair = repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.cpp"
    text = engine.read_text()
    header = engine_h.read_text()
    pair_text = pair.read_text()
    assert "expects explicit MBD dispersion edges" in text
    assert "Add 'dispersion <cutoff>' to pair_style mff/torch" in text
    assert "metadata_requires_mbd_dispersion_edges" in text
    assert text.count("requires_mbd_dispersion_edges()") >= 2
    assert 'dispersion_deployment_graph_rule_ == "explicit_canonical_single_image_edge_sparse"' in header
    assert 'mbd_operator_backend_ != "pme_fft"' not in header
    assert "dispersion_cutoff_ > 0.0" in header
    assert "double dispersion_cutoff() const" in header
    assert "bool requires_mbd_dispersion_edges() const" in header
    assert "engine_->dispersion_cutoff()" in pair_text
    assert "engine_->requires_mbd_dispersion_edges()" in pair_text
    assert "does not match the model MBD dispersion_cutoff" in pair_text
    assert "use the trained cutoff so deployment and training build the same dispersion graph rule" in pair_text


def test_mfftorch_mbd_edge_sparse_uses_exported_c2_dispersion_window():
    repo = Path(__file__).resolve().parents[2]
    engine = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_torch_engine.cpp").read_text()
    engine_h = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_torch_engine.h").read_text()
    mbd_h = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_mbd_solver.h").read_text()
    mbd = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_mbd_solver.cpp").read_text()
    pair = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.cpp").read_text()
    export = (repo / "chorus" / "cli" / "export_aoti_core.py").read_text()
    export_libtorch = (repo / "chorus" / "cli" / "export_libtorch_core.py").read_text()
    train_loop = (repo / "chorus" / "training" / "train_loop.py").read_text()

    assert '"dispersion_min_cutoff": _disp_min_cutoff' in export
    assert '"dispersion_switch_width": _disp_switch_width' in export
    assert '"dispersion_min_cutoff": float(' in export_libtorch
    assert '"dispersion_switch_width": float(' in export_libtorch
    assert '"dispersion_min_cutoff"' in train_loop
    assert '"dispersion_switch_width"' in train_loop
    assert 'parse_double_from_metadata(content, "\\"dispersion_min_cutoff\\""' in engine
    assert 'parse_double_from_metadata(content, "\\"dispersion_switch_width\\""' in engine
    assert "double dispersion_min_cutoff() const" in engine_h
    assert "double dispersion_switch_width() const" in engine_h
    assert "double min_cutoff = 0.0" in mbd_h
    assert "double switch_width = 0.5" in mbd_h
    assert "dispersion_edge_window" in mbd
    assert "(damp * window).unsqueeze(-1) * t_bare" in mbd
    assert "config_.real_cutoff = real_cutoff" in mbd
    assert "mcfg.real_cutoff = engine_->dispersion_cutoff()" in pair
    assert "mcfg.min_cutoff = engine_->dispersion_min_cutoff()" in pair
    assert "mcfg.switch_width = engine_->dispersion_switch_width()" in pair


def test_mfftorch_requests_dispersion_cutoff_for_lammps_neighbor_list():
    repo = Path(__file__).resolve().parents[2]
    pair = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.cpp").read_text()
    pair_kk = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch_kokkos.cpp").read_text()
    header = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.h").read_text()

    assert "request_cut_global_ = std::max(request_cut_global_, dispersion_cut_global_)" in pair
    assert "request_cutsq_global_ = request_cut_global_ * request_cut_global_" in pair
    assert "cutsq[i][j] = request_cutsq_global_" in pair
    assert "return request_cut_global_" in pair
    assert "const double halo = std::max(static_cast<double>(mp_depth_) * cut_global_, request_cut_global_)" in pair
    assert "const double halo = std::max(static_cast<double>(mp_depth_) * cut_global_, request_cut_global_)" in pair_kk
    assert "double request_cut_global_" in header


def test_mfftorch_reciprocal_solver_does_not_disable_global_grad_mode():
    repo = Path(__file__).resolve().parents[2]
    pair = repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.cpp"
    text = pair.read_text()
    assert "torch::GradMode::set_enabled(false)" not in text


def test_mfftorch_keeps_single_canonical_mbd_edge_graphs():
    repo = Path(__file__).resolve().parents[2]
    pair = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.cpp").read_text()
    pair_kk = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch_kokkos.cpp").read_text()
    export = (repo / "chorus" / "cli" / "export_aoti_core.py").read_text()
    assert "Edisp <= 1" not in pair
    assert "Edispfiltered <= 1" not in pair_kk
    assert "E <= 1 && Edisp <= 1" not in pair_kk
    assert "Etotal <= 1" not in pair_kk
    assert "disp_src.numel() < 2" not in export
    assert 'Dim("n_dispersion_edges", min=1)' in export
    assert "produced no dispersion edges" in export


def test_mfftorch_mbd_dispersion_edges_are_canonical_and_sorted():
    repo = Path(__file__).resolve().parents[2]
    pair = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.cpp").read_text()
    pair_kk = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch_kokkos.cpp").read_text()
    assert "lexicographic_positive_shift" in pair
    assert "keep_canonical_mbd_edge" in pair
    assert "sort_edge_vectors(buf_disp_edge_src_cpu_, buf_disp_edge_dst_cpu_, buf_disp_edge_shifts_cpu_)" in pair
    assert "keep_canonical_mbd_edge(jl, i, out_sx, out_sy, out_sz)" in pair
    assert "keep_canonical_mbd_edge(j, i, -sx, -sy, -sz)" in pair
    assert "jl < i" not in pair
    assert "j < i" not in pair

    assert "lexicographic_positive_shift" in pair_kk
    assert "keep_canonical_mbd_edge" in pair_kk
    assert "sort_edge_vectors(buf_disp_edge_src_cpu_, buf_disp_edge_dst_cpu_, buf_disp_edge_shifts_cpu_)" in pair_kk
    assert "sort_edges_by_dst_src_shift" in pair_kk
    assert "shift_i64.select(1, 2)" in pair_kk
    assert "keep_canonical_mbd_edge(jl, i, out_sx, out_sy, out_sz)" in pair_kk
    assert "keep_canonical_mbd_edge(src, i, -out_sx, -out_sy, -out_sz)" in pair_kk
    assert "const bool keep_disp = keep_disp_raw && src < i" not in pair_kk


def test_mfftorch_mbd_dispersion_rejects_multi_image_runtime_cutoff():
    repo = Path(__file__).resolve().parents[2]
    pair = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.cpp").read_text()
    pair_kk = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch_kokkos.cpp").read_text()

    for text in (pair, pair_kk):
        assert "periodic_face_height" in text
        assert "validate_mbd_dispersion_single_image_cutoff" in text
        assert "2.0 * dispersion_cutoff > height" in text
        assert "exact multi-image/self-image" in text
        assert "MBD graph used by the Python brute-force small-cell path" in text
        assert "train/deploy with mbd_operator_backend=pme_fft" in text
        assert "engine_->requires_mbd_dispersion_edges()" in text


def test_mfftorch_aoti_mbd_reciprocal_direct_path_contract():
    repo = Path(__file__).resolve().parents[2]
    engine = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_torch_engine.cpp").read_text()
    engine_h = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_torch_engine.h").read_text()
    export = (repo / "chorus" / "cli" / "export_aoti_core.py").read_text()
    export_libtorch = (repo / "chorus" / "cli" / "export_libtorch_core.py").read_text()
    lammps_mliap = (repo / "chorus" / "interfaces" / "lammps_mliap.py").read_text()
    train_loop = (repo / "chorus" / "training" / "train_loop.py").read_text()
    assert "combined AOTI (MBD dispersion edges + runtime reciprocal source)" in engine
    assert "detached reciprocal_source (no fallback needed)" in engine
    assert "run_forward_backward(pos0, A, edge_src, edge_dst, edge_shifts, cell" in engine
    assert export.count('mf.write(f"fallback {args.fallback}\\n")') >= 2
    assert "if emit_rs or use_explicit_dispersion_edges:" in export
    assert "_long_range_deploy_metadata(" in export
    assert "export_reciprocal_source=emit_rs" in export
    assert "canonical_undirected=dispersion_mode_uses_canonical_edges(dispersion_mode)" in export
    assert 'method=str(getattr(_bare, "dispersion_neighbor_method", "auto"))' in export
    assert 'bruteforce_threshold=int(getattr(_bare, "dispersion_bruteforce_threshold", 1024))' in export
    assert 'max_num_neighbors=getattr(_bare, "dispersion_max_num_neighbors", None)' in export
    assert 'getattr(_bare, "dispersion_allow_large_bruteforce_fallback", False)' in export
    assert '"dispersion_neighbor_method": str(getattr(model, "dispersion_neighbor_method", "auto"))' in export
    assert '"dispersion_bruteforce_threshold": int(getattr(model, "dispersion_bruteforce_threshold", 1024))' in export
    assert '"dispersion_training_graph_rule": dispersion_training_graph_rule(' in export
    assert '"dispersion_deployment_graph_rule": dispersion_deployment_graph_rule(' in export
    assert '"dispersion_train_deploy_graph_compatibility": dispersion_train_deploy_graph_compatibility(' in export
    assert 'mbd_operator_backend = str(getattr(model, "mbd_operator_backend", "edge_sparse"))' in export
    assert '"mbd_operator_backend": mbd_operator_backend' in export
    assert '"mbd_pme_mesh_size": int(getattr(model, "mbd_pme_mesh_size", 16))' in export
    assert '"mbd_pme_assignment": str(getattr(model, "mbd_pme_assignment", "pcs"))' in export
    assert '"long_range_mbd_energy_scale": _mbd_es' in export
    assert '"mbd_operator_backend": str(getattr(metadata_model, "mbd_operator_backend", "edge_sparse"))' in export_libtorch
    assert '"dispersion_neighbor_method": str(getattr(metadata_model, "dispersion_neighbor_method", "auto"))' in export_libtorch
    assert '"dispersion_bruteforce_threshold": int(getattr(metadata_model, "dispersion_bruteforce_threshold", 1024))' in export_libtorch
    assert '"dispersion_training_graph_rule": dispersion_training_graph_rule(' in export_libtorch
    assert '"dispersion_deployment_graph_rule": dispersion_deployment_graph_rule(' in export_libtorch
    assert (
        '"dispersion_train_deploy_graph_compatibility": dispersion_train_deploy_graph_compatibility('
        in export_libtorch
    )
    assert '"mbd_pme_mesh_size": int(getattr(metadata_model, "mbd_pme_mesh_size", 16))' in export_libtorch
    assert '"mbd_pme_assignment": str(getattr(metadata_model, "mbd_pme_assignment", "pcs"))' in export_libtorch
    assert '"long_range_mbd_energy_scale": long_range_mbd_energy_scale' in export_libtorch
    assert "validate_dispersion_training_graph_rule(" in lammps_mliap
    assert "checkpoint dispersion_training_graph_rule" in lammps_mliap
    assert "validate_dispersion_deployment_graph_rule(" in lammps_mliap
    assert "checkpoint dispersion_deployment_graph_rule" in lammps_mliap
    assert "validate_dispersion_train_deploy_graph_compatibility(" in lammps_mliap
    assert "checkpoint dispersion_train_deploy_graph_compatibility" in lammps_mliap
    assert '"mbd_operator_backend"' in train_loop
    assert '"dispersion_neighbor_method"' in train_loop
    assert '"dispersion_bruteforce_threshold"' in train_loop
    assert '"dispersion_allow_large_bruteforce_fallback"' in train_loop
    assert '"dispersion_training_graph_rule"' in train_loop
    assert '"dispersion_deployment_graph_rule"' in train_loop
    assert '"dispersion_train_deploy_graph_compatibility"' in train_loop
    assert '"mbd_pme_mesh_size"' in train_loop
    assert '"mbd_pme_ewald_alpha_prefactor"' in train_loop
    assert "expected_dispersion_deployment_graph_rule" in engine
    assert "expected_dispersion_training_graph_rule" in engine
    assert "reconcile_dispersion_deployment_graph_rule" in engine
    assert "reconcile_dispersion_training_graph_rule" in engine
    assert "export metadata is internally inconsistent" in engine
    assert "has_dispersion_graph_rule" in engine
    assert "has_dispersion_training_graph_rule" in engine
    assert '\\"dispersion_training_graph_rule\\"' in engine
    assert '\\"dispersion_deployment_graph_rule\\"' in engine
    assert 'parse_string_from_metadata(content, "\\"mbd_operator_backend\\""' in engine
    assert 'parse_double_from_metadata(content, "\\"long_range_mbd_energy_scale\\""' in engine
    assert "long_range_mbd_energy_scale() const" in engine_h
    assert "double long_range_mbd_energy_scale_ = 1.0" in engine_h
    mbd_solver_h = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_mbd_solver.h").read_text()
    mbd_solver = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "mff_mbd_solver.cpp").read_text()
    pair = (repo / "lammps_user_mfftorch" / "src" / "USER-MFFTORCH" / "pair_mff_torch.cpp").read_text()
    assert "double energy_scale = 1.0" in mbd_solver_h
    assert "return config_.energy_scale * (0.5 * tr - 1.5 * omega.sum())" in mbd_solver
    assert "mcfg.energy_scale = engine_->long_range_mbd_energy_scale()" in pair
    assert "mcfg.cheb_lmin = 0.0" in pair
    assert "mcfg.cheb_lmax = 0.0" in pair
    assert "mcfg.cheb_lmin = 0.3" not in pair
    assert "mcfg.cheb_lmax = 3.0" not in pair
    assert "mbd_operator_backend=pme_fft IS supported at deployment now" in engine
    assert "C++ MBD solver runs the matching reciprocal PME operator" in export
    assert "pme_fft (reciprocal-only, no cutoff dispersion edges) IS supported" in export_libtorch
    assert "const std::string& dispersion_training_graph_rule() const" in engine_h
    assert "const std::string& dispersion_deployment_graph_rule() const" in engine_h
    assert 'std::string dispersion_training_graph_rule_ = "none"' in engine_h
    assert 'std::string dispersion_deployment_graph_rule_ = "none"' in engine_h
    assert 'dispersion_deployment_graph_rule_ == "explicit_canonical_single_image_edge_sparse"' in engine_h
    assert 'mbd_operator_backend_ != "pme_fft"' not in engine_h
    assert "const std::string& mbd_operator_backend() const" in engine_h


if __name__ == "__main__":
    test_dispersion_physics()
    print("OK: dispersion physics (attractive, decaying, BJ-finite, attractive forces, invariant)")
    test_long_range_dispersion_wrapper_matches_pairwise_edge_mode()
    print("OK: unified long-range dispersion wrapper matches pairwise edge mode")
    test_many_body_dispersion_is_finite_invariant_and_nonadditive()
    print("OK: MBD finite, differentiable, invariant, and nonadditive")
    test_many_body_dispersion_slq_basis_matches_dense_oracle()
    print("OK: basis SLQ-MBD matches dense MBD oracle")
    test_model_accepts_explicit_dispersion_neighbor_list()
    print("OK: model accepts an explicit dispersion neighbor list")
    test_model_mbd_dispersion_smoke()
    print("OK: model-level MBD dispersion smoke")
    test_model_mbd_slq_dispersion_smoke()
    print("OK: model-level SLQ-MBD dispersion smoke")
    test_mbd_torchscript_core_accepts_variable_atom_and_edge_counts()
    print("OK: traced MBD deployment core accepts variable atom/edge counts")
    test_mbd_slq_torchscript_core_accepts_variable_atom_and_edge_counts()
    print("OK: traced SLQ-MBD deployment core accepts variable atom/edge counts")
    test_dispersion_neighbor_list_matches_bruteforce()
    print("OK: dispersion neighbor list matches brute-force periodic search")
    test_model_complete_long_range_smoke()
    print("OK: complete long-range smoke (multipole electrostatics + dispersion, both train)")
