"""Integration test for the multipole long-range path.

Test A: the reciprocal multipole energy (LatentReciprocalLongRange.forward_multipole
-> MeshLongRangeKernel3D.multipole_energy) is rotation-invariant under a joint
rotation of positions, lattice, dipole (R mu) and quadrupole (R Q R^T), and is
differentiable w.r.t. positions.
"""

from __future__ import annotations

import torch

from mace_ictc.models.long_range import (
    ReciprocalSpectralKernel3D,
    build_feature_spectral_module,
    build_long_range_module,
)


def _random_rotation(dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


def _make_lr(dtype):
    lr = build_long_range_module(
        mode="reciprocal-spectral-v1",
        feature_dim=8,
        reciprocal_backend="mesh_fft",
        boundary="periodic",
        mesh_size=32,
        source_channels=1,
        green_mode="poisson",
        assignment="pcs",  # higher-order (vs cic) for accurate, translation-stable PME
        neutralize=True,
        mesh_fft_full_ewald=True,  # Ewald Gaussian screening -> band-limited reciprocal sum
    ).to(dtype)
    if getattr(lr, "energy_scale", None) is not None:
        with torch.no_grad():
            lr.energy_scale.fill_(1.0)  # init may be 0 -> make the test non-trivial
    return lr


def test_scalar_mesh_fft_bootstraps_from_zero_energy_scale():
    """Scalar mesh_fft must not zero the source head: E is gated to zero, but the gate can train."""
    dtype = torch.float64
    torch.manual_seed(17)
    lr = build_long_range_module(
        mode="reciprocal-spectral-v1",
        feature_dim=6,
        hidden_dim=12,
        reciprocal_backend="mesh_fft",
        boundary="periodic",
        mesh_size=8,
        source_channels=2,
        green_mode="poisson",
        assignment="pcs",
        neutralize=True,
    ).to(dtype)
    features = torch.randn(5, 6, dtype=dtype)
    pos = torch.tensor(
        [[0.2, 0.3, 0.4], [1.0, 0.5, 1.2], [1.7, 1.8, 0.9], [2.4, 1.1, 1.5], [2.8, 2.2, 2.3]],
        dtype=dtype,
    )
    batch = torch.zeros(pos.size(0), dtype=torch.long)
    cell = (torch.eye(3, dtype=dtype) * 4.0).unsqueeze(0)

    out, source = lr(features, pos, batch, cell, return_source=True)
    assert torch.allclose(out.sum(), torch.zeros((), dtype=dtype))
    assert source.abs().max() > 0.0, "mesh_fft source head must start non-zero"
    loss = (out.sum() - 1.0).square()
    loss.backward()
    assert lr.energy_scale.grad is not None and lr.energy_scale.grad.abs() > 0.0


def test_direct_kspace_rotation_invariance_triclinic_cell():
    """Direct k-space must build k = 2*pi*m@inv(cell)^T; otherwise radial weights depend on orientation."""
    dtype = torch.float64
    torch.manual_seed(23)
    kernel = ReciprocalSpectralKernel3D(kmax=3, include_k0=False, energy_partition="uniform").to(dtype)
    cell = torch.tensor(
        [[[5.0, 0.7, 0.2], [0.1, 4.4, 0.6], [0.3, 0.2, 6.2]]],
        dtype=dtype,
    )
    frac = torch.tensor(
        [[0.10, 0.20, 0.30], [0.25, 0.70, 0.40], [0.60, 0.15, 0.80], [0.90, 0.40, 0.10]],
        dtype=dtype,
    )
    pos = frac @ cell[0]
    source = torch.tensor([[0.4], [-0.7], [0.5], [-0.2]], dtype=dtype)
    source = source - source.mean(dim=0, keepdim=True)
    batch = torch.zeros(pos.size(0), dtype=torch.long)

    rotation = _random_rotation(dtype)
    e0 = kernel(pos, batch, cell, source).sum()
    e1 = kernel(pos @ rotation.T, batch, cell @ rotation.T, source).sum()
    assert torch.allclose(e0, e1, atol=1e-10, rtol=1e-10), (e0, e1, (e0 - e1).abs())


def test_forward_multipole_rotation_invariance_and_forces():
    dtype = torch.float64
    lr = _make_lr(dtype)
    n = 5
    g = torch.Generator().manual_seed(1)
    L = 6.0
    pos = torch.rand(n, 3, generator=g, dtype=dtype) * L
    batch = torch.zeros(n, dtype=torch.long)
    cell = (torch.eye(3, dtype=dtype) * L).unsqueeze(0)  # [1,3,3], rows = lattice vectors
    q = torch.randn(n, 1, generator=g, dtype=dtype)
    mu = torch.randn(n, 1, 3, generator=g, dtype=dtype)
    Q = torch.randn(n, 1, 3, 3, generator=g, dtype=dtype)
    Q = 0.5 * (Q + Q.transpose(-1, -2))

    e1 = lr.forward_multipole(pos, batch, cell, q, mu, Q).sum()

    R = _random_rotation(dtype)
    pos_r = pos @ R.T
    cell_r = (cell[0] @ R.T).unsqueeze(0)
    mu_r = mu @ R.T
    Q_r = torch.einsum("ij,ncjk,lk->ncil", R, Q, R)
    e2 = lr.forward_multipole(pos_r, batch, cell_r, q, mu_r, Q_r).sum()

    assert torch.allclose(e1, e2, atol=1e-8), f"not rotation-invariant: {(e1 - e2).abs().item()}"

    # forces flow and are translation-invariant (uniform shift -> same energy)
    pos2 = pos.clone().requires_grad_(True)
    e = lr.forward_multipole(pos2, batch, cell, q, mu, Q).sum()
    (grad,) = torch.autograd.grad(e, pos2)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0, "no/invalid force"

    # exact invariance under a full lattice-vector translation: confirms periodic
    # wrapping/spreading is correct. (In-cell sub-grid translation is only as accurate
    # as the underlying mesh PME -- CIC + mesh resolution -- which is a property/knob of
    # the long_range module, not of this multipole wiring; rotation-invariance above is
    # the equivariance-correctness check.)
    e_cell = lr.forward_multipole(pos + cell[0, 0], batch, cell, q, mu, Q).sum()
    assert torch.allclose(e1, e_cell, atol=1e-9), (
        f"not invariant under a lattice-vector shift: {(e1 - e_cell).abs().item():.2e}"
    )

    # sub-grid (in-cell) translation accuracy: with Ewald screening (full_ewald) +
    # higher-order assignment (pcs) this is now ~1e-3 (was tens of % with bare
    # poisson + CIC -- the #2 fix).
    e_sub = lr.forward_multipole(pos + torch.tensor([0.05, 0.05, 0.05], dtype=dtype), batch, cell, q, mu, Q).sum()
    rel_sub = (e1 - e_sub).abs() / e1.abs().clamp_min(1e-12)
    assert rel_sub < 5e-3, f"sub-grid translation error too large: {rel_sub.item():.2e}"


def _build_model(*, max_multipole_l: int, dispersion: bool = False):
    from mace_ictc.models.pure_cartesian_ictd_fix import PureCartesianICTDFix

    torch.set_default_dtype(torch.float64)
    c = 8
    return PureCartesianICTDFix(
        max_embed_radius=4.5,
        main_max_radius=4.5,
        main_number_of_basis=8,
        hidden_dim_conv=c,
        hidden_dim_sh=c,
        hidden_dim=c,
        channel_in2=c,
        embedding_dim=c,
        max_atomvalue=10,
        atomic_numbers=[1, 6, 7, 8],
        num_interaction=2,
        function_type_main="bessel",
        lmax=2,
        ictd_fix_edge_lmax=2,
        ictd_fix_route="baseline",
        ictd_fix_product_backend="ictd-bridge-u",
        ictd_fix_use_reduced_cg=False,
        save_contraction_order=3,
        avg_num_neighbors=8.0,
        angular_basis="ictd",
        internal_compute_dtype=torch.float64,
        device="cpu",
        long_range_mode="reciprocal-spectral-v1",
        long_range_reciprocal_backend="mesh_fft",
        long_range_boundary="periodic",
        long_range_mesh_size=16,
        long_range_assignment="pcs",
        long_range_mesh_fft_full_ewald=True,
        long_range_max_multipole_l=int(max_multipole_l),
        long_range_dispersion=bool(dispersion),
    )


def test_model_multipole_gating():
    # OFF (default lmax 0): no multipole readout -> the scalar latent-source path is used,
    # so the long-range-off / bridge-U numerics are byte-identical.
    off = _build_model(max_multipole_l=0)
    assert off.multipole_readout is None
    assert off.long_range_max_multipole_l == 0
    # ON: the equivariant multipole readout is wired in.
    on = _build_model(max_multipole_l=2)
    assert on.multipole_readout is not None
    assert on.long_range_max_multipole_l == 2


def _neighbor_list(pos, cell, r_max):
    """Minimal-image periodic neighbor list. Returns edge_src(j), edge_dst(i), unit_shifts;
    model edge vec = pos[i] - pos[j] + unit_shifts @ cell."""
    import itertools

    n = pos.shape[0]
    src, dst, shifts = [], [], []
    for i in range(n):
        for j in range(n):
            for s in itertools.product([-1, 0, 1], repeat=3):
                s_t = torch.tensor(s, dtype=pos.dtype)
                d = pos[i] - pos[j] + s_t @ cell
                r = float(d.norm())
                if 0.0 < r <= r_max:
                    src.append(j)
                    dst.append(i)
                    shifts.append(list(s))
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
        torch.tensor(shifts, dtype=pos.dtype),
    )


def _two_graph_inputs(dtype=torch.float64):
    torch.manual_seed(12)
    L0 = 8.0
    L1 = 9.0
    cell0 = torch.eye(3, dtype=dtype) * L0
    cell1 = torch.eye(3, dtype=dtype) * L1
    A0 = torch.tensor([1, 6, 7, 8, 1], dtype=torch.long)
    A1 = torch.tensor([6, 8, 1, 7], dtype=torch.long)
    pos0 = torch.rand(A0.numel(), 3, dtype=dtype) * L0
    pos1 = torch.rand(A1.numel(), 3, dtype=dtype) * L1
    es0, ed0, sh0 = _neighbor_list(pos0, cell0, r_max=4.5)
    es1, ed1, sh1 = _neighbor_list(pos1, cell1, r_max=4.5)
    n0 = A0.numel()
    return dict(
        pos=torch.cat([pos0, pos1], dim=0),
        A=torch.cat([A0, A1], dim=0),
        batch=torch.cat([
            torch.zeros(A0.numel(), dtype=torch.long),
            torch.ones(A1.numel(), dtype=torch.long),
        ]),
        edge_src=torch.cat([es0, es1 + n0], dim=0),
        edge_dst=torch.cat([ed0, ed1 + n0], dim=0),
        shifts=torch.cat([sh0, sh1], dim=0),
        cell=torch.stack([cell0, cell1], dim=0),
        split=n0,
    )


def test_collate_offsets_explicit_dispersion_edges():
    from mace_ictc.data.collate import collate_fn_h5

    dtype = torch.float64
    sample0 = {
        "pos": torch.zeros(2, 3, dtype=dtype),
        "A": torch.tensor([1, 6], dtype=torch.long),
        "force": torch.zeros(2, 3, dtype=dtype),
        "y": torch.tensor([0.0], dtype=dtype),
        "edge_src": torch.tensor([0], dtype=torch.long),
        "edge_dst": torch.tensor([1], dtype=torch.long),
        "edge_shifts": torch.zeros(1, 3, dtype=dtype),
        "dispersion_edge_src": torch.tensor([0], dtype=torch.long),
        "dispersion_edge_dst": torch.tensor([1], dtype=torch.long),
        "dispersion_edge_shifts": torch.zeros(1, 3, dtype=dtype),
        "cell": torch.eye(3, dtype=dtype),
        "stress": torch.zeros(3, 3, dtype=dtype),
    }
    sample1 = {
        "pos": torch.zeros(3, 3, dtype=dtype),
        "A": torch.tensor([7, 8, 1], dtype=torch.long),
        "force": torch.zeros(3, 3, dtype=dtype),
        "y": torch.tensor([0.0], dtype=dtype),
        "edge_src": torch.tensor([0], dtype=torch.long),
        "edge_dst": torch.tensor([2], dtype=torch.long),
        "edge_shifts": torch.zeros(1, 3, dtype=dtype),
        "dispersion_edge_src": torch.tensor([0, 1], dtype=torch.long),
        "dispersion_edge_dst": torch.tensor([2, 0], dtype=torch.long),
        "dispersion_edge_shifts": torch.zeros(2, 3, dtype=dtype),
        "cell": torch.eye(3, dtype=dtype),
        "stress": torch.zeros(3, 3, dtype=dtype),
    }

    *_, extras = collate_fn_h5([sample0, sample1])
    assert torch.equal(extras["dispersion_edge_src"], torch.tensor([0, 2, 3]))
    assert torch.equal(extras["dispersion_edge_dst"], torch.tensor([1, 4, 2]))
    assert extras["dispersion_edge_shifts"].shape == (3, 3)


def test_force_trainer_disables_tf32():
    from mace_ictc.training.train_loop import ForceTrainer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = _build_model(max_multipole_l=0).float()
    ForceTrainer(model, train_loader=[], device="cpu", dtype=torch.float32, lr_scheduler="none")
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_model_multipole_forward_smoke():
    """End-to-end: a wired model with multipole long-range ON runs forward, gives a finite
    energy + finite forces, and a rotation-invariant total energy."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().eval()
    torch.manual_seed(0)
    L = 8.0
    cell = torch.eye(3, dtype=torch.float64) * L
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8], dtype=torch.long)
    n = A.numel()
    pos0 = torch.rand(n, 3, dtype=torch.float64) * L
    batch = torch.zeros(n, dtype=torch.long)
    edge_src, edge_dst, unit_shifts = _neighbor_list(pos0, cell, r_max=4.5)
    assert edge_src.numel() > 0, "empty neighbor list"

    pos = pos0.clone().requires_grad_(True)
    e = model(pos, A, batch, edge_src, edge_dst, unit_shifts, cell.reshape(1, 3, 3)).sum()
    assert torch.isfinite(e), "energy not finite"
    (forces,) = torch.autograd.grad(-e, pos)
    assert torch.isfinite(forces).all() and forces.abs().sum() > 0, "bad forces"

    # rotation invariance: rotate positions + lattice consistently (integer shifts unchanged)
    R = _random_rotation(torch.float64)
    pos_r = (pos0 @ R.T).requires_grad_(True)
    e_r = model(pos_r, A, batch, edge_src, edge_dst, unit_shifts, (cell @ R.T).reshape(1, 3, 3)).sum()
    assert torch.allclose(e.detach(), e_r.detach(), atol=1e-6), (
        f"total energy not rotation-invariant: {(e - e_r).abs().item():.2e}"
    )


def test_model_multipole_gentle_start_warm_safe():
    """End-to-end warm-start safety: a freshly-built multipole model starts gentle. Toggling the
    multipole gate isolates its contribution -- gate=0 -> zero multipole energy (pure backbone),
    so the long-range part at the default gate (0.1) is ~gate**2 (quadratic energy) of the ungated
    value. That small initial contribution is what lets `--finetune` add multipole electrostatics on
    a trained backbone without destabilizing it."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().eval()
    # the model wires the gate through to the readout with the small gentle-start default
    assert abs(float(model.multipole_readout.source_scale) - 0.1) < 1e-6

    torch.manual_seed(0)
    L = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8], dtype=torch.long)
    n = A.numel()
    pos = torch.rand(n, 3, dtype=torch.float64) * L
    batch = torch.zeros(n, dtype=torch.long)
    edge_src, edge_dst, shifts = _neighbor_list(pos, cell[0], r_max=4.5)

    def total_energy_with_gate(g: float) -> torch.Tensor:
        with torch.no_grad():
            model.multipole_readout.source_scale.fill_(g)
        return model(pos, A, batch, edge_src, edge_dst, shifts, cell).sum()

    e_backbone = total_energy_with_gate(0.0)  # multipole sources = 0 -> zero long-range energy
    e_gentle = total_energy_with_gate(0.1)
    e_full = total_energy_with_gate(1.0)
    lr_gentle = (e_gentle - e_backbone).abs()
    lr_full = (e_full - e_backbone).abs()

    assert lr_full > 0, "ungated multipole head must contribute a non-trivial long-range energy"
    # energy is quadratic in the sources, so gate 0.1 -> ~0.01 of the ungated contribution
    assert lr_gentle < 0.05 * lr_full, (
        f"gentle-start gate not gentle: lr_gentle/lr_full = {(lr_gentle / lr_full).item():.3e}"
    )


def test_long_range_multi_graph_matches_separate_graphs():
    """Batched mesh long-range must be separable across structures."""
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    inputs = _two_graph_inputs(dtype)
    n0 = inputs["split"]

    for max_l in (0, 2):
        model = _build_model(max_multipole_l=max_l).double().eval()
        for m in model.modules():
            if getattr(m, "energy_scale", None) is not None:
                with torch.no_grad():
                    m.energy_scale.fill_(0.1)

        e_batched = model(
            inputs["pos"],
            inputs["A"],
            inputs["batch"],
            inputs["edge_src"],
            inputs["edge_dst"],
            inputs["shifts"],
            inputs["cell"],
        )

        edge0_mask = inputs["edge_src"] < n0
        e0 = model(
            inputs["pos"][:n0],
            inputs["A"][:n0],
            torch.zeros(n0, dtype=torch.long),
            inputs["edge_src"][edge0_mask],
            inputs["edge_dst"][edge0_mask],
            inputs["shifts"][edge0_mask],
            inputs["cell"][:1],
        )
        edge1_mask = inputs["edge_src"] >= n0
        e1 = model(
            inputs["pos"][n0:],
            inputs["A"][n0:],
            torch.zeros(inputs["A"].numel() - n0, dtype=torch.long),
            inputs["edge_src"][edge1_mask] - n0,
            inputs["edge_dst"][edge1_mask] - n0,
            inputs["shifts"][edge1_mask],
            inputs["cell"][1:],
        )
        assert torch.allclose(e_batched, torch.cat([e0, e1], dim=0), atol=1e-8), max_l


def test_feature_spectral_multi_graph_matches_separate_graphs():
    """Feature-spectral long-range residual also uses the batched mesh path."""
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    inputs = _two_graph_inputs(dtype)
    n0 = inputs["split"]
    feature_dim = 8
    torch.manual_seed(13)
    x = torch.randn(inputs["A"].numel(), feature_dim, dtype=dtype)
    block = build_feature_spectral_module(
        mode="fft",
        feature_dim=feature_dim,
        bottleneck_dim=4,
        mesh_size=16,
        filter_hidden_dim=8,
        boundary="periodic",
        neutralize=True,
        assignment="pcs",
        gate_init=0.2,
    ).double().eval()

    pos = inputs["pos"].clone().requires_grad_(True)
    y_batch, _ = block(x, pos, inputs["batch"], inputs["cell"])
    y0, _ = block(
        x[:n0],
        inputs["pos"][:n0],
        torch.zeros(n0, dtype=torch.long),
        inputs["cell"][:1],
    )
    y1, _ = block(
        x[n0:],
        inputs["pos"][n0:],
        torch.zeros(inputs["A"].numel() - n0, dtype=torch.long),
        inputs["cell"][1:],
    )
    assert torch.allclose(y_batch, torch.cat([y0, y1], dim=0), atol=1e-8)
    (grad,) = torch.autograd.grad(y_batch.sum(), pos)
    assert torch.isfinite(grad).all()


def test_model_multipole_training_smoke():
    """A few optimizer steps with multipole long-range ON: the loss decreases and the
    multipole readout receives gradients (the long-range path is trainable end-to-end)."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().train()
    # the long-range energy_scale inits to 0 (no-op); activate it so the multipole path
    # is exercised from step 1.
    for m in model.modules():
        if getattr(m, "energy_scale", None) is not None:
            with torch.no_grad():
                m.energy_scale.fill_(0.1)

    torch.manual_seed(0)
    L = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8], dtype=torch.long)
    n = A.numel()
    pos0 = torch.rand(n, 3, dtype=torch.float64) * L
    batch = torch.zeros(n, dtype=torch.long)
    edge_src, edge_dst, shifts = _neighbor_list(pos0, cell[0], r_max=4.5)
    target_f = torch.randn(n, 3, dtype=torch.float64) * 0.1

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        pos = pos0.clone().requires_grad_(True)
        e = model(pos, A, batch, edge_src, edge_dst, shifts, cell).sum()
        (force,) = torch.autograd.grad(-e, pos, create_graph=True)
        loss = ((force - target_f) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3e} -> {losses[-1]:.3e}"
    mp_grads = [p.grad for p in model.multipole_readout.parameters() if p.grad is not None]
    assert mp_grads and any(g.abs().sum() > 0 for g in mp_grads), "multipole readout received no gradient"
    return losses[0], losses[-1]


def test_model_multipole_makefx_single_graph_trace():
    """Regression: single-graph multipole long-range training can be make_fx-traced.

    Multi-graph batches still need a batched mesh implementation; this locks the
    common one-cell path so it does not regress to data-dependent nonzero/int(numel)
    control flow.
    """
    from mace_ictc.training.makefx_compile import make_force_compute_fn, trace_and_compile_force

    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().train()
    for m in model.modules():
        if getattr(m, "energy_scale", None) is not None:
            with torch.no_grad():
                m.energy_scale.fill_(0.1)

    torch.manual_seed(7)
    L = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8], dtype=torch.long)
    pos = torch.rand(A.numel(), 3, dtype=torch.float64) * L
    batch = torch.zeros(A.numel(), dtype=torch.long)
    edge_src, edge_dst, shifts = _neighbor_list(pos, cell[0], r_max=4.5)

    gm = trace_and_compile_force(
        model,
        (pos, A, batch, edge_src, edge_dst, shifts, cell),
        training=True,
        compute_fn=make_force_compute_fn(model, training=True),
        do_compile=False,
    )
    energy, force = gm(pos, A, batch, edge_src, edge_dst, shifts, cell)
    assert torch.isfinite(energy)
    assert torch.isfinite(force).all() and force.shape == pos.shape


def test_model_multipole_makefx_multi_graph_trace():
    """Regression: multi-graph multipole long-range training can be make_fx-traced."""
    from mace_ictc.training.makefx_compile import make_force_compute_fn, trace_and_compile_force

    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().train()
    for m in model.modules():
        if getattr(m, "energy_scale", None) is not None:
            with torch.no_grad():
                m.energy_scale.fill_(0.1)
    inputs = _two_graph_inputs(torch.float64)

    gm = trace_and_compile_force(
        model,
        (
            inputs["pos"],
            inputs["A"],
            inputs["batch"],
            inputs["edge_src"],
            inputs["edge_dst"],
            inputs["shifts"],
            inputs["cell"],
        ),
        training=True,
        compute_fn=make_force_compute_fn(model, training=True),
        do_compile=False,
    )
    energy, force = gm(
        inputs["pos"],
        inputs["A"],
        inputs["batch"],
        inputs["edge_src"],
        inputs["edge_dst"],
        inputs["shifts"],
        inputs["cell"],
    )
    assert torch.isfinite(energy)
    assert torch.isfinite(force).all() and force.shape == inputs["pos"].shape


def test_model_latent_charge_makefx_single_graph_trace():
    """Regression: the scalar latent-source mesh long-range path is also traceable."""
    from mace_ictc.training.makefx_compile import make_force_compute_fn, trace_and_compile_force

    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0).double().train()
    for m in model.modules():
        if getattr(m, "energy_scale", None) is not None:
            with torch.no_grad():
                m.energy_scale.fill_(0.1)

    torch.manual_seed(8)
    L = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8], dtype=torch.long)
    pos = torch.rand(A.numel(), 3, dtype=torch.float64) * L
    batch = torch.zeros(A.numel(), dtype=torch.long)
    edge_src, edge_dst, shifts = _neighbor_list(pos, cell[0], r_max=4.5)

    gm = trace_and_compile_force(
        model,
        (pos, A, batch, edge_src, edge_dst, shifts, cell),
        training=True,
        compute_fn=make_force_compute_fn(model, training=True),
        do_compile=False,
    )
    energy, force = gm(pos, A, batch, edge_src, edge_dst, shifts, cell)
    assert torch.isfinite(energy)
    assert torch.isfinite(force).all() and force.shape == pos.shape


def test_model_latent_charge_makefx_multi_graph_trace():
    """Regression: multi-graph scalar latent-source mesh long-range is traceable."""
    from mace_ictc.training.makefx_compile import make_force_compute_fn, trace_and_compile_force

    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0).double().train()
    for m in model.modules():
        if getattr(m, "energy_scale", None) is not None:
            with torch.no_grad():
                m.energy_scale.fill_(0.1)
    inputs = _two_graph_inputs(torch.float64)

    gm = trace_and_compile_force(
        model,
        (
            inputs["pos"],
            inputs["A"],
            inputs["batch"],
            inputs["edge_src"],
            inputs["edge_dst"],
            inputs["shifts"],
            inputs["cell"],
        ),
        training=True,
        compute_fn=make_force_compute_fn(model, training=True),
        do_compile=False,
    )
    energy, force = gm(
        inputs["pos"],
        inputs["A"],
        inputs["batch"],
        inputs["edge_src"],
        inputs["edge_dst"],
        inputs["shifts"],
        inputs["cell"],
    )
    assert torch.isfinite(energy)
    assert torch.isfinite(force).all() and force.shape == inputs["pos"].shape


def test_model_combined_long_range_dispersion_makefx_multi_graph_trace():
    from mace_ictc.models.dispersion import dispersion_neighbor_list
    from mace_ictc.training.makefx_compile import trace_and_compile_force

    torch.set_default_dtype(torch.float64)
    inputs = _two_graph_inputs(torch.float64)
    disp_src, disp_dst, disp_shift = dispersion_neighbor_list(
        inputs["pos"],
        inputs["batch"],
        inputs["cell"],
        cutoff=10.0,
        pbc=True,
    )

    for max_multipole_l in (0, 2):
        model = _build_model(max_multipole_l=max_multipole_l, dispersion=True).double().train()
        for m in model.modules():
            if getattr(m, "energy_scale", None) is not None:
                with torch.no_grad():
                    m.energy_scale.fill_(0.1)

        def compute_fn(
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            shifts,
            cell,
            dispersion_edge_src,
            dispersion_edge_dst,
            dispersion_edge_shifts,
        ):
            p = pos.detach().requires_grad_(True)
            e_atom = model(
                p,
                A,
                batch,
                edge_src,
                edge_dst,
                shifts,
                cell,
                dispersion_edge_src=dispersion_edge_src,
                dispersion_edge_dst=dispersion_edge_dst,
                dispersion_edge_shifts=dispersion_edge_shifts,
            )
            if isinstance(e_atom, tuple):
                e_atom = e_atom[0]
            grad = torch.autograd.grad(e_atom.sum(), p, create_graph=True)[0]
            return e_atom.sum(), -grad

        example_inputs = (
            inputs["pos"],
            inputs["A"],
            inputs["batch"],
            inputs["edge_src"],
            inputs["edge_dst"],
            inputs["shifts"],
            inputs["cell"],
            disp_src,
            disp_dst,
            disp_shift,
        )
        gm = trace_and_compile_force(model, example_inputs, training=True, compute_fn=compute_fn, do_compile=False)
        energy, force = gm(*example_inputs)
        assert torch.isfinite(energy)
        assert torch.isfinite(force).all() and force.shape == inputs["pos"].shape


def test_export_reciprocal_source_equivariant_layout():
    """Export mode (return_reciprocal_source=True): the model emits a packed [q|mu|Q] source of
    width S*(1+3+9)=13S matching mff_reciprocal_solver's narrow/reshape decode, and the source
    transforms equivariantly (q invariant, mu->R mu, Q->R Q R^T) so the C++ reciprocal energy is
    rotation-invariant. Validates the Python-export <-> C++-solver contract."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().eval()
    assert model.long_range_exports_reciprocal_source, "multipole model must export reciprocal_source"
    box = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * box).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6])
    n = A.numel()
    s = model.multipole_readout.source_channels
    torch.manual_seed(3)
    pos = torch.rand(n, 3, dtype=torch.float64) * box
    batch = torch.zeros(n, dtype=torch.long)
    es, ed, sh = _neighbor_list(pos, cell[0], r_max=4.5)

    def emit(p, c):
        _out, rs = model(p, A, batch, es, ed, sh, c, return_reciprocal_source=True)
        q = rs[:, :s]
        mu = rs[:, s:4 * s].reshape(n, s, 3)            # C++ decode: narrow(C,3C).reshape(C,3)
        quad = rs[:, 4 * s:13 * s].reshape(n, s, 3, 3)  # narrow(4C,9C).reshape(C,3,3)
        return rs, q, mu, quad

    rs, q, mu, quad = emit(pos, cell)
    assert rs.shape == (n, s * 13), rs.shape
    assert torch.isfinite(rs).all()

    R = _random_rotation(torch.float64)
    _, q_r, mu_r, quad_r = emit(pos @ R.T, (cell[0] @ R.T).reshape(1, 3, 3))
    assert torch.allclose(q, q_r, atol=1e-8), "monopole source not invariant"
    assert torch.allclose(mu @ R.T, mu_r, atol=1e-8), (mu @ R.T - mu_r).abs().max()
    assert torch.allclose(
        torch.einsum("ij,nsjk,lk->nsil", R, quad, R), quad_r, atol=1e-8
    ), "quadrupole source not equivariant"


def test_export_torchscript_core_multipole():
    """The TorchScript export core (_TorchScriptEdgeVecCore, export_reciprocal_source=True) emits the
    6-tuple wire format the C++ engine reads: (atom_energy, global_phys[M,22], atom_phys[N,31],
    global_mask[5], atom_mask[5], reciprocal_source[N,13S]) with reciprocal_source at index 5. The 4
    physical-tensor slots are correct-width zeros (MACE-ICTC has no physical heads) so the engine's
    width checks pass. Also asserts the deploy metadata the .json writer reads is set on the model,
    and that the core jit.traces + reloads to the identical 6-tuple."""
    torch.set_default_dtype(torch.float64)
    from mace_ictc.interfaces.lammps_mliap import _TorchScriptEdgeVecCore

    model = _build_model(max_multipole_l=2).double().eval()
    s = model.multipole_readout.source_channels
    # deploy metadata the export .json writer reads (export_libtorch_core meta dict)
    assert model.long_range_exports_reciprocal_source is True
    assert model.long_range_max_multipole_l == 2
    assert model.long_range_runtime_source_channels == s
    assert model.long_range_runtime_backend == "mesh_fft"
    assert model.long_range_runtime_source_kind == "latent_multipole"
    assert model.long_range_mesh_fft_full_ewald is True

    box = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * box).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6])
    n = A.numel()
    torch.manual_seed(5)
    pos = torch.rand(n, 3, dtype=torch.float64) * box
    batch = torch.zeros(n, dtype=torch.long)
    es, ed, sh = _neighbor_list(pos, cell[0], r_max=4.5)
    edge_vec = pos[ed] - pos[es] + sh.to(pos.dtype) @ cell[0]
    ext = torch.empty(0, dtype=torch.float64)
    disp_args = (es, ed, sh, edge_vec)

    core = _TorchScriptEdgeVecCore(model, export_reciprocal_source=True).eval()
    out = core(pos, A, batch, es, ed, sh, cell, edge_vec, *disp_args, ext)
    assert isinstance(out, tuple) and len(out) == 6, f"expected 6-tuple, got len {len(out) if isinstance(out, tuple) else '-'}"
    atom_e, gphys, aphys, gmask, amask, rs = out
    assert atom_e.shape[0] == n
    assert rs.shape == (n, 13 * s), rs.shape  # reciprocal_source MUST be at index 5
    # physical-tensor slots: correct-width zeros (engine checks width only when numel>0)
    assert gphys.shape[-1] == 22 and aphys.shape[-1] == 31, (gphys.shape, aphys.shape)
    assert gmask.numel() == 5 and amask.numel() == 5
    assert float(gphys.abs().sum()) == 0.0 and float(aphys.abs().sum()) == 0.0, "phys slots must be zero (no heads)"

    # traces + reloads to the identical 6-tuple (this is what torch.jit.save serializes for LibTorch)
    trace_inputs = (pos, A, batch, es, ed, sh, cell, edge_vec, *disp_args, ext)
    traced = torch.jit.trace(core, trace_inputs, check_trace=False, strict=False)
    out2 = traced(*trace_inputs)
    assert len(out2) == 6 and out2[5].shape == (n, 13 * s)
    assert torch.allclose(out2[5], rs, atol=1e-8), "traced reciprocal_source diverged"
    assert torch.allclose(out2[0], atom_e, atol=1e-8), "traced energy diverged"


if __name__ == "__main__":
    test_forward_multipole_rotation_invariance_and_forces()
    print("OK: forward_multipole rotation-invariance + forces + translation-invariance")
    test_model_multipole_gating()
    print("OK: model multipole gating (off -> None / on -> readout)")
    test_model_multipole_forward_smoke()
    print("OK: full-model forward smoke (energy + forces + rotation-invariance, multipole ON)")
    test_model_multipole_gentle_start_warm_safe()
    print("OK: multipole gentle-start warm-safe (long-range contribution ~gate**2 at init)")
    l0, l1 = test_model_multipole_training_smoke()
    print(f"OK: training smoke (loss {l0:.3e} -> {l1:.3e}, multipole readout gets gradient)")
    test_export_reciprocal_source_equivariant_layout()
    print("OK: export reciprocal_source packed [q|mu|Q] layout + equivariance (C++ contract)")
    test_export_torchscript_core_multipole()
    print("OK: TorchScript export core emits the 6-tuple wire format (rs@5, phys zeros) + traces")
