"""Task 2: the Ewald dipole-dipole field operator (T.mu via the shared ReciprocalBackend).
Correctness = alpha-independence (the SR+LR+self split is exact) + the isolated-pair limit
reproducing the analytic point-dipole tensor (3 rr - I)/r^3."""
from __future__ import annotations

import torch

from mace_ictc.models.dispersion import dispersion_neighbor_list
from mace_ictc.models.mbd import dipole_field
from mace_ictc.models.reciprocal_backend import ReciprocalBackend


def _field(backend, pos, mu, cell, alpha, cutoff):
    batch = torch.zeros(pos.size(0), dtype=torch.long)
    src, dst, sh = dispersion_neighbor_list(pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True)
    return dipole_field(backend, pos, mu, cell, alpha=alpha, src=src, dst=dst, shifts=sh)


def test_alpha_independence():
    """The total periodic Ewald dipole field must not depend on the splitting parameter alpha."""
    torch.set_default_dtype(torch.float64)
    be = ReciprocalBackend(mesh_size=48, assignment="pcs", boundary="periodic")
    N, L = 6, 8.0
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(N, 3, generator=g, dtype=torch.float64) * L
    mu = torch.randn(N, 3, generator=g, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * L
    cutoff = 0.5 * L - 1e-6  # < L: no self-image pairs; self handled by the reciprocal self term

    e1 = _field(be, pos, mu, cell, alpha=0.9, cutoff=cutoff)
    e2 = _field(be, pos, mu, cell, alpha=1.4, cutoff=cutoff)
    d = (e1 - e2).abs().max().item()
    scale = e1.abs().max().item()
    # converges cleanly with mesh (2.5e-2@24 -> 1.1e-3@48 -> 3.5e-4@64): an exact split, not a bug
    assert d / scale < 3e-3, f"alpha-dependent (split wrong): rel {d/scale:.2e}"


def test_isolated_pair_matches_point_dipole():
    """Huge box, one pair: E_i ~= T_full(r) mu_j with T_ab=(3 r^_a r^_b - d_ab)/r^3 (LR,self -> 0)."""
    torch.set_default_dtype(torch.float64)
    be = ReciprocalBackend(mesh_size=32, assignment="pcs", boundary="periodic")
    L = 60.0
    pos = torch.tensor([[0.0, 0, 0], [3.2, 0.7, -0.4]], dtype=torch.float64) + L / 2
    mu = torch.tensor([[0.3, -0.5, 0.8], [0.0, 0.0, 0.0]], dtype=torch.float64)  # only mu_1 nonzero
    cell = torch.eye(3, dtype=torch.float64) * L
    e = _field(be, pos, mu, cell, alpha=0.06, cutoff=12.0)

    rvec = pos[1] - pos[0]
    r = rvec.norm()
    rhat = rvec / r
    T = (3.0 * torch.outer(rhat, rhat) - torch.eye(3, dtype=torch.float64)) / r ** 3
    e2_analytic = T @ mu[0]
    d = (e[1] - e2_analytic).abs().max().item()
    assert d / e2_analytic.abs().max().item() < 5e-3, (
        f"field at 2 from dipole 1 wrong: got {e[1].tolist()} vs {e2_analytic.tolist()}"
    )


def test_lanczos_quadform_exact():
    """Lanczos quadrature for z^T sqrt(A) z is exact at full Krylov depth (vs dense sqrt(A))."""
    torch.set_default_dtype(torch.float64)
    from mace_ictc.models.mbd import lanczos_sqrt_quadform

    n = 12
    g = torch.Generator().manual_seed(3)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    A = M @ M.T + n * torch.eye(n, dtype=torch.float64)  # SPD
    z = torch.randn(n, generator=g, dtype=torch.float64)
    lam, U = torch.linalg.eigh(A)
    sqrtA = U @ torch.diag(lam.sqrt()) @ U.T
    exact = z @ (sqrtA @ z)
    approx = lanczos_sqrt_quadform(lambda v: A @ v, z, n)
    assert (approx - exact).abs() / exact.abs() < 1e-9, f"{approx} vs {exact}"


def test_slq_trace_sqrt_matches_dense():
    """The SLQ primitive Tr[sqrt C] (no eigendecomposition of C) reproduces the dense Sum sqrt(lambda).
    This validates the spectral-solver machinery; E_MBD = 1/2 Tr[sqrt C] - 3/2 sum w is a small
    difference of large numbers, so its RELATIVE error is a deploy-time probe-count / control-variate
    concern (and the diagonal self-image / rsSCS range-separation physics is deferred to task 5)."""
    torch.set_default_dtype(torch.float64)
    from mace_ictc.models.mbd import coupled_dipole_matvec, make_probes, slq_trace_sqrt

    be = ReciprocalBackend(mesh_size=32, assignment="pcs")
    sp = 2.6
    L = 2 * sp
    coords = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    pos = torch.tensor(coords, dtype=torch.float64) * sp + 0.13
    N = pos.size(0)
    n = 3 * N
    cell = torch.eye(3, dtype=torch.float64) * L
    batch = torch.zeros(N, dtype=torch.long)
    src, dst, sh = dispersion_neighbor_list(pos, batch, cell.reshape(1, 3, 3), L / 2 - 1e-6, pbc=True)

    def field_fn(mu):
        return dipole_field(be, pos, mu, cell, alpha=1.0, src=src, dst=dst, shifts=sh)

    g = torch.Generator().manual_seed(1)
    omega = 0.9 + 0.3 * torch.rand(N, generator=g, dtype=torch.float64)
    alpha_pol = torch.full((N,), 0.3, dtype=torch.float64)

    def mv(v):
        return coupled_dipole_matvec(v.view(N, 3), omega, alpha_pol, field_fn).reshape(-1)

    # dense reference
    C = torch.stack([mv(torch.eye(n, dtype=torch.float64)[j]) for j in range(n)], dim=1)
    lam = torch.linalg.eigvalsh(0.5 * (C + C.T))
    tr_dense = lam.clamp_min(0).sqrt().sum()
    assert (lam > 0).all(), "C not PD"

    probes = make_probes(n, 800, device=omega.device, dtype=omega.dtype, seed=0)
    tr_slq = slq_trace_sqrt(mv, n, probes, steps=n)
    rel = (tr_slq - tr_dense).abs() / tr_dense
    assert rel < 1.5e-2, f"Tr[sqrt C] SLQ {tr_slq.item():.5f} vs dense {tr_dense.item():.5f} rel {rel.item():.2e}"


def test_chebyshev_trace_sqrt_matches_dense():
    """Deployment path: Chebyshev Tr[sqrt C] (pure matvec + fixed-degree polynomial, NO eigensolve)
    reproduces dense Sum sqrt(lambda); spectral bounds from matvec-only power iteration. Differentiable."""
    torch.set_default_dtype(torch.float64)
    from mace_ictc.models.mbd import (
        chebyshev_trace_sqrt,
        coupled_dipole_matvec,
        make_probes,
        power_iter_lambda_max,
    )

    be = ReciprocalBackend(mesh_size=32, assignment="pcs")
    sp = 2.6
    L = 2 * sp
    coords = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    pos = torch.tensor(coords, dtype=torch.float64) * sp + 0.13
    N = pos.size(0)
    n = 3 * N
    cell = torch.eye(3, dtype=torch.float64) * L
    batch = torch.zeros(N, dtype=torch.long)
    src, dst, sh = dispersion_neighbor_list(pos, batch, cell.reshape(1, 3, 3), L / 2 - 1e-6, pbc=True)

    def field_fn(mu):
        return dipole_field(be, pos, mu, cell, alpha=1.0, src=src, dst=dst, shifts=sh)

    g = torch.Generator().manual_seed(1)
    omega = (0.9 + 0.3 * torch.rand(N, generator=g, dtype=torch.float64)).requires_grad_(True)
    alpha = torch.full((N,), 0.3, dtype=torch.float64)

    def mv(v):
        return coupled_dipole_matvec(v.view(N, 3), omega, alpha, field_fn).reshape(-1)

    with torch.no_grad():
        C = torch.stack([mv(torch.eye(n, dtype=torch.float64)[j]) for j in range(n)], dim=1)
        lam = torch.linalg.eigvalsh(0.5 * (C + C.T))
        tr_dense = lam.clamp_min(0).sqrt().sum()
        lmax = float(power_iter_lambda_max(mv, n, steps=30, device="cpu", dtype=torch.float64, seed=1)) * 1.05
        gap = float(power_iter_lambda_max(lambda v: lmax * v - mv(v), n, steps=30, device="cpu", dtype=torch.float64, seed=2))
        lmin = max((lmax - gap) * 0.95, 1e-6)
    # bounds bracket the spectrum
    assert lmin <= lam.min() and lam.max() <= lmax, f"bounds [{lmin:.3f},{lmax:.3f}] vs spectrum [{lam.min():.3f},{lam.max():.3f}]"

    probes = make_probes(n, 800, device="cpu", dtype=torch.float64, seed=0)
    tr_cheb = chebyshev_trace_sqrt(mv, n, probes, degree=30, lmin=lmin, lmax=lmax)
    rel = (tr_cheb - tr_dense).abs() / tr_dense
    assert rel < 1.5e-2, f"Chebyshev Tr {tr_cheb.item():.5f} vs dense {tr_dense.item():.5f} rel {rel.item():.2e}"
    (go,) = torch.autograd.grad(tr_cheb, omega)
    assert torch.isfinite(go).all() and go.abs().sum() > 0


def test_slq_energy_conservative_gradient():
    """The fixed-probe SLQ energy is a deterministic surrogate -> autograd gradient is exact
    (conservative forces). Finite-difference confirms. The custom-backward low-mem variant runs and
    is finite (a different unbiased estimator -- documented as non-exactly-conservative)."""
    torch.set_default_dtype(torch.float64)
    from mace_ictc.models.mbd import mbd_energy_slq, mbd_energy_slq_lowmem

    be = ReciprocalBackend(mesh_size=20, assignment="pcs")
    sp = 2.6
    L = 2 * sp
    coords = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    pos = torch.tensor(coords, dtype=torch.float64) * sp + 0.13
    N = pos.size(0)
    cell = torch.eye(3, dtype=torch.float64) * L
    batch = torch.zeros(N, dtype=torch.long)
    src, dst, sh = dispersion_neighbor_list(pos, batch, cell.reshape(1, 3, 3), L / 2 - 1e-6, pbc=True)

    def field_fn(mu):
        return dipole_field(be, pos, mu, cell, alpha=1.0, src=src, dst=dst, shifts=sh)

    g = torch.Generator().manual_seed(1)
    omega = (0.9 + 0.3 * torch.rand(N, generator=g, dtype=torch.float64)).requires_grad_(True)
    alpha = torch.full((N,), 0.3, dtype=torch.float64)
    E = mbd_energy_slq(omega, alpha, field_fn, num_probes=48, lanczos_steps=18, seed=0)
    (ga,) = torch.autograd.grad(E, omega)
    eps = 1e-6
    op = omega.detach().clone(); op[0] += eps
    om = omega.detach().clone(); om[0] -= eps
    fd = (mbd_energy_slq(op, alpha, field_fn, num_probes=48, lanczos_steps=18, seed=0)
          - mbd_energy_slq(om, alpha, field_fn, num_probes=48, lanczos_steps=18, seed=0)) / (2 * eps)
    assert (ga[0] - fd).abs() / fd.abs() < 1e-6, f"autograd not the exact surrogate gradient: {ga[0]} vs {fd}"

    om2 = omega.detach().clone().requires_grad_(True)
    E2 = mbd_energy_slq_lowmem(om2, alpha, field_fn, num_probes=48, lanczos_steps=18, seed=0)
    (g2,) = torch.autograd.grad(E2, om2)
    assert torch.isfinite(g2).all() and (E2 - E.detach()).abs() < 1e-10  # forward identical, grad finite


if __name__ == "__main__":
    test_alpha_independence()
    print("OK: dipole-field Ewald is alpha-independent (SR+LR+self split exact)")
    test_isolated_pair_matches_point_dipole()
    print("OK: isolated-pair field == analytic (3 rr - I)/r^3 point-dipole tensor")
    test_lanczos_quadform_exact()
    print("OK: Lanczos quadrature z^T sqrt(A) z exact at full depth")
    test_slq_trace_sqrt_matches_dense()
    print("OK: SLQ Tr[sqrt C] (no eigendecomp) == dense Sum sqrt(lambda)")
    test_chebyshev_trace_sqrt_matches_dense()
    print("OK: Chebyshev Tr[sqrt C] (no eigensolve, AOTI-path) == dense; bounds bracket spectrum; differentiable")
    test_slq_energy_conservative_gradient()
    print("OK: SLQ energy autograd = exact surrogate gradient (conservative); low-mem backward runs")
