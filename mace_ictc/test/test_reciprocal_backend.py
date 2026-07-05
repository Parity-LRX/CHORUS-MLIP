"""Task 1 regression: the shared ReciprocalBackend's scalar route reproduces the existing
MeshLongRangeKernel3D reciprocal potential exactly (same grid ops, just factored out)."""
from __future__ import annotations

from pathlib import Path

import torch

from mace_ictc.models.long_range import MeshLongRangeKernel3D
from mace_ictc.models.reciprocal_backend import ReciprocalBackend


def _kernel_spectral(k, cell_b):
    """Replicate apply_green_kernel_batched's per-k scalar spectral weight for cell_b [1,3,3]."""
    k_norms, volume = k.build_k_norms_batched(cell_b, dtype=torch.float64)  # [1,M,M,M]
    sw = k.green_kernel(k_norms) / volume.view(-1, 1, 1, 1)
    if k.full_ewald:
        rc = k._estimate_real_cutoff_batched(cell_b.to(torch.float64))
        alpha = k._estimate_ewald_alpha(rc)
        sw = sw * torch.exp(-(k_norms.square()) / (4.0 * alpha.view(-1, 1, 1, 1).square()))
    win = k._build_assignment_window(device=cell_b.device, dtype=torch.float64)
    sw = sw * torch.reciprocal(win.clamp_min(k.assignment_window_floor).square()).unsqueeze(0)
    if k.full_ewald or (not k.include_k0):
        sw = torch.where(k_norms > k.k_norm_floor, sw, torch.zeros_like(sw))
    return sw[0]  # [M,M,M]


def test_scalar_backend_matches_existing():
    torch.set_default_dtype(torch.float64)
    for full_ewald in (False, True):
        for assignment in ("cic", "pcs"):
            cfg = dict(mesh_size=16, boundary="periodic", slab_padding_factor=2, assignment=assignment)
            k = MeshLongRangeKernel3D(full_ewald=full_ewald, include_k0=False, **cfg)
            be = ReciprocalBackend(include_k0=False, k_norm_floor=k.k_norm_floor, **cfg)

            N, L = 8, 8.0
            g = torch.Generator().manual_seed(0)
            pos = torch.rand(N, 3, generator=g, dtype=torch.float64) * L
            cell = torch.eye(3, dtype=torch.float64) * L
            source = torch.randn(N, 1, generator=g, dtype=torch.float64)

            frac = be.frac(pos, cell)
            mesh = be.spread(frac, source)                                  # [M,M,M,1]
            # existing kernel path (batched, B=1)
            pot_b, *_ = k.apply_green_kernel_batched(mesh.unsqueeze(0), cell.unsqueeze(0))
            recip_existing = be.gather(frac, pot_b[0])
            # backend scalar route with the kernel's spectral
            recip_backend = be.scalar_potential(frac, source, _kernel_spectral(k, cell.unsqueeze(0)))

            d = (recip_existing - recip_backend).abs().max().item()
            assert d < 1e-12, f"full_ewald={full_ewald} assignment={assignment}: mismatch {d:.2e}"

    # k_grid sanity + 3-channel (MBD dipole) spread/gather adjoint
    be = ReciprocalBackend(mesh_size=16, assignment="cic")
    cell = torch.eye(3, dtype=torch.float64) * 8.0
    k_cart, k_norm, vol = be.k_grid(cell, dtype=torch.float64)
    assert abs(vol.item() - 8.0 ** 3) < 1e-9 and k_cart.shape == (16 ** 3, 3)
    pos = torch.rand(5, 3, dtype=torch.float64) * 8.0
    frac = be.frac(pos, cell)
    assert be.gather(frac, be.spread(frac, torch.randn(5, 3, dtype=torch.float64))).shape == (5, 3)


def test_mesh_potential_and_multipole_monopole_paths_match_for_cic():
    """CIC non-Ewald potential route must use the same assignment deconvolution as |S(k)|^2."""
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(12)
    n, box = 10, 7.5
    pos = torch.rand(n, 3, generator=g, dtype=torch.float64) * box
    batch = torch.zeros(n, dtype=torch.long)
    cell = (torch.eye(3, dtype=torch.float64) * box).unsqueeze(0)
    source = torch.randn(n, 1, generator=g, dtype=torch.float64)
    source = source - source.mean(dim=0, keepdim=True)

    kernel = MeshLongRangeKernel3D(
        mesh_size=16,
        assignment="cic",
        full_ewald=False,
        include_k0=False,
        energy_partition="potential",
    )
    e_potential = kernel(pos, batch, cell, source).sum()
    e_multipole = kernel.multipole_energy(pos, batch, cell, source, None, None).sum()
    assert torch.allclose(e_potential, e_multipole, atol=1e-10, rtol=1e-10), (e_potential, e_multipole)


def test_lammps_reciprocal_paths_deconvolve_assignment_window():
    """Deployment scalar and CPU multipole PME paths must match the training assignment window."""
    repo = Path(__file__).resolve().parents[2]
    source = repo / "lammps_user_mfftorch/src/USER-MFFTORCH/mff_reciprocal_solver.cpp"
    text = source.read_text()

    scalar = text.split("MFFReciprocalSolver::build_local_spectral_weights", 1)[1].split(
        "MFFReciprocalSolver::multipole_reciprocal_energy", 1
    )[0]
    assert "assignment_order = (assignment_ == 1) ? 4 : 2" in scalar
    assert "torch::sinc(freq / static_cast<double>(mesh_size_)).pow(assignment_order)" in scalar
    assert "torch::sinc(freq_y / static_cast<double>(mesh_size_)).pow(assignment_order)" in scalar
    assert "cached_local_spectral_weights_cpu_ = (4.0 * M_PI) / (safe * safe) * wdeconv" in scalar
    assert "axis_assignment_weights" in text
    assert "build_assignment_idx_weights(frac, mesh_size_, assignment_, pbc, flat_idx, weights)" in text

    multipole = text.split("MFFReciprocalSolver::multipole_reciprocal_energy", 1)[1].split(
        "MFFReciprocalSolver::multipole_reciprocal_forces_explicit", 1
    )[0]
    assert "assignment_order = (assignment_ == 1) ? 4 : 2" in multipole
    assert "torch::sinc(freq / static_cast<double>(mesh_size_)).pow(assignment_order)" in multipole


if __name__ == "__main__":
    test_scalar_backend_matches_existing()
    test_mesh_potential_and_multipole_monopole_paths_match_for_cic()
    test_lammps_reciprocal_paths_deconvolve_assignment_window()
    print("OK: Task1 ReciprocalBackend scalar route == existing reciprocal (cic+pcs, bare+ewald);"
          " k_grid + 3-channel spread/gather OK")
