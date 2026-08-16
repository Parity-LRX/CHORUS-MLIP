"""Portability gates for the frozen arbitrary-L ICTC harmonic basis."""

from __future__ import annotations

import math

import torch
from e3nn import o3

from chorus.mace_basis import orthogonal_Q_blocks
from chorus.models.ictd_irreps import (
    _CANONICAL_HARMONIC_BASIS_SHA256,
    _deterministic_harmonic_basis_cpu_f64,
    _gram_gaussian,
    _harmonic_basis_cpu_f64,
    build_cg_tensor,
    direction_harmonics_all,
)
from chorus.models.ictd_fast import _build_laplacian_matrix


def test_frozen_harmonic_basis_assets_cover_o2u1_high_l() -> None:
    assert set(_CANONICAL_HARMONIC_BASIS_SHA256) == set(range(13))
    for l in range(13):
        basis = _harmonic_basis_cpu_f64(l)
        assert basis.dtype == torch.float64
        assert basis.shape == ((l + 1) * (l + 2) // 2, 2 * l + 1)


def test_above_frozen_range_uses_deterministic_harmonic_nullspace() -> None:
    l = 13
    basis = _deterministic_harmonic_basis_cpu_f64(l)
    laplacian = _build_laplacian_matrix(l, dtype=torch.float64)
    gram = _gram_gaussian(l)
    assert float((laplacian @ basis).abs().max()) < 1e-10
    identity = torch.eye(2 * l + 1, dtype=torch.float64)
    assert float((basis.T @ gram @ basis - identity).abs().max()) < 1e-10


def test_high_l_basis_and_cg_match_e3nn_convention() -> None:
    # e3nn 0.5.x implements spherical harmonics through l=11.  Exercise that
    # entire independent reference range; l=12 is still covered by the frozen
    # asset/integrity gate above without claiming an unavailable e3nn oracle.
    lmax = 11
    generator = torch.Generator().manual_seed(20260816)
    directions = torch.randn(257, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    ictd = direction_harmonics_all(directions, lmax)
    e3nn = o3.spherical_harmonics(
        o3.Irreps.spherical_harmonics(lmax),
        directions,
        normalize=True,
        normalization="component",
    )
    q_blocks = orthogonal_Q_blocks(lmax)
    offset = 0
    for l in range(lmax + 1):
        width = 2 * l + 1
        error = (ictd[l] @ q_blocks[l] - e3nn[:, offset : offset + width]).abs().max()
        assert float(error) < 1e-9, f"l={l}: harmonic/Q mismatch {float(error):.3e}"
        offset += width

    # Boundary/high-degree paths include the exact L4 failure previously seen
    # on Torch 2.7, plus representative L6/L8 couplings.
    for l1, l2, l3 in (
        (1, 3, 4),
        (2, 4, 6),
        (3, 5, 8),
        (4, 4, 8),
        (7, 1, 8),
        (2, 8, 10),
        (5, 6, 11),
        (10, 1, 11),
    ):
        coupling = build_cg_tensor(l1, l2, l3).to(torch.float64)
        folded = torch.einsum(
            "ax,by,cz,abc->xyz",
            q_blocks[l1],
            q_blocks[l2],
            q_blocks[l3],
            coupling,
        )
        target = o3.wigner_3j(l1, l2, l3, dtype=torch.float64) * math.sqrt(2 * l3 + 1)
        scale = (folded * target).sum() / (folded * folded).sum().clamp_min(1e-30)
        residual = (target - scale * folded).abs().max()
        assert float(residual) < 1e-9, (
            f"path {(l1, l2, l3)}: ICTC/e3nn CG mismatch {float(residual):.3e}"
        )
