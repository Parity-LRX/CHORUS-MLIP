"""Round-trip + rotation-equivariance test for the multipole readout head.

Strategy: build known Cartesian multipoles (charge / dipole / quadrupole), encode
them into ICTC irreps blocks with the existing embedder, assemble a full-SO(3)
feature, run ``MultipoleReadout`` (channel mixers pinned to identity), and check
(a) round-trip recovery and (b) that under a Cartesian rotation R the outputs
transform as monopole->invariant, dipole->R mu, quadrupole->R Q R^T.
"""

from __future__ import annotations

import torch

from mace_ictc.models.multipole_readout import MultipoleReadout
from mace_ictc.utils.tensor_utils import build_physical_tensor_label_blocks


def _random_rotation(dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))  # fix QR sign ambiguity
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


def _blocks(tensor, rank, dtype):
    return build_physical_tensor_label_blocks(
        tensor, rank=rank, lmax=rank, include_trace_chain=False,
        representation="ictd", device=torch.device("cpu"),
    )


def _assemble(charge, mu, quad, *, lmax, dtype):
    """charge [N], mu [N,3], quad [N,3,3] -> full-SO(3) feature [N, (lmax+1)**2] (C=1)."""
    n = charge.shape[0]
    b1 = _blocks(mu, 1, dtype)[1].reshape(n, 1, 3)
    b2 = _blocks(quad, 2, dtype)[2].reshape(n, 1, 5)
    b0 = charge.reshape(n, 1, 1)
    blocks = {0: b0, 1: b1, 2: b2}
    return torch.cat([blocks[l].reshape(n, 1 * (2 * l + 1)) for l in range(lmax + 1)], dim=-1)


def _pin_identity(head: MultipoleReadout) -> None:
    with torch.no_grad():
        for lin in head.mix:
            lin.weight.copy_(torch.ones_like(lin.weight))
        head.source_scale.fill_(1.0)  # disable the gentle-start gate for the exact round-trip


def test_multipole_readout_roundtrip_and_equivariance():
    dtype = torch.float64
    torch.manual_seed(0)
    n = 6
    charge = torch.randn(n, dtype=dtype)
    mu = torch.randn(n, 3, dtype=dtype)
    qsym = torch.randn(n, 3, 3, dtype=dtype)
    qsym = 0.5 * (qsym + qsym.transpose(-1, -2))
    eye = torch.eye(3, dtype=dtype)
    quad = qsym - (qsym.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0).reshape(n, 1, 1) * eye  # traceless

    head = MultipoleReadout(channels=1, lmax=2, max_multipole_l=2, source_channels=1).to(dtype)
    _pin_identity(head)

    feat = _assemble(charge, mu, quad, lmax=2, dtype=dtype)
    m, d, q = head(feat)
    m, d, q = m.reshape(n), d.reshape(n, 3), q.reshape(n, 3, 3)

    # (a) round-trip: decode(encode(.)) == identity (convention-correct)
    assert torch.allclose(m, charge, atol=1e-9), (m - charge).abs().max()
    assert torch.allclose(d, mu, atol=1e-9), (d - mu).abs().max()
    assert torch.allclose(q, quad, atol=1e-9), (q - quad).abs().max()

    # (b) equivariance under a Cartesian rotation R
    R = _random_rotation(dtype)
    mu_r = mu @ R.T
    quad_r = torch.einsum("ij,njk,lk->nil", R, quad, R)
    feat_r = _assemble(charge, mu_r, quad_r, lmax=2, dtype=dtype)
    m2, d2, q2 = head(feat_r)
    m2, d2, q2 = m2.reshape(n), d2.reshape(n, 3), q2.reshape(n, 3, 3)

    assert torch.allclose(m2, charge, atol=1e-9), "monopole not invariant"
    assert torch.allclose(d2, mu_r, atol=1e-9), (d2 - mu_r).abs().max()
    assert torch.allclose(q2, quad_r, atol=1e-9), (q2 - quad_r).abs().max()


def test_multipole_readout_gentle_start_gate():
    """The learnable source_scale gate makes a freshly-built multipole head start gentle (every
    q/mu/Q source scaled by the small init -> reciprocal energy, quadratic in the sources, starts at
    ~gate**2 of full strength) yet trainable (gate + channel mixers receive gradient, unlike a
    zero-source start which is a zero-gradient fixed point). The gate multiplies the SAME sources
    that feed both the in-model energy and the exported pack -> train/deploy stay consistent."""
    dtype = torch.float64
    torch.manual_seed(0)
    n, c = 5, 8
    head = MultipoleReadout(
        channels=c, lmax=2, max_multipole_l=2, source_channels=1, source_scale_init=0.1
    ).to(dtype)
    # default gate is the small gentle-start value, not 1 (warm-start safety)
    assert abs(float(head.source_scale) - 0.1) < 1e-6

    feat = torch.randn(n, c * 9, dtype=dtype)
    gate_val = float(head.source_scale)
    m_gentle, d_gentle, q_gentle = head(feat)
    with torch.no_grad():
        head.source_scale.fill_(1.0)
    m_full, d_full, q_full = head(feat)
    # the gate scales every source linearly by exactly its value
    assert torch.allclose(m_gentle, gate_val * m_full, atol=1e-12)
    assert torch.allclose(d_gentle, gate_val * d_full, atol=1e-12)
    assert torch.allclose(q_gentle, gate_val * q_full, atol=1e-12)

    # trainable: both the gate and the channel mixers get a non-zero gradient
    head.source_scale.data.fill_(0.1)
    head.zero_grad(set_to_none=True)
    m, d, q = head(feat)
    ((m ** 2).sum() + (d ** 2).sum() + (q ** 2).sum()).backward()
    assert head.source_scale.grad is not None and head.source_scale.grad.abs() > 0
    assert any(lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0 for lin in head.mix)


if __name__ == "__main__":
    test_multipole_readout_roundtrip_and_equivariance()
    print("OK: multipole readout round-trip + equivariance")
    test_multipole_readout_gentle_start_gate()
    print("OK: multipole readout gentle-start gate (gentle + trainable + consistent)")
