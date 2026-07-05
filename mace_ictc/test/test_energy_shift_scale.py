"""Check MACE-style per-atom interaction energy scale/shift.

Run:
    python -m mace_ictc.test.test_energy_shift_scale
"""

from __future__ import annotations

import torch

from mace_ictc.synthetic import build_model, make_fixed_graph, compute_energy_forces


def main() -> None:
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    device = torch.device("cpu")
    scale = 2.75
    shift = -0.125

    torch.manual_seed(123)
    base = build_model(
        channels=8,
        lmax=2,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-bridge-u",
        dtype=dtype,
        device=device,
        correlation=2,
    ).eval()
    torch.manual_seed(123)
    shifted = build_model(
        channels=8,
        lmax=2,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-bridge-u",
        dtype=dtype,
        device=device,
        correlation=2,
        energy_output_scale=scale,
        energy_output_scale_enabled=True,
        energy_output_shift=shift,
        energy_output_shift_enabled=True,
    ).eval()

    graph = make_fixed_graph(
        num_nodes=18,
        avg_degree=16,
        dtype=dtype,
        device=device,
        seed=5,
    )
    e_base, f_base, e_atom_base = compute_energy_forces(base, graph, create_graph=False)
    e_scaled, f_scaled, e_atom_scaled = compute_energy_forces(shifted, graph, create_graph=False)

    expected_atom = e_atom_base * scale + shift
    expected_energy = e_base * scale + shift * e_atom_base.shape[0]
    expected_forces = f_base * scale

    d_atom = (e_atom_scaled - expected_atom).abs().max().item()
    d_energy = (e_scaled - expected_energy).abs().item()
    d_force = (f_scaled - expected_forces).abs().max().item()
    print(
        "energy scale/shift:",
        f"max|dE_atom|={d_atom:.3e}",
        f"|dE|={d_energy:.3e}",
        f"max|dF|={d_force:.3e}",
    )
    assert d_atom < 1e-12
    assert d_energy < 1e-12
    assert d_force < 1e-12
    print("PASS: scale/shift matches MACE-style per-atom interaction energy semantics.")


def test_energy_shift_scale() -> None:
    main()


if __name__ == "__main__":
    main()
