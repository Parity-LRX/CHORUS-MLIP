"""Validate ``convert_mace_to_ictd``: a real e3nn MACE and a MACE-ICTC model, after weight
conversion, must produce the SAME energy and forces (float64).

Run (from the repo root):
    python -m mace_ictc.test.test_mace_converter
    # or:  PYTHONPATH=. python3 mace_ictc/test/test_mace_converter.py
or under pytest.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from e3nn import o3

from mace.modules import ScaleShiftMACE, interaction_classes, gate_dict

from mace_ictc.models.pure_cartesian_ictd_fix import PureCartesianICTDFix
from mace_ictc.interfaces.mace_converter import convert_mace_to_ictd


ATOMIC_NUMBERS = [1, 6, 7, 8]
ATOMIC_ENERGIES = np.array([-13.6, -1029.0, -1485.0, -2042.0], dtype=float)
R_MAX = 5.0
AVG_NUM_NEIGHBORS = 20.0
CHANNELS = 16
LMAX = 2
ATOMIC_INTER_SCALE = 1.7
ATOMIC_INTER_SHIFT = -0.23


def _hidden_irreps(channels: int, hidden_lmax: int) -> o3.Irreps:
    return o3.Irreps(
        " + ".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(int(hidden_lmax) + 1))
    )


def build_mace(
    first_interaction: str = "residual",
    *,
    num_interactions: int = 2,
    max_ell: int = LMAX,
    hidden_lmax: int = LMAX,
) -> ScaleShiftMACE:
    torch.set_default_dtype(torch.float64)
    if first_interaction == "residual":
        first_cls = interaction_classes["RealAgnosticResidualInteractionBlock"]
    elif first_interaction == "nonresidual":
        first_cls = interaction_classes["RealAgnosticInteractionBlock"]
    else:
        raise ValueError(first_interaction)
    model = ScaleShiftMACE(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=int(max_ell),
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=first_cls,
        num_interactions=int(num_interactions),
        num_elements=len(ATOMIC_NUMBERS),
        hidden_irreps=_hidden_irreps(CHANNELS, int(hidden_lmax)),
        MLP_irreps=o3.Irreps(f"{CHANNELS}x0e"),
        atomic_energies=ATOMIC_ENERGIES,
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=3,
        gate=gate_dict["silu"],
        radial_type="bessel",
        radial_MLP=[64, 64, 64],
        atomic_inter_scale=ATOMIC_INTER_SCALE,
        atomic_inter_shift=ATOMIC_INTER_SHIFT,
    )
    return model.double().eval()


def build_ictd(
    product_backend: str = "ictd-bridge-u",
    *,
    use_reduced_cg: bool = False,
    num_interactions: int = 2,
    max_ell: int = LMAX,
    hidden_lmax: int = LMAX,
) -> PureCartesianICTDFix:
    torch.set_default_dtype(torch.float64)
    model = PureCartesianICTDFix(
        max_embed_radius=R_MAX,
        main_max_radius=R_MAX,
        main_number_of_basis=8,
        hidden_dim_conv=CHANNELS,
        hidden_dim_sh=CHANNELS,
        hidden_dim=CHANNELS,
        channel_in2=CHANNELS,
        embedding_dim=CHANNELS,
        max_atomvalue=10,
        atomic_numbers=ATOMIC_NUMBERS,
        num_interaction=int(num_interactions),
        function_type_main="bessel",
        lmax=int(hidden_lmax),
        ictd_fix_edge_lmax=int(max_ell),
        ictd_fix_route="baseline",
        ictd_fix_product_backend=product_backend,
        ictd_fix_use_reduced_cg=bool(use_reduced_cg),
        save_contraction_order=3,
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        angular_basis="ictd",
        internal_compute_dtype=torch.float64,
        device="cpu",
    )
    return model.double().eval()


def build_neighbor_list(positions, cell, r_max):
    """Minimal-image neighbor list with explicit shifts (single periodic box).

    row0 = sender j, row1 = receiver i; edge vec = pos[i] - pos[j] + unit_shifts @ cell.
    """
    N = positions.shape[0]
    senders, receivers, unit_shifts_l = [], [], []
    images = list(itertools.product([-1, 0, 1], repeat=3))
    for i in range(N):
        for j in range(N):
            for s in images:
                s = np.array(s, dtype=float)
                rij = positions[j] + s @ cell - positions[i]
                d = np.linalg.norm(rij)
                if 1e-8 < d < r_max:
                    senders.append(j)
                    receivers.append(i)
                    unit_shifts_l.append(s)
    edge_index = np.array([senders, receivers], dtype=np.int64)
    unit_shifts = np.array(unit_shifts_l, dtype=float)
    shifts = unit_shifts @ cell
    return edge_index, shifts, unit_shifts


def make_graph(n_atoms=24, box=8.0, seed=7):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * box
    positions = rng.uniform(0.5, box - 0.5, size=(n_atoms, 3))
    z_idx = rng.integers(0, len(ATOMIC_NUMBERS), size=n_atoms)
    edge_index, shifts, unit_shifts = build_neighbor_list(positions, cell, R_MAX)
    return positions, cell, z_idx, edge_index, shifts, unit_shifts


def run_mace(model, positions, cell, z_idx, edge_index, shifts, unit_shifts):
    N = positions.shape[0]
    one_hot = np.zeros((N, len(ATOMIC_NUMBERS)), dtype=float)
    one_hot[np.arange(N), z_idx] = 1.0
    data = {
        "positions": torch.tensor(positions, dtype=torch.float64),
        "node_attrs": torch.tensor(one_hot, dtype=torch.float64),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "shifts": torch.tensor(shifts, dtype=torch.float64),
        "unit_shifts": torch.tensor(unit_shifts, dtype=torch.float64),
        "cell": torch.tensor(cell, dtype=torch.float64),
        "batch": torch.zeros(N, dtype=torch.long),
        "ptr": torch.tensor([0, N], dtype=torch.long),
    }
    out = model(data, training=False, compute_force=True)
    return out["energy"].detach(), out["forces"].detach()


def run_ictd(model, report, positions, cell, z_idx, edge_index, shifts, unit_shifts):
    """ICTC total energy assembled to match MACE: E = E_scaled_interaction + E0(Z).

    The converted model returns MACE's scaled per-atom interaction energy [N,1] (no E0).
    ``convert_mace_to_ictd`` installs MACE's ScaleShiftBlock into the ICTC model, so the caller
    only adds atomic_energies[Z].
    """
    N = positions.shape[0]
    A = torch.tensor([ATOMIC_NUMBERS[i] for i in z_idx], dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    cell_t = torch.tensor(cell, dtype=torch.float64).reshape(1, 3, 3)  # (n_graphs, 3, 3)
    batch = torch.zeros(N, dtype=torch.long)
    edge_src = torch.tensor(edge_index[0], dtype=torch.long)
    edge_dst = torch.tensor(edge_index[1], dtype=torch.long)
    edge_shifts = torch.tensor(unit_shifts, dtype=torch.float64)  # model multiplies by cell

    e_inter = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell_t).squeeze(-1)
    # E0 per atom (atomic_energies indexed by element index = position in ATOMIC_NUMBERS)
    atomic_energies = report["atomic_energies"].to(dtype=torch.float64)
    e0 = atomic_energies[torch.tensor(z_idx, dtype=torch.long)]
    e_total = (e_inter + e0).sum()
    forces = -torch.autograd.grad(e_total, pos, create_graph=False)[0]
    return e_total.detach().reshape(1), forces.detach()


def _compare_once(
    n_atoms,
    box,
    seed,
    *,
    product_backend="ictd-bridge-u",
    first_interaction="residual",
    num_interactions: int = 2,
    max_ell: int = LMAX,
    hidden_lmax: int = LMAX,
    verbose=True,
):
    """Build both models, convert, and compare energy+forces on one random graph.

    A fresh MACE (random weights) is built per call so the test exercises arbitrary weights,
    not just one initialization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mace = build_mace(
        first_interaction=first_interaction,
        num_interactions=num_interactions,
        max_ell=max_ell,
        hidden_lmax=hidden_lmax,
    )
    ictd = build_ictd(
        product_backend=product_backend,
        use_reduced_cg=bool(getattr(mace, "use_reduced_cg", False)),
        num_interactions=num_interactions,
        max_ell=max_ell,
        hidden_lmax=hidden_lmax,
    )
    report = convert_mace_to_ictd(mace, ictd)
    assert all(len(p._forward_hooks) == 0 for p in ictd.products), "converter must not install forward hooks"

    positions, cell, z_idx, edge_index, shifts, unit_shifts = make_graph(n_atoms=n_atoms, box=box, seed=seed)
    E_mace, F_mace = run_mace(mace, positions, cell, z_idx, edge_index, shifts, unit_shifts)
    E_ictd, F_ictd = run_ictd(ictd, report, positions, cell, z_idx, edge_index, shifts, unit_shifts)

    dE = (E_mace - E_ictd).abs().max().item()
    dF = (F_mace - F_ictd).abs().max().item()
    relE = dE / max(E_mace.abs().item(), 1e-12)
    if verbose:
        print(f"[backend={product_backend:13s} N={n_atoms:3d} box={box:.1f} seed={seed}] "
              f"first={first_interaction:11s} "
              f"layers={num_interactions} hidden_lmax={hidden_lmax} max_ell={max_ell} "
              f"E_mace={E_mace.item():.6f}  max|dE|={dE:.2e} rel={relE:.2e}  "
              f"max|dF|={dF:.2e} (|F|max={F_mace.abs().max().item():.4f})")
    return report, dE, dF, relE, (mace, ictd)


def _check_energy_rotation_invariance(mace, ictd, report, seed=99):
    """Energy is an SO(3) invariant: rotating the system must leave MACE and ICTC energies equal to
    each other AND each unchanged under rotation. Validates the conversion is genuinely equivariant
    (the ICTC<->e3nn basis differs by a fixed orthogonal Q, which energy is invariant to)."""
    rng = np.random.default_rng(seed)
    positions, cell, z_idx, edge_index, shifts, unit_shifts = make_graph(n_atoms=20, box=8.0, seed=seed)
    # random rotation
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    pos_rot = positions @ q.T
    cell_rot = cell @ q.T
    shifts_rot = shifts @ q.T
    E0_m, _ = run_mace(mace, positions, cell, z_idx, edge_index, shifts, unit_shifts)
    E1_m, _ = run_mace(mace, pos_rot, cell_rot, z_idx, edge_index, shifts_rot, unit_shifts)
    E0_i, _ = run_ictd(ictd, report, positions, cell, z_idx, edge_index, shifts, unit_shifts)
    E1_i, _ = run_ictd(ictd, report, pos_rot, cell_rot, z_idx, edge_index, shifts_rot, unit_shifts)
    d_rot_m = (E0_m - E1_m).abs().item()
    d_rot_i = (E0_i - E1_i).abs().item()
    d_mi = (E1_m - E1_i).abs().item()
    print(f"[rotation] MACE rot-invariance dE={d_rot_m:.2e}  ICTC rot-invariance dE={d_rot_i:.2e}  "
          f"MACE-vs-ICTC(rotated) dE={d_mi:.2e}")
    assert d_rot_i < 1e-6, f"ICTC energy not rotation invariant: {d_rot_i}"
    assert d_mi < 1e-5, f"MACE/ICTC disagree after rotation: {d_mi}"


def main():
    torch.set_default_dtype(torch.float64)
    print("=" * 72)
    print("convert_mace_to_ictd: energy + force agreement (float64)")
    print("=" * 72)

    exact_backends = ("ictd-bridge-u", "native-mace")
    variants = [
        {
            "name": "base",
            "num_interactions": 2,
            "max_ell": 2,
            "hidden_lmax": 2,
            "configs": [(24, 8.0, 0), (20, 8.0, 7), (32, 9.0, 3), (16, 7.0, 11)],
            "backends": exact_backends,
            "first_interactions": ("residual", "nonresidual"),
            "rotation": True,
        },
        {
            "name": "three-interactions",
            "num_interactions": 3,
            "max_ell": 2,
            "hidden_lmax": 2,
            "configs": [(18, 8.0, 13)],
            "backends": exact_backends,
            "first_interactions": ("residual",),
            "rotation": True,
        },
        {
            "name": "four-interactions",
            "num_interactions": 4,
            "max_ell": 2,
            "hidden_lmax": 2,
            "configs": [(14, 7.5, 17)],
            "backends": ("ictd-bridge-u",),
            "first_interactions": ("residual",),
            "rotation": False,
        },
        {
            "name": "max-ell3-hidden2",
            "num_interactions": 2,
            "max_ell": 3,
            "hidden_lmax": 2,
            "configs": [(18, 8.0, 19)],
            "backends": exact_backends,
            "first_interactions": ("residual", "nonresidual"),
            "rotation": True,
        },
        {
            "name": "max-ell4-hidden3",
            "num_interactions": 2,
            "max_ell": 4,
            "hidden_lmax": 3,
            "configs": [(14, 8.0, 23)],
            "backends": ("ictd-bridge-u",),
            "first_interactions": ("residual",),
            "rotation": False,
        },
    ]
    global_worst_dE = global_worst_dF = 0.0
    last = None
    n_checks = 0
    for variant in variants:
        print(f"-- variant={variant['name']} --")
        for first_interaction in variant["first_interactions"]:
            for product_backend in variant["backends"]:
                worst_dE = worst_dF = 0.0
                for n_atoms, box, seed in variant["configs"]:
                    report, dE, dF, relE, models = _compare_once(
                        n_atoms,
                        box,
                        seed,
                        product_backend=product_backend,
                        first_interaction=first_interaction,
                        num_interactions=variant["num_interactions"],
                        max_ell=variant["max_ell"],
                        hidden_lmax=variant["hidden_lmax"],
                    )
                    n_checks += 1
                    worst_dE = max(worst_dE, relE)
                    worst_dF = max(worst_dF, dF)
                    last = (report, models)
                global_worst_dE = max(global_worst_dE, worst_dE)
                global_worst_dF = max(global_worst_dF, worst_dF)
                print(
                    f"[variant={variant['name']} backend={product_backend} first={first_interaction}] "
                    f"worst rel|dE|={worst_dE:.3e} max|dF|={worst_dF:.3e}"
                )

                if variant["rotation"]:
                    # rotation/equivariance sanity on a fresh pair for each exact backend
                    _, _, _, _, (mace_r, ictd_r) = _compare_once(
                        20,
                        8.0,
                        5,
                        product_backend=product_backend,
                        first_interaction=first_interaction,
                        num_interactions=variant["num_interactions"],
                        max_ell=variant["max_ell"],
                        hidden_lmax=variant["hidden_lmax"],
                        verbose=False,
                    )
                    report_r = convert_mace_to_ictd(mace_r, ictd_r)
                    _check_energy_rotation_invariance(mace_r, ictd_r, report_r)

    # per-block conv_tp calibration report (from the last config)
    report = last[0]
    print("-" * 72)
    for k, v in report["conv_tp"].items():
        print(k, "c_by_path:", v["c_by_path"])
    print("-" * 72)

    print(f"WORST over {n_checks} graph/backend/first-layer checks:  "
          f"rel|dE| = {global_worst_dE:.3e}   max|dF| = {global_worst_dF:.3e}")

    # Tolerances: pure float64 round-off. Every block converts to machine precision; the only
    # architectural subtlety (MACE's first-layer element self-connection, absent from the ICTC
    # baseline's first interaction) is injected as an additive per-element bias, so the result is
    # bit-exact up to float64 accumulation (~1e-11 absolute on a ~3e4 energy, ~1e-15 on forces).
    assert global_worst_dE < 1e-9, f"relative energy mismatch too large: {global_worst_dE}"
    assert global_worst_dF < 1e-6, f"force mismatch too large: {global_worst_dF}"
    print("\nPASS: MACE and MACE-ICTC agree on energy and forces (bit-exact, float64).")


def test_mace_converter():
    main()


if __name__ == "__main__":
    main()
