"""angular_basis switch: 'e3nn' makes the model compute its l>=1 features natively in the
original-MACE/e3nn spherical basis (Q folded into the harmonics + interaction CG; the order-3
symmetric contraction is run in the stable ICTC basis with an exact e3nn<->ICTC rotation at its
I/O). It is a single global orthogonal change of the angular basis, so:

  * energy/forces/virial are SO(3) invariants -> BIT-IDENTICAL to angular_basis='ictd';
  * every intermediate l>=1 node feature in 'e3nn' equals the 'ictd' feature re-expressed in the
    e3nn basis (x @ Q) to machine precision.

Run:  python -m mace_ictc.test.test_angular_basis
"""
import torch

from mace_ictc.synthetic import build_model, make_fixed_graph, compute_energy_forces
from mace_ictc.models.pure_cartesian_ictd_fix import SO3ToE3NNBasisBridge

CH, LMAX = 16, 2


def _mk(basis, *, dtype, seed=0):
    torch.manual_seed(seed)
    return build_model(channels=CH, lmax=LMAX, num_interaction=2, route="baseline",
                       product_backend="ictd-pure-u", dtype=dtype,
                       device=torch.device("cpu"), correlation=3, angular_basis=basis).eval()


def run(dtype=torch.float64):
    g = make_fixed_graph(num_nodes=40, avg_degree=18, dtype=dtype, device="cpu", seed=3)
    m_i = _mk("ictd", dtype=dtype)
    m_e = _mk("e3nn", dtype=dtype)  # identical weights; only the fixed angular operators are rotated

    feats = {}
    def hook(tag):
        def h(_m, _i, out):
            feats[tag] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return h
    m_i.products[0].register_forward_hook(hook("i"))
    m_e.products[0].register_forward_hook(hook("e"))

    E_i, F_i, _ = compute_energy_forces(m_i, g, create_graph=False)
    E_e, F_e, _ = compute_energy_forces(m_e, g, create_graph=False)
    assert getattr(m_e, "_e3nn_folded", False), "e3nn fold did not run"

    tol = 1e-9 if dtype == torch.float64 else 1e-4
    dE = (E_i - E_e).abs().max().item()
    dF = (F_i - F_e).abs().max().item()
    assert dE < tol and dF < tol, f"e3nn output not bit-identical to ictd: dE={dE:.2e} dF={dF:.2e}"

    # intermediate l>=1 feature parity: e3nn feature == ictd feature @ Q
    fi, fe = feats["i"], feats["e"]
    bridge = SO3ToE3NNBasisBridge(CH, LMAX)
    fi_e3nn = bridge.ictd_flat_to_e3nn_flat(fi, LMAX)
    df = (fe - fi_e3nn).abs().max().item()
    rot = (fi - fi_e3nn).abs().max().item()  # how much Q actually rotates l>=1 (must be non-trivial)
    assert rot > 1e-3, "Q is trivial here (test would be vacuous)"
    assert df < tol, f"e3nn features != ictd features @ Q: {df:.2e}"
    return dict(dE=dE, dF=dF, feat=df, rot=rot)


def main():
    for dt, name in ((torch.float64, "float64"), (torch.float32, "float32")):
        torch.set_default_dtype(dt)
        r = run(dt)
        print(f"[{name}] PASS  output dE={r['dE']:.1e} dF={r['dF']:.1e}  |  "
              f"feature |e3nn - ictd@Q|={r['feat']:.1e} (Q rotates l>=1 by {r['rot']:.1e})")
    print("angular_basis switch: e3nn == ictd in output, == ictd@Q in features")


if __name__ == "__main__":
    main()
