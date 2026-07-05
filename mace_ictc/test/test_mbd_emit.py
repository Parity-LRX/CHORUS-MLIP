"""Deploy-path verification: the ManyBodyDispersionSLQ head emits (omega, alpha); the dense MBD energy
from those emitted values (== what the C++ MBD solver consumes) equals the module's own forward energy.
Combined with the C++ mbd_energy_dense == Python dense parity (verified in the C++ MBD_PHYS_TEST), this
shows the deployed MBD (head -> (omega,alpha) -> C++ solver) reproduces the training-time energy.
"""
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.environ.get("MACE_ICTD_ROOT", os.path.join(os.path.dirname(__file__), "..", "..")))
from mace_ictc.models.dispersion import ManyBodyDispersionSLQ


def test_emit_source_matches_module_energy():
    torch.manual_seed(0)
    fd, N = 8, 4
    slq = ManyBodyDispersionSLQ(feature_dim=fd, hidden_dim=16, probe_mode="basis",
                                operator_backend="edge_sparse").double().eval()
    feats = torch.randn(N, fd)
    pos = torch.tensor([[0.0, 0, 0], [2.5, 0, 0], [0, 2.8, 0], [1.2, 1.3, 2.1]])
    batch = torch.zeros(N, dtype=torch.long)
    es, ed, ev = [], [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                es.append(j); ed.append(i); ev.append((pos[i] - pos[j]).tolist())
    es, ed, ev = torch.tensor(es), torch.tensor(ed), torch.tensor(ev)

    e_module = slq(feats, batch, es, ed, ev, num_graphs=1).sum().item()

    src = slq.emit_source(feats)
    omega, alpha = src[:, 0], src[:, 1]
    beta, cs = slq.mbd_beta(), slq.mbd_coupling_scale()
    assert (omega > 0).all() and (alpha > 0).all() and src.shape == (N, 2)

    eye3 = torch.eye(3)
    C = torch.diag(omega.repeat_interleave(3) ** 2)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rv = pos[i] - pos[j]; r = rv.norm(); rhat = rv / r
            T = (3 * rhat[:, None] * rhat[None, :] - eye3) / r ** 3
            radius = alpha[i] ** (1 / 3) + alpha[j] ** (1 / 3) + 1e-6
            damp = 1 - torch.exp(-((r / (beta * radius)) ** 6))
            C[3 * i:3 * i + 3, 3 * j:3 * j + 3] += cs * omega[i] * omega[j] * torch.sqrt(alpha[i] * alpha[j]) * damp * T
    eig = torch.linalg.eigvalsh(C).clamp_min(slq.eig_floor)
    e_dense = (0.5 * eig.sqrt().sum() - 1.5 * omega.sum()).item()

    abs_err = abs(e_module - e_dense)
    rel = abs_err / (abs(e_dense) + 1e-12)
    assert abs_err < 1e-10 or rel < 1e-6, (
        f"emit-source dense energy {e_dense} != module energy {e_module} "
        f"(abs {abs_err}, rel {rel})"
    )


if __name__ == "__main__":
    test_emit_source_matches_module_energy()
    print("OK: ManyBodyDispersionSLQ.emit_source (omega,alpha) -> dense MBD energy == module forward energy")
