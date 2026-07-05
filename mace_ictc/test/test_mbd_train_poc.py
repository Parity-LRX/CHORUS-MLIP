"""Synthetic-data POC that the ManyBodyDispersionSLQ (omega,alpha) head TRAINS: ground truth is a dense
MBD energy surface with FIXED per-element (omega*, alpha*, coupling_scale*, beta*); the head (one-hot
element features -> alpha_head/omega_head + coupling_scale + beta_raw) is fit on energy MSE. The head
reproduces the energies and coupling_scale grows from its 0.03 init. (The MBD energy has a (omega,alpha,
coupling_scale) degeneracy, so the specific params are not uniquely recovered -- the *energy* is; real
training breaks the degeneracy with forces + more data + physical priors.)

Run the full report: python mace_ictc/test/test_mbd_train_poc.py
"""
import os
import sys

import torch
torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.environ.get("MACE_ICTD_ROOT", os.path.join(os.path.dirname(__file__), "..", "..")))
from mace_ictc.models.dispersion import ManyBodyDispersionSLQ

NEL = 3
TRUE_OM = torch.tensor([1.0, 0.8, 1.2])
TRUE_AL = torch.tensor([0.5, 0.7, 0.4])
TRUE_CS, TRUE_BETA = 1.5, 1.5
_EYE3 = torch.eye(3)


def _edges(n):
    es, ed = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                es.append(j); ed.append(i)
    return torch.tensor(es), torch.tensor(ed)


def _dense_energy(pos, elems, om_e, al_e, cs, beta):
    n = len(elems)
    om, al = om_e[elems], al_e[elems]
    C = torch.diag(om.repeat_interleave(3) ** 2)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rv = pos[i] - pos[j]; r = rv.norm(); rhat = rv / r
            T = (3 * rhat[:, None] * rhat[None, :] - _EYE3) / r ** 3
            radius = al[i] ** (1 / 3) + al[j] ** (1 / 3) + 1e-6
            damp = 1 - torch.exp(-((r / (beta * radius)) ** 6))
            C[3 * i:3 * i + 3, 3 * j:3 * j + 3] += cs * om[i] * om[j] * torch.sqrt(al[i] * al[j]) * damp * T
    eig = torch.linalg.eigvalsh(C).clamp_min(1e-8)
    return 0.5 * eig.sqrt().sum() - 1.5 * om.sum()


def _make_data(num, seed=1):
    g = torch.Generator().manual_seed(seed)
    data = []
    for _ in range(num):
        n = int(torch.randint(4, 8, (1,), generator=g))
        pos = torch.rand(n, 3, generator=g) * 3.5 + 1.0
        elems = torch.randint(0, NEL, (n,), generator=g)
        data.append((pos, elems, _dense_energy(pos, elems, TRUE_OM, TRUE_AL, TRUE_CS, TRUE_BETA).detach()))
    return data


def _train(data, steps, lr=0.02):
    torch.manual_seed(0)
    model = ManyBodyDispersionSLQ(feature_dim=NEL, hidden_dim=16, probe_mode="basis", operator_backend="edge_sparse")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def e_model(pos, elems):
        n = len(elems)
        feats = torch.nn.functional.one_hot(elems, NEL).double()
        es, ed = _edges(n)
        return model(feats, torch.zeros(n, dtype=torch.long), es, ed, pos[ed] - pos[es], num_graphs=1).sum()

    loss0 = None
    for step in range(steps):
        opt.zero_grad()
        loss = sum((e_model(p, e) - E) ** 2 for p, e, E in data) / len(data)
        loss.backward(); opt.step()
        if loss0 is None:
            loss0 = loss.item()
    return model, e_model, loss0, loss.item()


def test_mbd_head_trains():
    data = _make_data(24)
    model, e_model, loss0, loss1 = _train(data, steps=200)
    assert loss1 < loss0 / 50.0, f"head did not learn: loss {loss0} -> {loss1}"
    assert float(model.coupling_scale) > 0.05, f"coupling_scale did not grow from 0.03: {float(model.coupling_scale)}"


if __name__ == "__main__":
    data = _make_data(24)
    model, e_model, loss0, loss1 = _train(data, steps=401)
    src = model.emit_source(torch.eye(NEL))
    mean_absE = sum(abs(E) for _, _, E in data) / len(data)
    rmse = (sum((e_model(p, e).detach() - E) ** 2 for p, e, E in data) / len(data)).sqrt()
    print(f"loss {loss0:.3e} -> {loss1:.3e}   rmse {rmse.item():.3e}  mean|E| {mean_absE.item():.3e}  "
          f"rel {float(rmse / mean_absE):.2%}   coupling_scale 0.03 -> {float(model.coupling_scale):.3f}")
    print("OK: ManyBodyDispersionSLQ (omega,alpha) head trains to reproduce the MBD energy surface")
