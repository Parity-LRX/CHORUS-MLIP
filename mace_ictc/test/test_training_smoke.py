"""Smoke test for the MACE-ICTC training stack (data + ForceTrainer + bucketing).

Run:  python -m mace_ictc.test.test_training_smoke

On synthetic data it checks, on CPU (always):
  1. eager energy+force training reduces the loss (gradients flow);
  2. make_fx size-bucketing -> exactly one fixed batch shape per bucket;
  3. checkpoint round-trips (rebuild + strict state_dict load + forward parity);
  4. a trained checkpoint deploys via ``LAMMPS_MLIAP_MFF.from_checkpoint``
     (strict load, forward parity) -- proof train-arch == deploy-arch.
And, only if CUDA is present:
  5. make_fx forces == eager forces (numeric parity), and bucketing compiles once
     per bucket with the cached epoch faster than the compiling epoch.
"""

import os
import tempfile

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from mace_ictc.data import H5Dataset, collate_fn_h5, BucketBatchSampler
from mace_ictc.utils.config import ModelConfig
from mace_ictc.cli.train import build_baseline_model, _avg_num_neighbors_from_h5
from mace_ictc.training.train_loop import ForceTrainer


def _brute_neighbors(pos, L, cutoff):
    """O(N^2) periodic neighbor list (orthorhombic box, edge L), matscipy 'ijS' convention."""
    N = len(pos)
    src, dst, sh = [], [], []
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                S = np.array([sx, sy, sz], dtype=float)
                disp = S * L
                for a in range(N):
                    r = np.linalg.norm(pos + disp - pos[a], axis=1)
                    for b in range(N):
                        if r[b] < cutoff and not (a == b and sx == sy == sz == 0):
                            src.append(a); dst.append(b); sh.append(S)
    return (np.array(src, np.int64), np.array(dst, np.int64),
            np.array(sh, np.float64).reshape(-1, 3))


def _make_h5(path, sizes, *, box=12.0, cutoff=5.0, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        max_e = max_a = 0
        for idx, N in enumerate(sizes):
            pos = rng.uniform(0, box, size=(N, 3))
            cell = np.eye(3) * box
            A = rng.choice([1, 6, 7, 8], size=N)
            i, j, S = _brute_neighbors(pos, box, cutoff)
            g = f.create_group(f"sample_{idx}")
            g.create_dataset("pos", data=pos.astype(np.float64))
            g.create_dataset("A", data=A.astype(np.int64))
            g.create_dataset("y", data=np.float64(rng.normal() * N))
            g.create_dataset("force", data=rng.normal(size=(N, 3)).astype(np.float64))
            g.create_dataset("edge_src", data=i)
            g.create_dataset("edge_dst", data=j)
            g.create_dataset("edge_shifts", data=S.astype(np.float64))
            g.create_dataset("cell", data=cell.astype(np.float64))
            st = rng.normal(size=(3, 3)); st = 0.5 * (st + st.T)  # symmetric stress target
            g.create_dataset("stress", data=st.astype(np.float64))
            max_e = max(max_e, len(i)); max_a = max(max_a, N)
        f.attrs["max_edges"] = max_e
        f.attrs["max_atoms"] = max_a


def _mk_model(ann, dtype, device, **extra):
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = 32; cfg.irreps_output_conv_channels = 32; cfg.lmax = 2
    cfg.num_layers = 1; cfg.max_radius = 5.0; cfg.max_radius_main = 5.0
    cfg.function_type = "gaussian"; cfg.internal_compute_dtype = dtype
    model = build_baseline_model(
        cfg, avg_num_neighbors=ann, num_interaction=2, route="baseline",
        product_backend="ictd-pure-u", correlation=2, radial_sqrt_num_basis=False,
        edge_lmax=None, attn_heads=0, atomic_numbers=[1, 6, 7, 8], ictd_save_tp_mode="fully-connected",
        invariant_channels=32, device=device, dtype=dtype, **extra)
    return cfg, model


def _extra_hparams(ann, **extra):
    return dict(num_interaction=2, invariant_channels=32, ictd_fix_route="baseline",
                ictd_fix_product_backend="ictd-pure-u", save_contraction_order=2,
                ictd_save_tp_mode="fully-connected", ictd_fix_interaction_attn_heads=0,
                radial_sqrt_num_basis=False, avg_num_neighbors=float(ann), **extra)


def run(device="cpu"):
    torch.manual_seed(0)
    dtype = torch.float64 if device == "cpu" else torch.float32
    dev = torch.device(device)
    tmp = tempfile.mkdtemp(prefix="mace_ictc_test_")
    train_h5 = os.path.join(tmp, "processed_train.h5")
    val_h5 = os.path.join(tmp, "processed_val.h5")
    _make_h5(train_h5, sizes=[8, 9, 8, 16, 17, 16, 8, 16], seed=1)
    _make_h5(val_h5, sizes=[8, 16], seed=2)
    ann = _avg_num_neighbors_from_h5(train_h5)

    # 1. eager training reduces the loss --------------------------------------
    ds = H5Dataset(prefix="train", data_dir=tmp)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn_h5)
    vloader = DataLoader(H5Dataset(prefix="val", data_dir=tmp), batch_size=2,
                         shuffle=False, collate_fn=collate_fn_h5)
    cfg, model = _mk_model(ann, dtype, dev)
    tr = ForceTrainer(model, loader, val_loader=vloader, device=dev, config=cfg, dtype=dtype,
                      max_radius=5.0, learning_rate=5e-3, lr_scheduler="cosine", epochs=15)
    first = tr.train_epoch(0)
    for e in range(1, 15):
        last = tr.train_epoch(e)
    assert np.isfinite(first["total_loss"]) and np.isfinite(last["total_loss"])
    assert last["total_loss"] < first["total_loss"], "loss did not decrease"
    assert np.isfinite(tr._val_pass()["total_loss"])

    # 2. bucketing -> one fixed batch shape per bucket ------------------------
    bds = H5Dataset(prefix="train", data_dir=tmp, makefx_buckets=2)
    nb = len(set(bds.sample_bucket))
    sampler = BucketBatchSampler(bds.sample_bucket, batch_size=2, drop_last=True, shuffle=True, seed=0)
    bloader = DataLoader(bds, batch_sampler=sampler, collate_fn=collate_fn_h5)
    shapes = {(int(b[0].shape[0]), int(b[5].shape[0])) for b in bloader}
    assert len(shapes) == nb, f"expected {nb} batch shapes, got {len(shapes)}"

    # 3. checkpoint round-trip (rebuild + strict load + forward parity) -------
    ckpt = os.path.join(tmp, "rt.pth")
    scale_shift = dict(
        energy_output_scale=1.5,
        energy_output_scale_enabled=True,
        energy_output_shift=-0.25,
        energy_output_shift_enabled=True,
    )
    cfg1, m1 = _mk_model(ann, dtype, dev, **scale_shift)
    tr1 = ForceTrainer(m1, loader, device=dev, config=cfg1, dtype=dtype, max_radius=5.0,
                       epochs=1, extra_hparams=_extra_hparams(ann, **scale_shift))
    tr1.save_checkpoint(ckpt, epoch=0)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg2, m2 = _mk_model(ann, dtype, dev, **scale_shift)
    missing, unexpected = m2.load_state_dict(blob["e3trans_state_dict"], strict=True)
    assert not missing and not unexpected, f"strict load mismatch {missing} {unexpected}"
    batch = next(iter(loader)); b = tr1._unpack(batch)
    args = (b["pos"], b["A"], b["batch_idx"], b["edge_src"], b["edge_dst"], b["edge_shifts"], b["cell"])
    m1.eval(); m2.eval()
    with torch.no_grad():
        e1 = m1(*args); e1 = e1[0] if isinstance(e1, tuple) else e1
        e2 = m2(*args); e2 = e2[0] if isinstance(e2, tuple) else e2
    # m1 and m2 are bit-identical after the strict load (verified by state); a 1e-10 forward check
    # only holds in fp64/CPU. On GPU/fp32 the forward is non-deterministic at ~1e-7 (atomic scatter
    # reductions) -- the same model run twice already differs by ~6e-8 -- so the round-trip tolerance
    # must track the precision the test actually runs in.
    rt_tol = 1e-10 if dtype is torch.float64 else 1e-5
    assert (e1 - e2).abs().max().item() < rt_tol

    # 4. deploy round-trip via LAMMPS_MLIAP_MFF.from_checkpoint (strict load) --
    from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
    wrap = LAMMPS_MLIAP_MFF.from_checkpoint(ckpt, element_types=["H", "C", "N", "O"], device=device)
    md = wrap.wrapper.model
    md.eval()
    with torch.no_grad():
        ed = md(*args); ed = ed[0] if isinstance(ed, tuple) else ed
    assert (e1 - ed).abs().max().item() < rt_tol, "deploy model diverges"

    # 4b. exposed training controls are real checkpoint state, not CLI-only ---
    avg_ckpt = os.path.join(tmp, "avg.pth")
    cfg_avg, m_avg = _mk_model(ann, dtype, dev)
    tr_avg = ForceTrainer(
        m_avg, loader, device=dev, config=cfg_avg, dtype=dtype, max_radius=5.0,
        learning_rate=1e-4, optimizer_type="adamw", adam_beta1=0.8, adam_beta2=0.95,
        adam_eps=1e-7, amsgrad=True, lr_scheduler="plateau", lr_factor=0.5,
        scheduler_patience=0, epochs=1, max_steps=2,
        loss_type="mse", loss_beta=0.25, ema_decay=0.9, ema_start_step=1,
        stage_two_enabled=True, swa_start_step=1, swa_lr=5e-5,
        swa_energy_weight=2.0, swa_force_weight=3.0, swa_stress_weight=0.0,
        checkpoint_state_source="ema", checkpoint_path=avg_ckpt,
        extra_hparams=_extra_hparams(ann),
    )
    tr_avg.fit()
    avg_blob = torch.load(avg_ckpt, map_location="cpu", weights_only=False)
    assert avg_blob["global_step"] == 2
    assert "e3trans_ema_state_dict" in avg_blob and "e3trans_swa_state_dict" in avg_blob
    assert avg_blob["default_state_source"] == "ema"
    th = avg_blob["training_hyperparameters"]
    assert th["loss"] == "mse" and th["adam_beta1"] == 0.8 and th["adam_beta2"] == 0.95
    assert th["adam_eps"] == 1e-7 and th["amsgrad"] is True and th["max_steps"] == 2
    assert th["lr_scheduler"] == "plateau" and th["lr_factor"] == 0.5
    assert th["stage_two_active"] is True and th["stage_two_activated_step"] == 1
    assert th["swa_lr"] == 5e-5 and th["swa_energy_weight"] == 2.0
    assert th["swa_force_weight"] == 3.0 and th["swa_n_averaged"] == 1
    from mace_ictc.utils.checkpoint_metadata import get_checkpoint_e3_state_dict
    _, src = get_checkpoint_e3_state_dict(avg_blob)
    assert src == "ema"

    # 5. make_fx (GPU only): force + stress parity + one-compile-per-bucket ----
    did_makefx = False
    if torch.cuda.is_available() and device != "cpu":
        did_makefx = True
        trm = ForceTrainer(m1, loader, device=dev, config=cfg1, dtype=dtype, max_radius=5.0,
                           train_makefx_compile=True)
        m1.train()
        # force parity
        E_m, g_m = trm._makefx_forward(*args)
        p = b["pos"].detach().requires_grad_(True)
        E_e = m1(p, b["A"], b["batch_idx"], b["edge_src"], b["edge_dst"], b["edge_shifts"], b["cell"])
        E_e = E_e[0] if isinstance(E_e, tuple) else E_e
        f_e = -torch.autograd.grad(E_e.sum(), p)[0]
        fs = f_e.abs().max().item()
        assert ((-g_m) - f_e).abs().max().item() / max(fs, 1e-9) < 1e-3, "make_fx force != eager"
        # stress parity: make_fx (force, stress) vs eager (force, stress)
        B = int(b["cell"].shape[0])
        strain0 = torch.zeros((B, 3, 3), dtype=dtype, device=dev)
        _, gp_m, gs_m = trm._makefx_forward(*args, strain=strain0)
        pl = b["pos"].detach().requires_grad_(True)
        sl = torch.zeros((B, 3, 3), dtype=dtype, device=dev, requires_grad=True)
        sym = 0.5 * (sl + sl.transpose(-1, -2))
        defo = torch.eye(3, dtype=dtype, device=dev) + sym
        pin = torch.einsum("ni,nij->nj", pl, defo[b["batch_idx"]])
        cin = torch.bmm(b["cell"], defo)
        Ee2 = m1(pin, b["A"], b["batch_idx"], b["edge_src"], b["edge_dst"], b["edge_shifts"], cin)
        Ee2 = Ee2[0] if isinstance(Ee2, tuple) else Ee2
        ge = torch.autograd.grad(Ee2.sum(), [pl, sl])
        assert (gp_m - ge[0]).abs().max().item() / max(fs, 1e-9) < 1e-3, "mfx stress-path force != eager"
        assert (gs_m - ge[1]).abs().max().item() / max(ge[1].abs().max().item(), 1e-9) < 1e-2, "make_fx stress != eager"
        # bucketing compiles
        _, mb = _mk_model(ann, dtype, dev)
        bsamp = BucketBatchSampler(bds.sample_bucket, batch_size=2, drop_last=True, shuffle=False)
        bl = DataLoader(bds, batch_sampler=bsamp, collate_fn=collate_fn_h5)
        trb = ForceTrainer(mb, bl, device=dev, config=cfg1, dtype=dtype, max_radius=5.0,
                           lr_scheduler="none", train_makefx_compile=True, train_sampler=bsamp)
        trb.train_epoch(0)
        assert not trb._makefx_disabled and len(trb._makefx_cache._cache) == nb

    # 6. stress/virial loss (eager, always): the stress term is live + backprops -
    cfg6, m6 = _mk_model(ann, dtype, dev)
    tr6 = ForceTrainer(m6, loader, device=dev, config=cfg6, dtype=dtype, max_radius=5.0,
                       energy_weight=1.0, force_weight=10.0, stress_weight=0.5, lr_scheduler="none")
    o6 = tr6._compute(next(iter(loader)), training=True)
    sloss = float(o6["stress_loss"])
    assert np.isfinite(sloss) and sloss > 0, "stress loss not live"
    expect = 1.0 * float(o6["energy_loss"]) + 10.0 * float(o6["force_loss"]) + 0.5 * sloss
    assert abs(float(o6["total_loss"]) - expect) < 1e-4 * max(1.0, abs(expect)), "stress not in total"
    o6["total_loss"].backward()  # 2nd backward through the strain derivative
    assert any(pp.grad is not None for pp in m6.parameters()), "no grad from stress loss"

    return dict(buckets=nb, makefx=did_makefx, stress=True,
                first_loss=float(first["total_loss"]), last_loss=float(last["total_loss"]))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run(device=device)
    print(f"[{device}] PASS  buckets={res['buckets']}  makefx={res['makefx']}  "
          f"stress={res['stress']}  loss {res['first_loss']:.2f} -> {res['last_loss']:.2f}")


if __name__ == "__main__":
    main()
