#!/usr/bin/env python3
"""Small empirical NTK-spectrum diagnostic for matched MACE and MACE-ICTC models.

This is intentionally a *diagnostic*, not a full training benchmark. The full
force-field empirical kernel over all validation configurations and force
components is too large to materialize. Instead, this script fixes a small H5
batch, constructs the weighted output vector

    g = [sqrt(w_E / B) E_b / N_b, sqrt(w_F / M) F_s],

where F_s is a deterministic sample of force components, and reports the spectrum
of K = J J^T with respect to the trainable parameters at the random
MACE-compatible initialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from e3nn import o3
from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes
from mace.modules.wrapper_ops import CuEquivarianceConfig

from mace_ictc.cli.train import (
    _atomic_inter_scale_shift_from_h5,
    _set_global_seed,
    build_baseline_model,
)
from mace_ictc.data import H5Dataset, collate_fn_h5
from mace_ictc.interfaces.mace_converter import convert_mace_to_ictd
from mace_ictc.utils.config import ModelConfig
from mace_ictc.utils.scatter import scatter


def parse_csv_numbers(value: str, cast=float):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def hidden_irreps(channels: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{channels}x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(lmax + 1)))


def build_mace_model(
    args,
    atomic_numbers: list[int],
    e0_values: list[float],
    scale: float,
    shift: float,
    device,
    dtype,
    *,
    use_cueq: bool = False,
):
    torch.set_default_dtype(torch.float64 if dtype == torch.float64 else torch.float32)
    cueq_config = None
    if use_cueq:
        cueq_config = CuEquivarianceConfig(
            enabled=True,
            layout="ir_mul",
            group="O3_e3nn",
            optimize_all=True,
            conv_fusion=(str(device).startswith("cuda")),
        )
    return ScaleShiftMACE(
        r_max=float(args.r_max),
        num_bessel=int(args.num_basis),
        num_polynomial_cutoff=int(args.polynomial_cutoff_p),
        max_ell=int(args.max_ell),
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=int(args.num_interactions),
        num_elements=len(atomic_numbers),
        hidden_irreps=hidden_irreps(args.channels, args.hidden_lmax),
        MLP_irreps=o3.Irreps(f"{int(args.readout_hidden_channels)}x0e"),
        atomic_energies=np.asarray(e0_values, dtype=np.float64),
        avg_num_neighbors=float(args.avg_num_neighbors),
        atomic_numbers=[int(z) for z in atomic_numbers],
        correlation=int(args.correlation),
        gate=gate_dict["silu"],
        radial_type="bessel",
        radial_MLP=[64, 64, 64],
        atomic_inter_scale=float(scale),
        atomic_inter_shift=float(shift),
        use_reduced_cg=bool(args.use_reduced_cg),
        cueq_config=cueq_config,
    ).to(device=device, dtype=dtype)


def build_ictd_model(args, atomic_numbers: list[int], scale: float, shift: float, product_backend: str, device, dtype):
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = int(args.channels)
    cfg.irreps_output_conv_channels = int(args.channels)
    cfg.lmax = int(args.hidden_lmax)
    cfg.num_layers = 1
    cfg.max_radius = float(args.r_max)
    cfg.max_radius_main = float(args.r_max)
    cfg.number_of_basis = int(args.num_basis)
    cfg.number_of_basis_main = int(args.num_basis)
    cfg.function_type = "bessel"
    cfg.internal_compute_dtype = dtype
    return build_baseline_model(
        cfg,
        avg_num_neighbors=float(args.avg_num_neighbors),
        num_interaction=int(args.num_interactions),
        route="baseline",
        product_backend=product_backend,
        correlation=int(args.correlation),
        use_reduced_cg=bool(args.use_reduced_cg),
        first_layer_self_connection=True,
        interaction_scale="none",
        conv_tp_scale_init="none",
        freeze_conv_tp_weight=False,
        interaction_init="identity",
        readout_hidden_channels=int(args.readout_hidden_channels),
        polynomial_cutoff_p=int(args.polynomial_cutoff_p),
        angular_basis="ictd",
        radial_sqrt_num_basis=False,
        edge_lmax=int(args.max_ell),
        attn_heads=0,
        atomic_numbers=atomic_numbers,
        ictd_save_tp_mode="fully-connected",
        invariant_channels=32,
        energy_output_scale=float(scale),
        energy_output_scale_enabled=True,
        energy_output_shift=float(shift),
        energy_output_shift_enabled=True,
        device=device,
        dtype=dtype,
    )


def load_batch(data_dir: Path, split: str, batch_size: int, batch_index: int):
    ds = H5Dataset(prefix=split, data_dir=str(data_dir))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_h5, num_workers=0)
    for i, batch in enumerate(loader):
        if i == int(batch_index):
            return batch
    raise IndexError(f"batch_index={batch_index} out of range for {data_dir}/{split}")


def move_batch(batch, device, dtype):
    if len(batch) == 11:
        batch = batch[:10]
    pos, A, batch_idx, force_ref, target_e, edge_src, edge_dst, edge_shifts, cell, stress = batch
    return (
        pos.to(device=device, dtype=dtype),
        A.to(device=device, dtype=torch.long),
        batch_idx.to(device=device, dtype=torch.long),
        force_ref.to(device=device, dtype=dtype),
        target_e.to(device=device, dtype=dtype),
        edge_src.to(device=device, dtype=torch.long),
        edge_dst.to(device=device, dtype=torch.long),
        edge_shifts.to(device=device, dtype=dtype),
        cell.to(device=device, dtype=dtype),
        stress.to(device=device, dtype=dtype),
    )


def ptr_from_batch(batch_idx: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(batch_idx, minlength=int(batch_idx.max().item()) + 1)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


def one_hot_attrs(A: torch.Tensor, atomic_numbers: list[int], dtype: torch.dtype) -> torch.Tensor:
    z_to_i = {int(z): i for i, z in enumerate(atomic_numbers)}
    idx = torch.tensor([z_to_i[int(z)] for z in A.detach().cpu().tolist()], device=A.device, dtype=torch.long)
    out = torch.zeros((A.numel(), len(atomic_numbers)), device=A.device, dtype=dtype)
    out[torch.arange(A.numel(), device=A.device), idx] = 1.0
    return out


def mace_outputs(model, batch, atomic_numbers: list[int], e0_values: list[float], create_graph: bool):
    pos, A, batch_idx, _, _, edge_src, edge_dst, edge_shifts, cell, _ = batch
    pos = pos.detach().clone().requires_grad_(True)
    edge_batch = batch_idx[edge_src]
    shifts = torch.einsum("ni,nij->nj", edge_shifts, cell[edge_batch])
    data = {
        "positions": pos,
        "node_attrs": one_hot_attrs(A, atomic_numbers, pos.dtype),
        "edge_index": torch.stack([edge_src, edge_dst], dim=0),
        "shifts": shifts,
        "unit_shifts": edge_shifts,
        "cell": cell,
        "batch": batch_idx,
        "ptr": ptr_from_batch(batch_idx),
    }
    out = model(data, training=create_graph, compute_force=True)
    energy = out["energy"].reshape(-1)
    forces = out["forces"]
    return energy, forces


def ictd_outputs(model, batch, atomic_numbers: list[int], e0_values: list[float], create_graph: bool):
    pos, A, batch_idx, _, _, edge_src, edge_dst, edge_shifts, cell, _ = batch
    pos = pos.detach().clone().requires_grad_(True)
    e_atom = model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell).squeeze(-1)
    e0_by_z = {int(z): float(v) for z, v in zip(atomic_numbers, e0_values)}
    e0_atom = torch.tensor([e0_by_z[int(z)] for z in A.detach().cpu().tolist()], device=A.device, dtype=pos.dtype)
    e_total = scatter(e_atom + e0_atom, batch_idx, dim=0)
    forces = -torch.autograd.grad(e_total.sum(), pos, create_graph=create_graph, retain_graph=True)[0]
    return e_total.reshape(-1), forces


def weighted_outputs(energy, forces, batch_idx, *, energy_weight: float, force_weight: float, max_force_components: int):
    n_graphs = int(energy.numel())
    counts = torch.bincount(batch_idx, minlength=n_graphs).to(dtype=energy.dtype)
    e_out = energy / counts.clamp_min(1)
    e_out = e_out * math.sqrt(float(energy_weight) / max(n_graphs, 1))
    f_flat = forces.reshape(-1)
    m = min(int(max_force_components), int(f_flat.numel()))
    if m <= 0:
        return e_out
    if m == f_flat.numel():
        idx = torch.arange(m, device=f_flat.device)
    else:
        idx = torch.linspace(0, f_flat.numel() - 1, steps=m, device=f_flat.device).round().to(torch.long)
    f_out = f_flat[idx] * math.sqrt(float(force_weight) / max(m, 1))
    return torch.cat([e_out, f_out], dim=0)


def flat_grad(y, params):
    grads = torch.autograd.grad(y, params, retain_graph=True, allow_unused=True)
    chunks = []
    for p, g in zip(params, grads):
        if g is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            chunks.append(g.reshape(-1))
    return torch.cat(chunks)


def kernel_spectrum(model, output_fn, batch, atomic_numbers, e0_values, args):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    energy, forces = output_fn(model, batch, atomic_numbers, e0_values, True)
    outs = weighted_outputs(
        energy,
        forces,
        batch[2],
        energy_weight=float(args.energy_weight),
        force_weight=float(args.force_weight),
        max_force_components=int(args.max_force_components),
    )
    rows = []
    for i in range(int(outs.numel())):
        rows.append(flat_grad(outs[i], params).detach())
    J = torch.stack(rows, dim=0)
    K = (J @ J.T).detach().to(dtype=torch.float64, device="cpu")
    eig = torch.linalg.eigvalsh(K).numpy()
    eig = np.maximum(eig, 0.0)
    trace = float(eig.sum())
    tol = max(float(eig.max()) * float(args.eig_tol), float(args.abs_eig_tol))
    positive = eig[eig > tol]
    if positive.size:
        lam_min = float(positive.min())
        lam_max = float(positive.max())
        kappa = float(lam_max / lam_min)
        lam_min_trace = float(lam_min / trace) if trace > 0.0 else 0.0
        lam_max_trace = float(lam_max / trace) if trace > 0.0 else 0.0
        rank = int(positive.size)
    else:
        lam_min = 0.0
        lam_max = float(eig.max()) if eig.size else 0.0
        kappa = math.inf
        lam_min_trace = 0.0
        lam_max_trace = 0.0
        rank = 0
    return {
        "n_outputs": int(outs.numel()),
        "n_params": int(sum(p.numel() for p in params if p.requires_grad)),
        "rank_tol": tol,
        "rank": rank,
        "lambda_min_pos": lam_min,
        "lambda_max": lam_max,
        "kappa_pos": kappa,
        "trace": trace,
        "lambda_min_pos_over_trace": lam_min_trace,
        "lambda_max_over_trace": lam_max_trace,
        "stable_gd_lr_bound": float(2.0 / lam_max) if lam_max > 0.0 else math.inf,
        "train_lr_times_lambda_max": float(args.lr * lam_max),
        "train_lr_times_lambda_min_pos": float(args.lr * lam_min),
        "eigvals": eig.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--max-force-components", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--modes", default="mace_e3nn,mace_cueq,ictd_bridge_u,ictd_cueq")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--hidden-lmax", type=int, default=1)
    p.add_argument("--max-ell", type=int, default=2)
    p.add_argument("--num-interactions", type=int, default=2)
    p.add_argument("--correlation", type=int, default=2)
    p.add_argument("--readout-hidden-channels", type=int, default=64)
    p.add_argument("--r-max", type=float, default=4.5)
    p.add_argument("--num-basis", type=int, default=8)
    p.add_argument("--polynomial-cutoff-p", type=int, default=6)
    p.add_argument("--avg-num-neighbors", type=float, default=8.0)
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--force-weight", type=float, default=100.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--atomic-energy-keys", required=True)
    p.add_argument("--atomic-energy-values", required=True)
    p.add_argument("--scaling", default="std_scaling", choices=["std_scaling", "rms_forces_scaling", "no_scaling"])
    p.add_argument("--use-reduced-cg", action="store_true")
    p.add_argument("--eig-tol", type=float, default=1e-10)
    p.add_argument("--abs-eig-tol", type=float, default=1e-12)
    args = p.parse_args()

    _set_global_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    atomic_numbers = parse_csv_numbers(args.atomic_energy_keys, int)
    e0_values = parse_csv_numbers(args.atomic_energy_values, float)
    scale, shift = _atomic_inter_scale_shift_from_h5(
        str(args.data_dir / "processed_train.h5"),
        atomic_energy_keys=atomic_numbers,
        atomic_energy_values=e0_values,
        scaling=args.scaling,
    )
    batch = move_batch(load_batch(args.data_dir, args.split, args.batch_size, args.batch_index), device, dtype)

    _set_global_seed(args.seed)
    mace = build_mace_model(args, atomic_numbers, e0_values, scale, shift, device, dtype, use_cueq=False)
    models = {"mace_e3nn": (mace, mace_outputs)}
    requested_modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    if "mace_cueq" in requested_modes:
        _set_global_seed(args.seed)
        models["mace_cueq"] = (
            build_mace_model(args, atomic_numbers, e0_values, scale, shift, device, dtype, use_cueq=True),
            mace_outputs,
        )
    for mode in requested_modes:
        if mode in {"mace_e3nn", "mace_cueq"}:
            continue
        backend = {"ictd_bridge_u": "ictd-bridge-u", "ictd_cueq": "cueq"}[mode]
        ictd = build_ictd_model(args, atomic_numbers, scale, shift, backend, device, dtype)
        convert_mace_to_ictd(mace.eval(), ictd)
        models[mode] = (ictd, ictd_outputs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "batch_index": args.batch_index,
        "max_force_components": args.max_force_components,
        "dtype": args.dtype,
        "scale": scale,
        "shift": shift,
        "atomic_numbers": atomic_numbers,
        "e0_values": e0_values,
        "modes": {},
    }
    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        model, fn = models[mode]
        spec = kernel_spectrum(model, fn, batch, atomic_numbers, e0_values, args)
        details["modes"][mode] = spec
        rows.append(
            {
                "mode": mode,
                "n_outputs": spec["n_outputs"],
                "n_params": spec["n_params"],
                "rank": spec["rank"],
                "rank_tol": spec["rank_tol"],
                "lambda_min_pos": spec["lambda_min_pos"],
                "lambda_max": spec["lambda_max"],
                "kappa_pos": spec["kappa_pos"],
                "trace": spec["trace"],
                "lambda_min_pos_over_trace": spec["lambda_min_pos_over_trace"],
                "lambda_max_over_trace": spec["lambda_max_over_trace"],
                "stable_gd_lr_bound": spec["stable_gd_lr_bound"],
                "train_lr_times_lambda_max": spec["train_lr_times_lambda_max"],
                "train_lr_times_lambda_min_pos": spec["train_lr_times_lambda_min_pos"],
            }
        )
        print(rows[-1])
    with (args.out_dir / "ntk_spectrum.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "ntk_spectrum.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Empirical NTK Spectrum Diagnostic",
        "",
        "Small-batch weighted-output kernel. Force components are deterministically sampled.",
        "",
        "| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['n_outputs']} | {row['n_params']} | {row['rank']} | "
            f"{row['lambda_min_pos']:.6g} | {row['lambda_max']:.6g} | {row['kappa_pos']:.6g} | "
            f"{row['trace']:.6g} | {row['lambda_min_pos_over_trace']:.6g} | "
            f"{row['lambda_max_over_trace']:.6g} | {row['train_lr_times_lambda_max']:.6g} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
