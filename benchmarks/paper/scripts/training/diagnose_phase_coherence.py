#!/usr/bin/env python3
"""Measure how much the learned off-diagonal U(1) density contributes.

This is a checkpoint diagnostic, not a training benchmark.  For each validation
batch it runs the same trained full-density model twice:

1. its native full Hermitian density;
2. a counterfactual diagonal-only (j=k) density with identical weights.

The interaction hook exposes the charged node sum and charged edge messages.
That lets us compare both the raw Hermitian density blocks and the final neutral
residual injected into the ordinary equivariant message.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chorus.data import H5Dataset, collate_fn_h5
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--elements", required=True, help="Comma-separated symbols")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def move_batch(batch: tuple[Any, ...], device: torch.device, dtype: torch.dtype):
    if len(batch) == 11:
        batch = batch[:10]
    pos, atomic_numbers, batch_idx, force, target_e, src, dst, shifts, cell, stress = batch
    return (
        pos.to(device=device, dtype=dtype),
        atomic_numbers.to(device=device, dtype=torch.long),
        batch_idx.to(device=device, dtype=torch.long),
        force.to(device=device, dtype=dtype),
        target_e.to(device=device, dtype=dtype),
        src.to(device=device, dtype=torch.long),
        dst.to(device=device, dtype=torch.long),
        shifts.to(device=device, dtype=dtype),
        cell.to(device=device, dtype=dtype),
        stress.to(device=device, dtype=dtype),
    )


def rms(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x.float().square())).item())


def ratio(num: float, den: float) -> float:
    return float(num / max(den, 1.0e-30))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    den = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    if float(den.item()) <= 1.0e-30:
        return 0.0
    return float(torch.dot(af, bf).div(den).item())


def scatter_sum_rows(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = values.new_zeros((int(size),) + tuple(values.shape[1:]))
    out.index_add_(0, index, values)
    return out


def split_irreps(x: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    blocks: dict[int, torch.Tensor] = {}
    start = 0
    for ell in range(int(lmax) + 1):
        width = int(channels) * (2 * ell + 1)
        blocks[ell] = x[..., start : start + width]
        start += width
    if start != x.shape[-1]:
        raise RuntimeError(
            f"SO(3) width mismatch: consumed {start}, tensor has {x.shape[-1]}"
        )
    return blocks


def summarize(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for record in records for key in record})
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([record[key] for record in records if key in record], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    elements = [item.strip() for item in args.elements.split(",") if item.strip()]

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    deployed = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=args.checkpoint,
        element_types=elements,
        device=str(device),
    )
    model = deployed.wrapper.model
    model.eval()
    model.skip_input_validation = False
    dtype = next(model.parameters()).dtype

    if not hasattr(model, "phase_adapters") or not model.phase_adapters:
        raise RuntimeError("checkpoint has no phase adapter")
    if str(model.ictd_fix_phase_scope) != "final":
        raise RuntimeError(
            "this same-orbital full/diagonal diagnostic currently expects phase_scope='final', "
            f"got {model.ictd_fix_phase_scope!r}"
        )
    phase_layer = max(int(key) for key in model.phase_adapters.keys())
    phase_key = str(phase_layer)
    adapter = model.phase_adapters[phase_key]
    interaction = model.interactions[phase_layer]
    original_density_pairs = str(model.ictd_fix_phase_density_pairs)
    if original_density_pairs not in {"full", "full-gated", "full-adaptive"}:
        raise RuntimeError(
            f"expected a Full-U1 checkpoint, got density_pairs={original_density_pairs!r}"
        )

    captured: dict[str, tuple[torch.Tensor, ...]] = {}

    def capture_interaction(_module, _inputs, output):
        if not isinstance(output, tuple):
            raise RuntimeError("phase interaction did not return a tuple")
        captured["output"] = output

    hook = interaction.register_forward_hook(capture_interaction)
    dataset = H5Dataset(prefix=args.split, data_dir=args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_h5,
        num_workers=0,
    )

    records: list[dict[str, float]] = []
    try:
        with torch.no_grad():
            for batch_number, cpu_batch in enumerate(loader):
                if batch_number >= args.max_batches:
                    break
                batch = move_batch(cpu_batch, device, dtype)
                pos, atomic_numbers, batch_idx, _, _, src, dst, shifts, cell, _ = batch
                compact_idx = model.atomic_number_to_index[atomic_numbers]
                if bool((compact_idx < 0).any().item()):
                    raise ValueError("validation batch contains an element absent from checkpoint")

                model.ictd_fix_phase_density_pairs = "full"
                captured.clear()
                energy_full = model(pos, atomic_numbers, batch_idx, src, dst, shifts, cell)
                full_out = captured["output"]
                if len(full_out) != 3:
                    raise RuntimeError(f"full interaction returned {len(full_out)} values, expected 3")
                neutral_full, _, charged_nodes = full_out
                delta_full = adapter.forward_doublet(
                    charged_nodes,
                    node_attrs=None,
                    node_type_idx=compact_idx,
                )

                model.ictd_fix_phase_density_pairs = "diagonal"
                captured.clear()
                energy_diagonal = model(pos, atomic_numbers, batch_idx, src, dst, shifts, cell)
                diagonal_out = captured["output"]
                if len(diagonal_out) != 5:
                    raise RuntimeError(
                        f"diagonal interaction returned {len(diagonal_out)} values, expected 5"
                    )
                (
                    neutral_diagonal,
                    _,
                    charged_nodes_diagonal,
                    edge_orbital,
                    edge_norm_sq,
                ) = diagonal_out

                if getattr(model, "preserve_edge_order", False):
                    used_dst = dst
                else:
                    used_dst = dst[torch.argsort(dst)]
                delta_diagonal = adapter.forward_diagonal_edges_factorized(
                    edge_orbital,
                    edge_norm_sq,
                    edge_dst=used_dst,
                    num_nodes=pos.shape[0],
                    node_attrs=None,
                    node_type_idx=compact_idx,
                )

                delta_off = delta_full - delta_diagonal
                density_full = adapter.hermitian_blocks_doublet(charged_nodes)
                density_diagonal = adapter._diagonal_blocks_factorized(
                    edge_orbital,
                    edge_norm_sq,
                    edge_dst=used_dst,
                    num_nodes=pos.shape[0],
                )

                neutral_rms = rms(neutral_full)
                full_rms = rms(delta_full)
                diagonal_rms = rms(delta_diagonal)
                off_rms = rms(delta_off)
                record: dict[str, float] = {
                    "batch": float(batch_number),
                    "num_graphs": float(int(batch_idx.max().item()) + 1),
                    "num_nodes": float(pos.shape[0]),
                    "num_edges": float(src.numel()),
                    "mean_neighbors": float(src.numel() / max(pos.shape[0], 1)),
                    "neutral_message_rms": neutral_rms,
                    "delta_full_rms": full_rms,
                    "delta_diagonal_rms": diagonal_rms,
                    "delta_offdiagonal_rms": off_rms,
                    "delta_full_over_neutral": ratio(full_rms, neutral_rms),
                    "delta_diagonal_over_neutral": ratio(diagonal_rms, neutral_rms),
                    "delta_offdiagonal_over_neutral": ratio(off_rms, neutral_rms),
                    "delta_offdiagonal_over_diagonal": ratio(off_rms, diagonal_rms),
                    "delta_diag_off_cosine": cosine(delta_diagonal, delta_off),
                    "neutral_repeat_max_abs": float(
                        (neutral_full - neutral_diagonal).abs().max().item()
                    ),
                    "charged_repeat_max_abs": float(
                        (charged_nodes - charged_nodes_diagonal).abs().max().item()
                    ),
                    "counterfactual_atomic_energy_mae": float(
                        (energy_full - energy_diagonal).abs().float().mean().item()
                    ),
                    "counterfactual_atomic_energy_rms": rms(
                        energy_full - energy_diagonal
                    ),
                }

                node_power = charged_nodes.float().square().sum(dim=(-2, -1))
                edge_power = (
                    edge_orbital.float().square().sum(dim=-1)
                    * edge_norm_sq.float()
                )
                diagonal_power = scatter_sum_rows(
                    edge_power.unsqueeze(-1), used_dst, pos.shape[0]
                ).squeeze(-1)
                coherence_per_node = node_power / diagonal_power.clamp_min(1.0e-30)
                finite = torch.isfinite(coherence_per_node) & (diagonal_power > 1.0e-20)
                coherence_valid = coherence_per_node[finite]
                record["orbital_coherence_ratio_global"] = float(
                    node_power.sum().div(diagonal_power.sum().clamp_min(1.0e-30)).item()
                )
                record["orbital_interference_fraction_global"] = (
                    record["orbital_coherence_ratio_global"] - 1.0
                )
                if coherence_valid.numel():
                    q = torch.quantile(
                        coherence_valid,
                        torch.tensor(
                            [0.1, 0.5, 0.9],
                            device=coherence_valid.device,
                            dtype=coherence_valid.dtype,
                        ),
                    )
                    record["orbital_coherence_node_mean"] = float(
                        coherence_valid.mean().item()
                    )
                    record["orbital_coherence_node_q10"] = float(q[0].item())
                    record["orbital_coherence_node_q50"] = float(q[1].item())
                    record["orbital_coherence_node_q90"] = float(q[2].item())

                delta_full_l = split_irreps(delta_full, adapter.channels, adapter.lmax)
                delta_diagonal_l = split_irreps(
                    delta_diagonal, adapter.channels, adapter.lmax
                )
                delta_off_l = split_irreps(delta_off, adapter.channels, adapter.lmax)
                neutral_l = split_irreps(
                    neutral_full, adapter.channels, adapter.lmax
                )
                for ell in range(adapter.lmax + 1):
                    raw_off = density_full[ell] - density_diagonal[ell]
                    raw_diag_rms = rms(density_diagonal[ell])
                    raw_off_rms = rms(raw_off)
                    l_neutral_rms = rms(neutral_l[ell])
                    l_full_rms = rms(delta_full_l[ell])
                    l_diag_rms = rms(delta_diagonal_l[ell])
                    l_off_rms = rms(delta_off_l[ell])
                    prefix = f"L{ell}"
                    record[f"{prefix}_density_off_over_diag"] = ratio(
                        raw_off_rms, raw_diag_rms
                    )
                    record[f"{prefix}_delta_full_over_neutral"] = ratio(
                        l_full_rms, l_neutral_rms
                    )
                    record[f"{prefix}_delta_diagonal_over_neutral"] = ratio(
                        l_diag_rms, l_neutral_rms
                    )
                    record[f"{prefix}_delta_off_over_neutral"] = ratio(
                        l_off_rms, l_neutral_rms
                    )
                    record[f"{prefix}_delta_off_over_diag"] = ratio(
                        l_off_rms, l_diag_rms
                    )
                    record[f"{prefix}_delta_diag_off_cosine"] = cosine(
                        delta_diagonal_l[ell], delta_off_l[ell]
                    )

                records.append(record)
                print(
                    f"batch={batch_number} nodes={pos.shape[0]} edges={src.numel()} "
                    f"Cpsi={record['orbital_coherence_ratio_global']:.4f} "
                    f"|off|/|diag|={record['delta_offdiagonal_over_diagonal']:.4f} "
                    f"|full|/|trunk|={record['delta_full_over_neutral']:.4f}",
                    flush=True,
                )
    finally:
        model.ictd_fix_phase_density_pairs = original_density_pairs
        hook.remove()

    if not records:
        raise RuntimeError("no validation batches were processed")
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "split": args.split,
        "elements": elements,
        "device": str(device),
        "dtype": str(dtype),
        "phase_layer": phase_layer,
        "phase_adapter_lmax": int(adapter.lmax),
        "phase_density_rank": int(adapter.density_rank),
        "trained_density_pairs": original_density_pairs,
        "trained_phase_normalization": str(
            model.ictd_fix_phase_normalization
        ),
        "learned_coherence_scale": (
            adapter.effective_coherence_scale().detach().float().cpu().tolist()
            if (
                adapter.coherence_scale is not None
                or adapter.coherence_logit is not None
            )
            else None
        ),
        "learned_coherence_logit": (
            adapter.coherence_logit.detach().float().cpu().tolist()
            if adapter.coherence_logit is not None
            else None
        ),
        "num_batches": len(records),
        "definitions": {
            "orbital_coherence_ratio_global": "sum_i ||sum_j psi_ij||^2 / sum_ij ||psi_ij||^2",
            "density_off": "rho_full - sum_j rho_jj, before learned output mixing",
            "delta_offdiagonal": "DeltaA_full - DeltaA_diagonal, after learned output mixing",
            "neutral_message": "ordinary final-layer pre-product equivariant message before U1 injection",
        },
        "summary": summarize(records),
        "batches": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
