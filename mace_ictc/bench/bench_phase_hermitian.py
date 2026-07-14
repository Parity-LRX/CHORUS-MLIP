#!/usr/bin/env python3
"""Matched 4090 benchmark for the PEMP Hermitian phase branch.

This reuses the fixed-edge whole-model workload and timing functions from the
paper benchmark.  It compares the existing ICTC model against unit-amplitude
and learned-amplitude phase residuals with the same backend and graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from mace_ictc.bench.bench_mace_ictc_vs_mace import (
    AngularConfig,
    benchmark_ictc_inference_aoti,
    benchmark_ictc_inference_eager,
    benchmark_ictc_training,
    build_ictc,
    dtype_from_name,
    make_graph,
    parse_int_list,
    sync,
)
from mace_ictc.training.train_loop import disable_tf32


PHASE_MODES = {
    "baseline": ("none", "unit"),
    "phase_unit": ("final-scalar-residual", "unit"),
    "phase_softplus": ("final-scalar-residual", "softplus"),
}


def _time_mode(
    *,
    label: str,
    phase_mode: str,
    phase_amplitude: str,
    args: argparse.Namespace,
    cfg: AngularConfig,
    graph,
    atoms: int,
    task: str,
) -> dict:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = build_ictc(
        product_backend=args.product_backend,
        cfg=cfg,
        channels=args.channels,
        num_interactions=args.num_interactions,
        correlation=args.correlation,
        dtype=dtype,
        device=device,
        use_reduced_cg=bool(args.use_reduced_cg),
        phase_mode=phase_mode,
        phase_hidden_channels=args.phase_hidden_channels,
        phase_residual_scale_init=args.phase_scale_init,
        phase_amplitude=phase_amplitude,
    )
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    compile_s = ""
    cache_entries = ""
    artifact = ""
    loss = ""
    if task == "train_eager":
        time_ms, loss, cache_entries, _ = benchmark_ictc_training(
            model,
            graph,
            device=device,
            dtype=dtype,
            lr=args.lr,
            warmup=args.train_warmup,
            iters=args.train_iters,
            makefx=False,
            require_makefx=False,
        )
    elif task == "train_makefx":
        time_ms, loss, cache_entries, compile_s = benchmark_ictc_training(
            model,
            graph,
            device=device,
            dtype=dtype,
            lr=args.lr,
            warmup=args.train_warmup,
            iters=args.train_iters,
            makefx=True,
            require_makefx=True,
        )
    elif task == "inference_eager":
        time_ms = benchmark_ictc_inference_eager(
            model,
            graph,
            device=device,
            warmup=args.infer_warmup,
            iters=args.infer_iters,
        )
    elif task == "inference_aoti":
        package_dir = Path(args.out_dir) / "aoti_packages"
        package_dir.mkdir(parents=True, exist_ok=True)
        time_ms, compile_s, artifact = benchmark_ictc_inference_aoti(
            model,
            graph,
            device=device,
            out_dir=package_dir,
            stem=f"{label}_{args.product_backend}_l{args.hidden_lmax}_e{args.max_ell}_n{atoms}",
            warmup=args.infer_warmup,
            iters=args.infer_iters,
            export_strict=args.product_backend != "cueq",
        )
    else:
        raise ValueError(f"unknown task {task!r}")
    sync(device)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else ""
    )
    return {
        "task": task,
        "mode": label,
        "phase_mode": phase_mode,
        "phase_amplitude": phase_amplitude,
        "product_backend": args.product_backend,
        "atoms": atoms,
        "edges": atoms * args.avg_degree,
        "channels": args.channels,
        "hidden_lmax": args.hidden_lmax,
        "max_ell": args.max_ell,
        "parameters": params,
        "time_ms": float(time_ms),
        "overhead_vs_baseline": "",
        "peak_memory_mb": peak_mb,
        "compile_s": compile_s,
        "cache_entries": cache_entries,
        "loss": loss,
        "artifact": artifact,
        "status": "ok",
        "error": "",
    }


def _annotate_overhead(rows: list[dict]) -> None:
    baselines = {
        (row["task"], row["atoms"]): float(row["time_ms"])
        for row in rows
        if row["mode"] == "baseline" and row["status"] == "ok"
    }
    for row in rows:
        baseline = baselines.get((row["task"], row["atoms"]))
        if baseline is not None and row["status"] == "ok":
            row["overhead_vs_baseline"] = float(row["time_ms"]) / baseline - 1.0


def _write(rows: list[dict], meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase_hermitian_benchmark.csv"
    json_path = out_dir / "phase_hermitian_benchmark.json"
    md_path = out_dir / "phase_hermitian_benchmark.md"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    with md_path.open("w") as handle:
        handle.write("# PEMP Hermitian phase matched benchmark\n\n")
        for key, value in meta.items():
            handle.write(f"- {key}: `{value}`\n")
        handle.write(
            "\n| task | mode | atoms | parameters | ms | overhead | peak MiB | status |\n"
            "|---|---|---:|---:|---:|---:|---:|---|\n"
        )
        for row in rows:
            overhead = row["overhead_vs_baseline"]
            overhead_text = "" if overhead == "" else f"{100.0 * float(overhead):.1f}%"
            handle.write(
                f"| {row['task']} | {row['mode']} | {row['atoms']} | "
                f"{row['parameters']} | {row['time_ms']:.4f} | {overhead_text} | "
                f"{row['peak_memory_mb']} | {row['status']} |\n"
            )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--product-backend", default="ictd-bridge-u", choices=["ictd-bridge-u", "ictd-pure-u", "cueq"])
    parser.add_argument("--atoms-list", default="128,512")
    parser.add_argument("--avg-degree", type=int, default=50)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--hidden-lmax", type=int, default=1)
    parser.add_argument("--max-ell", type=int, default=2)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--use-reduced-cg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase-hidden-channels", type=int, default=32)
    parser.add_argument("--phase-scale-init", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--train-warmup", type=int, default=3)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--infer-warmup", type=int, default=10)
    parser.add_argument("--infer-iters", type=int, default=50)
    parser.add_argument("--include-makefx", action="store_true")
    parser.add_argument("--include-aoti", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="benchmarks/paper/results/phase/phase_hermitian_4090",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    disable_tf32()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    cfg = AngularConfig(args.hidden_lmax, args.max_ell)
    tasks = ["train_eager", "inference_eager"]
    if args.include_makefx:
        tasks.append("train_makefx")
    if args.include_aoti:
        tasks.append("inference_aoti")
    rows: list[dict] = []
    started = time.time()
    for atoms in parse_int_list(args.atoms_list):
        graph = make_graph(
            atoms=atoms,
            avg_degree=args.avg_degree,
            dtype=dtype,
            device=device,
            seed=args.seed + atoms,
        )
        for task in tasks:
            for label, (phase_mode, amplitude) in PHASE_MODES.items():
                try:
                    row = _time_mode(
                        label=label,
                        phase_mode=phase_mode,
                        phase_amplitude=amplitude,
                        args=args,
                        cfg=cfg,
                        graph=graph,
                        atoms=atoms,
                        task=task,
                    )
                except Exception as exc:  # keep the matched sweep auditable
                    row = {
                        "task": task,
                        "mode": label,
                        "phase_mode": phase_mode,
                        "phase_amplitude": amplitude,
                        "product_backend": args.product_backend,
                        "atoms": atoms,
                        "edges": atoms * args.avg_degree,
                        "channels": args.channels,
                        "hidden_lmax": args.hidden_lmax,
                        "max_ell": args.max_ell,
                        "parameters": "",
                        "time_ms": "",
                        "overhead_vs_baseline": "",
                        "peak_memory_mb": "",
                        "compile_s": "",
                        "cache_entries": "",
                        "loss": "",
                        "artifact": "",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1000],
                    }
                rows.append(row)
                print(json.dumps(row), flush=True)
    _annotate_overhead(rows)
    meta = {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "tf32": False,
        "product_backend": args.product_backend,
        "elapsed_s": time.time() - started,
        "command": " ".join(sys.argv),
    }
    _write(rows, meta, Path(args.out_dir))
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
