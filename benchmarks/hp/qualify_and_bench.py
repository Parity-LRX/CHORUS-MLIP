"""Same-weight CHORUS/ICTC-O3-HP parity and paired full energy/force timings.

Run only on the user-designated remote GPU. Results are synthetic implementation
qualification, not a retraining/accuracy or MD-validation experiment.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from chorus.hp import configure_model
from chorus.synthetic import build_model, compute_energy_forces, make_fixed_graph


def record(model, graph):
    energy, forces, atomic = compute_energy_forces(model, graph, create_graph=True)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    loss = energy.square()*0.01 + forces.square().sum()
    grad = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    return [energy.detach(), forces.detach(), atomic.detach()], {
        name: None if g is None else g.detach().clone() for (name, _), g in zip(named, grad)}


def errors(actual, reference):
    # Reporting only: FP32 norms can overflow on stress inputs even when every
    # entry is finite. This does not change model arithmetic or timed execution.
    actual = actual.detach().to(torch.float64)
    reference = reference.detach().to(torch.float64)
    delta = (actual-reference).abs()
    return {"max_abs": float(delta.max()) if delta.numel() else 0.0,
            "relative_l2": float((actual-reference).norm()/reference.norm().clamp_min(1e-20))}


def measure(model, graph, warmup, iterations):
    for _ in range(warmup):
        compute_energy_forces(model, graph, create_graph=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        compute_energy_forces(model, graph, create_graph=False)
    end.record()
    torch.cuda.synchronize()
    return {"ms": start.elapsed_time(end)/iterations,
            "wall_ms": (time.perf_counter()-wall)*1000/iterations,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--atoms", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--scope", choices=("final", "persistent"), default="persistent")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-density-fusion", action="store_true")
    parser.add_argument("--no-stream-fusion", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261017)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(
        channels=args.channels, lmax=args.lmax, num_interaction=args.layers,
        route="baseline", product_backend="ictd-bridge-u", dtype=torch.float32,
        device=torch.device("cuda"), correlation=args.correlation,
        ictd_fix_freeze_conv_tp_weight=True, ictd_fix_phase_mode="final-full-l-residual",
        ictd_fix_phase_amplitude="softplus", ictd_fix_phase_placement="pre-product-full-l",
        ictd_fix_phase_density_pairs="full-nonlinear", ictd_fix_phase_density_rank=8,
        ictd_fix_phase_scope=args.scope,
    )
    graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=args.neighbors,
                             dtype=torch.float32, device=torch.device("cuda"), seed=91)
    # Use a compact valid geometry; topology includes repeated/unsorted indices.
    graph = (graph[0]*0.15,) + graph[1:]
    reference, ref_grad = record(model, graph)
    options = dict(stream_fusion=not args.no_stream_fusion, density_fusion=not args.no_density_fusion)
    info = configure_model(model, training=True, policy="generated", **options)
    actual, grad = record(model, graph)
    parity = {key: errors(got, expected) for key, got, expected in
              zip(("energy", "forces", "atomic_energy"), actual, reference)}
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, atol=3e-5, rtol=1e-3)
    gradient_errors = {}
    for name, expected in ref_grad.items():
        got = grad[name]
        if (got is None) != (expected is None):
            raise AssertionError(f"parameter gradient connectivity changed: {name}")
        if got is not None:
            gradient_errors[name] = errors(got, expected)
            torch.testing.assert_close(got, expected, atol=5e-5, rtol=2e-3, msg=name)
    model.eval().requires_grad_(False)
    samples = {"reference": [], "generated": []}
    for repeat in range(args.repeats):
        order = ("reference", "generated") if repeat % 2 == 0 else ("generated", "reference")
        for policy in order:
            configure_model(model, policy=policy, **options)
            samples[policy].append(measure(model, graph, args.warmup, args.iterations))
    median = {key: statistics.median(s["ms"] for s in values) for key, values in samples.items()}
    result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "device": torch.cuda.get_device_name(), "torch": torch.__version__, "plan": info,
              "parity": parity, "force_loss_gradients": gradient_errors,
              "paired_full_force_samples": samples, "median_ms": median,
              "speedup": median["reference"]/median["generated"], "status": "PASS"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("force_loss_gradients", "paired_full_force_samples")}, indent=2))


if __name__ == "__main__":
    main()
