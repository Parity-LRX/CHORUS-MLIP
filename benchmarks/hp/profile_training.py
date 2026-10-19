"""Remote-only full checkpoint training-step benchmark, including force loss.

The deployment loader's constructor trainability is recorded before it freezes
parameters, then restored exactly. No model coefficient is frozen for speed.
The shared synthetic E/F objective and AdamW update measure implementation
throughput, not a change to the paper's training/accuracy protocol.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from profile_scaling import _NoExternalTP, geometry, neighbor_statistics
from qualify_and_bench import errors
from chorus.hp import configure_model
from chorus.interfaces.lammps_mliap import AtomForcesWrapper, LAMMPS_MLIAP_MFF
from chorus.synthetic import compute_energy_forces


def load_trainable(checkpoint):
    trainability = {}
    original = AtomForcesWrapper.__init__

    def record(self, model, *args, **kwargs):
        trainability.update({name: value.requires_grad for name, value in model.named_parameters()})
        original(self, model, *args, **kwargs)

    with patch.object(AtomForcesWrapper, "__init__", record):
        loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
            checkpoint, element_types=["H", "C", "N", "O"], device="cuda",
            ictd_fix_product_backend="ictd-bridge-u", ictd_tp_backend="pytorch",
            reuse_checkpoint_u=True)
    model = loaded.wrapper.model
    if not trainability or set(trainability) != {name for name, _ in model.named_parameters()}:
        raise RuntimeError("Did not capture the original constructor trainability")
    for name, value in model.named_parameters():
        value.requires_grad_(trainability[name])
    return model.train(), trainability


def objective(model, graph):
    energy, force, atomic = compute_energy_forces(model, graph, create_graph=True)
    loss = (energy/graph[0].shape[0]).square() + 100.*force.square().mean()
    return loss, energy, force, atomic


def gradients(model, graph):
    loss, energy, force, _ = objective(model, graph)
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    values = torch.autograd.grad(loss, [value for _, value in named], allow_unused=True)
    return (loss.detach(), energy.detach(), force.detach()), {
        name: None if value is None else value.detach().clone()
        for (name, _), value in zip(named, values)}


def benchmark(model, graph, warmup, repeats, profile_path=None, capture=False):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-5, weight_decay=0., capturable=capture)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = objective(model, graph)
        loss.backward()
        optimizer.step()
        return loss.detach()

    started = time.perf_counter()
    for _ in range(warmup):
        loss = step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-started
    if not bool(loss.isfinite()):
        raise RuntimeError("Nonfinite warmup training loss")
    run = step
    capture_validation = None
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            captured_loss = step()
        parameters = [p for p in model.parameters() if p.requires_grad]
        captured_gradients = [p.grad for p in parameters]
        weights = [p.detach().clone() for p in parameters]
        optimizer_tensors = [(value, value.clone()) for state in optimizer.state.values()
                             for value in state.values() if isinstance(value, torch.Tensor)]
        def restore():
            with torch.no_grad():
                for p, value in zip(parameters, weights):
                    p.copy_(value)
                for value, saved in optimizer_tensors:
                    value.copy_(saved)
        cg.replay()
        graph_weights = [p.detach().clone() for p in parameters]
        graph_loss = captured_loss.clone()
        restore()
        eager_loss = step()
        torch.testing.assert_close(graph_loss, eager_loss, atol=1e-4, rtol=1e-3)
        maximum = 0.
        for p, value in zip(parameters, graph_weights):
            if float((p.double()-value.double()).norm()/value.double().norm().clamp_min(1e-20)) > 1e-5:
                raise AssertionError("Captured parameter update relative L2 exceeds 1e-5")
            maximum = max(maximum, float((p-value).abs().max()))
        restore()
        for p, gradient in zip(parameters, captured_gradients):
            p.grad = gradient
        del weights, graph_weights, optimizer_tensors, graph_loss, eager_loss
        capture_validation = {"updated_weights_vs_eager": True, "parameter_max_abs": maximum,
                              "optimizer_state_live_on_replay": True}
        def run():
            cg.replay()
            return captured_loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples, gpu_samples = [], []
    first = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        loss = run()
        end.record()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-wall)*1000)
        gpu_samples.append(start.elapsed_time(end))
        if first is None:
            first = float(loss)
    if not bool(loss.isfinite()):
        raise RuntimeError("Nonfinite final training loss")
    result = {"median_ms": statistics.median(samples), "samples_ms": samples,
              "gpu_samples_ms": gpu_samples, "warmup_seconds": warmup_seconds,
              "atoms_per_second": graph[0].shape[0]*1000/statistics.median(samples),
              "peak_allocated": torch.cuda.max_memory_allocated(),
              "peak_reserved": torch.cuda.max_memory_reserved(),
              "first_timed_loss": first, "last_loss": float(loss),
              "optimizer_steps_before_profile": warmup+repeats+(2 if capture else 0),
              "timed_optimizer_updates": repeats, "capture": capture,
              "capture_validation": capture_validation}
    if profile_path is not None:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            run()
            torch.cuda.synchronize()
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=60))
    optimizer.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--neighbors", type=int, choices=(32, 50), default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261017)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, trainability = load_trainable(args.checkpoint)
    graph = geometry(args.atoms, model.max_radius, 20260728, args.neighbors)
    neighbors = neighbor_statistics(graph, args.neighbors)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    reference, ref_grad = gradients(model, graph)
    model.skip_input_validation = True
    model.preserve_edge_order = True
    options = dict(training=True, symmetric_contraction=True, checkpoint_contraction=True,
                   topology_backend="device")
    info = configure_model(model, policy="generated", **options)
    actual, actual_grad = gradients(model, graph)
    parity = {key: errors(got, ref) for key, got, ref in
              zip(("loss", "energy", "forces"), actual, reference)}
    for got, ref in zip(actual, reference):
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)
    parameter_parity = {}
    gradient_failures = []
    norm_failures = []
    for name, ref in ref_grad.items():
        got = actual_grad[name]
        if (ref is None) != (got is None):
            raise AssertionError(f"Gradient dependency changed: {name}")
        if ref is not None:
            if not bool(got.isfinite().all()):
                raise AssertionError(f"Nonfinite gradient: {name}")
            parameter_parity[name] = errors(got, ref)
            if ((float(ref.double().norm()) > 1e-8 and parameter_parity[name]["relative_l2"] > 1e-4)
                    or (float(ref.double().norm()) <= 1e-8 and parameter_parity[name]["max_abs"] > 1e-7)):
                norm_failures.append(name)
            try:
                torch.testing.assert_close(got, ref, atol=2e-4, rtol=3e-3)
            except AssertionError as error:
                gradient_failures.append({"parameter": name, "details": str(error),
                                          **parameter_parity[name]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".parity.json").write_text(json.dumps({
        "outputs": parity, "parameter_gradient_parity": parameter_parity,
        "strict_elementwise_failures": gradient_failures, "tensor_norm_failures": norm_failures}, indent=2)+"\n")
    if norm_failures:
        raise AssertionError(f"Parameter-gradient relative L2 exceeded 1e-4: {norm_failures}")
    # Independently check the optimizer consequence of every parameter gradient,
    # including small entries subject to cancellation in this lattice stress.
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    updates = []
    for recorded in (ref_grad, actual_grad):
        copied = [torch.nn.Parameter(p.detach().clone()) for _, p in named]
        for (name, _), p in zip(named, copied):
            p.grad = recorded[name]
        torch.optim.AdamW(copied, lr=1e-5, weight_decay=0.).step()
        updates.append(copied)
    update_max = 0.
    update_relative = 0.
    for ref, got in zip(*updates):
        difference = float((got.double()-ref.double()).norm()/ref.double().norm().clamp_min(1e-20))
        update_relative = max(update_relative, difference)
        if difference > 1e-5:
            raise AssertionError("AdamW updated parameter relative L2 exceeds 1e-5")
        update_max = max(update_max, float((got-ref).abs().max()))
    del updates, copied
    del reference, actual, ref_grad, actual_grad
    measurements = []
    # Reverse the paired order in the second round; each mode starts from the
    # exact original weights and empty optimizer state, not the other's update.
    for round_index, order in enumerate((("reference", "generated"), ("generated", "reference"))):
        for policy in order:
            model.load_state_dict(state, strict=True)
            configure_model(model, policy=policy, **options)
            measured = benchmark(model, graph, args.warmup, args.repeats,
                args.out.with_suffix(f".{policy}.profile.txt") if args.profile and round_index == 0 else None,
                capture=args.capture)
            measurements.append({"round": round_index, "policy": policy, **measured})
            print(json.dumps(measurements[-1]), flush=True)
    medians = {key: statistics.median(v["median_ms"] for v in measurements if v["policy"] == key)
               for key in ("reference", "generated")}
    result = {"status": "PASS", "checkpoint": args.checkpoint, "atoms": args.atoms,
              "edges": graph[3].numel(), "neighbors": neighbors, "plan": info,
              "trainability": trainability, "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "tp_coefficient_trainability": [layer.tp.weight.requires_grad for layer in model.interactions],
              "timed_work": "energy+conservative force+energy/force MSE+double backward+AdamW update",
              "loss": "(energy/N)^2+100*mean(force^2); fixed zero synthetic targets",
              "optimizer": {"type": "AdamW", "lr": 1e-5, "weight_decay": 0.},
              "parity": parity, "parameter_gradient_parity": parameter_parity,
              "gradient_criterion": "per-parameter relative L2<=1e-4; near-zero tensor max abs<=1e-7; AdamW updated parameter relative L2<=1e-5",
              "strict_elementwise_gradient_failures": gradient_failures,
              "optimizer_update_max_abs": update_max, "optimizer_update_max_relative_l2": update_relative,
              "optimizer_note": "First-step AdamW is sign-sensitive at cancelled near-zero gradients; raw max and tensor norms are retained",
              "capture": args.capture,
              "measurements": measurements, "median_ms": medians,
              "speedup": medians["reference"]/medians["generated"],
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "source_snapshot": str(Path(__file__).resolve().parents[2]),
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "arithmetic": "FP32; TF32 disabled; no external TP library"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"parameter_gradient_parity", "trainability", "measurements"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
