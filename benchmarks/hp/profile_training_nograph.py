"""Remote full E/F training steps, with CUDA Graph explicitly forbidden.

Every iteration builds a fresh autograd graph and updates all original trainable
parameters. Prepared input batches change coordinates, species and edge order;
--variable-atoms also changes N/E between steps. Data preparation is not timed.
"""
import argparse
import cProfile
import io
import json
import os
import pstats
from pathlib import Path
from contextlib import nullcontext
import statistics
import time

import torch

from no_cuda_graph import forbid_cuda_graph
from profile_scaling import geometry, neighbor_statistics
from profile_training import load_trainable, objective
from qualify_and_bench import errors
from chorus.hp import configure_model
from checkpoint_reference_tp import checkpoint_original_tp


OPTIONS = dict(training=True, symmetric_contraction=True,
               checkpoint_contraction=True, topology_backend="device")


def record(model, graph, clear_cuda_cache=False):
    loss, energy, force, atomic = objective(model, graph)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if clear_cuda_cache:
        # Untimed original oracle only: release forward temporaries before its
        # very wide force-loss backward. No tensor or derivative is changed.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    grad = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    return tuple(v.detach() for v in (loss, energy, force, atomic)), {
        name: None if g is None else g.detach().clone() for (name, _), g in zip(named, grad)}


def qualify(model, batches, report_path=None, reference_tp_chunk_edges=0, reference_save_on_cpu=False,
            reference_empty_cache=False, reference_sc_chunk_nodes=0, reference_sc_chunk_channels=0):
    reports = []
    for index, graph in enumerate(batches):
        configure_model(model, policy="reference", **OPTIONS)
        with (torch.autograd.graph.save_on_cpu() if reference_save_on_cpu else nullcontext()), \
                checkpoint_original_tp(model, reference_tp_chunk_edges, reference_sc_chunk_nodes,
                                       reference_sc_chunk_channels):
            ref_values, ref_grad = record(model, graph, clear_cuda_cache=reference_empty_cache)
        plan = configure_model(model, policy="generated", **OPTIONS)
        values, grad = record(model, graph)
        output_errors = {}
        for name, got, ref in zip(("loss", "energy", "forces", "atomic_energy"), values, ref_values):
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)
            output_errors[name] = errors(got, ref)
        gradient_errors, strict_failures = {}, []
        for name, ref in ref_grad.items():
            got = grad[name]
            if (ref is None) != (got is None):
                raise AssertionError(f"Gradient dependency changed: {name}")
            if ref is None:
                continue
            if not bool(got.isfinite().all()):
                raise AssertionError(f"Nonfinite parameter gradient: {name}")
            delta = errors(got, ref)
            gradient_errors[name] = delta
            norm = float(ref.double().norm())
            if ((norm > 1e-8 and delta["relative_l2"] > 1e-4)
                    or (norm <= 1e-8 and delta["max_abs"] > 1e-7)):
                raise AssertionError(f"Original tensor-gradient criterion failed: {name}: {delta}")
            try:
                torch.testing.assert_close(got, ref, atol=2e-4, rtol=3e-3)
            except AssertionError as error:
                strict_failures.append({"parameter": name, "details": str(error), **delta})
        # First-step optimizer consequence is checked without modifying the model.
        updated = []
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        for recorded in (ref_grad, grad):
            copied = [torch.nn.Parameter(p.detach().clone()) for _, p in named]
            for (name, _), p in zip(named, copied):
                p.grad = recorded[name]
            torch.optim.AdamW(copied, lr=1e-5, weight_decay=0.).step()
            updated.append(copied)
        update_errors = [errors(got, ref) for ref, got in zip(*updated)]
        maximum = max(v["relative_l2"] for v in update_errors)
        if maximum > 1e-5:
            raise AssertionError(f"AdamW parameter relative L2 exceeded original 1e-5: {maximum}")
        reports.append({"batch": index, "atoms": graph[0].shape[0], "edges": graph[3].numel(),
            "outputs": output_errors, "parameter_gradient_parity": gradient_errors,
            "strict_elementwise_gradient_failures": strict_failures,
            "optimizer_update_max_relative_l2": maximum, "plan": plan})
        print(json.dumps({"stage": "qualification", "batch": index,
            "strict_elementwise_gradient_failures": len(strict_failures)}), flush=True)
        del updated, copied, grad, ref_grad, values, ref_values
        if report_path is not None:
            report_path.write_text(json.dumps({
                "status": "PASS_WITH_ELEMENTWISE_EXCEPTIONS" if any(
                    row["strict_elementwise_gradient_failures"] for row in reports) else "PASS",
                "strict_elementwise_gradient_pass": not any(
                    row["strict_elementwise_gradient_failures"] for row in reports),
                "qualification": reports,
                "criterion": {"output_atol": 1e-4, "output_rtol": 1e-3,
                              "gradient_atol": 2e-4, "gradient_rtol": 3e-3},
                "performance_qualified": False,
            }, indent=2) + "\n")
    return reports


def measure(model, batches, warmup, repeats, profile_path, parameter_targets_only=False):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-5, weight_decay=0.)

    def step(graph):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = objective(model, graph)
        if parameter_targets_only:
            from chorus.hp import backward_parameters
            backward_parameters(loss, model)
        else:
            loss.backward()
        optimizer.step()
        return loss.detach()

    started = time.perf_counter()
    # Warm every input shape, not only the first shape of the batch sequence.
    for _ in range(warmup):
        for graph in batches:
            loss = step(graph)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    if not bool(loss.isfinite()):
        raise RuntimeError("Nonfinite warmup training loss")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for iteration in range(repeats):
        graph = batches[iteration % len(batches)]
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        loss = step(graph)
        end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000
        if not bool(loss.isfinite()):
            raise RuntimeError("Nonfinite timed training loss")
        samples.append({"batch": iteration % len(batches), "atoms": graph[0].shape[0],
            "edges": graph[3].numel(), "wall_ms": wall_ms,
            "gpu_ms": begin.elapsed_time(end), "loss": float(loss)})
    result = {"median_ms": statistics.median(v["wall_ms"] for v in samples),
        "median_gpu_ms": statistics.median(v["gpu_ms"] for v in samples), "samples": samples,
        "atoms_per_second_sequence": sum(v["atoms"] for v in samples) * 1000 / sum(v["wall_ms"] for v in samples),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(), "warmup_seconds": warmup_seconds,
        "optimizer": {"type": "AdamW", "lr": 1e-5, "weight_decay": 0., "capturable": False},
        "fresh_autograd_graph_per_step": True, "timed_optimizer_updates": repeats}
    result["backward_targets"] = "all_trainable_parameters" if parameter_targets_only else "all_leaves"
    if not all(bool(p.isfinite().all()) and (p.grad is None or bool(p.grad.isfinite().all()))
               for p in model.parameters() if p.requires_grad):
        raise RuntimeError("Nonfinite parameter or gradient after timed full training steps")
    result["post_timing_parameters_and_gradients_finite"] = True
    if profile_path:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            step(batches[0])
            torch.cuda.synchronize()
        profile_path.write_text(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=80))
        prof.export_chrome_trace(str(profile_path.with_suffix(".trace.json")))
        with cProfile.Profile() as python_profile:
            step(batches[0])
            torch.cuda.synchronize()
        stream = io.StringIO()
        pstats.Stats(python_profile, stream=stream).strip_dirs().sort_stats("tottime").print_stats(70)
        profile_path.with_suffix(".python.txt").write_text(stream.getvalue())
    optimizer.zero_grad(set_to_none=True)
    return result


def main(graph_policy):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--variable-atoms", type=int, nargs="+")
    parser.add_argument("--neighbors", type=int, choices=(32, 50), default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--policy", choices=("reference", "generated"), default="generated")
    parser.add_argument("--phase-context-mixed-fusion", action="store_true")
    parser.add_argument("--phase-coefficient-mixed-fusion", action="store_true")
    parser.add_argument("--gate-coefficient-fusion", action="store_true")
    parser.add_argument("--radial-coefficient-fusion", action="store_true")
    parser.add_argument("--training-species-fusion", action="store_true")
    parser.add_argument("--training-mixed-threads", type=int, choices=(0, 32, 64, 128), default=0)
    parser.add_argument("--training-split-first", action="store_true")
    parser.add_argument("--training-mixed-grouped", action="store_true")
    parser.add_argument("--training-mixed-compact", action="store_true")
    parser.add_argument("--training-mixed-cache", type=int, choices=(0, 8, 16, 32, 64), default=0)
    parser.add_argument("--training-first-cache", action="store_true")
    parser.add_argument("--training-first-schedule", default="legacy",
        choices=("legacy", "pruned", "grouped", "compact", "compact_grouped"))
    parser.add_argument("--training-mixed-fuse-jvp", action="store_true")
    parser.add_argument("--training-radial-tp-fusion", action="store_true")
    parser.add_argument("--training-radial-cached", action="store_true")
    parser.add_argument("--training-radial-first-bounded", action="store_true")
    parser.add_argument("--radial-tile-edges", type=int, default=0)
    parser.add_argument("--training-terminal-fusion", action="store_true")
    parser.add_argument("--parameter-targets-only", action="store_true")
    parser.add_argument("--training-csr-splits", type=int, choices=(0, 1, 2, 4, 8), default=0)
    parser.add_argument("--training-mixed-destination", action="store_true")
    parser.add_argument("--training-destination-first", type=int, choices=(0, 1, 2, 4), default=0)
    parser.add_argument("--reference-tp-chunk-edges", type=int, default=0,
                        help="Untimed original-formula oracle: checkpoint edge chunks to bound its memory")
    parser.add_argument("--reference-sc-chunk-nodes", type=int, default=0,
                        help="Untimed original SC only: checkpoint independent node slices without changing its formula")
    parser.add_argument("--reference-sc-chunk-channels", type=int, default=0,
                        help="Untimed original SC only: use differentiable views of every channel's original parameters")
    parser.add_argument("--reference-save-on-cpu", action="store_true",
                        help="Offload saved original-oracle tensors only; never applied to a timed step")
    parser.add_argument("--reference-empty-cache", action="store_true",
                        help="Synchronize/release unused forward memory before the untimed original backward only")
    parser.add_argument("--measurement-only", action="store_true",
                        help="Explicit capacity/speed screen without a full large-N original oracle; not formal qualification")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase_context_mixed_fusion:
        OPTIONS["phase_context_mixed_fusion"] = True
    if args.phase_coefficient_mixed_fusion:
        OPTIONS["phase_coefficient_mixed_fusion"] = True
    if args.gate_coefficient_fusion:
        OPTIONS["gate_coefficient_fusion"] = True
    if args.radial_coefficient_fusion:
        OPTIONS["radial_coefficient_fusion"] = True
    if args.training_species_fusion:
        OPTIONS["training_species_fusion"] = True
    if args.training_mixed_threads:
        OPTIONS["training_mixed_threads"] = args.training_mixed_threads
    if args.training_split_first:
        OPTIONS["training_split_first"] = True
    if args.training_mixed_grouped:
        OPTIONS["training_mixed_grouped"] = True
    if args.training_mixed_compact:
        OPTIONS["training_mixed_compact"] = True
    if args.training_mixed_cache:
        OPTIONS["training_mixed_cache"] = args.training_mixed_cache
    if args.training_first_cache:
        OPTIONS["training_first_cache"] = True
    if args.training_first_schedule != "legacy":
        OPTIONS["training_first_schedule"] = args.training_first_schedule
    if args.training_mixed_fuse_jvp:
        OPTIONS["training_mixed_fuse_jvp"] = True
    if args.training_radial_tp_fusion:
        OPTIONS["training_radial_tp_fusion"] = True
    if args.training_radial_cached:
        OPTIONS["training_radial_cached"] = True
    if args.training_radial_first_bounded:
        OPTIONS["training_radial_first_bounded"] = True
    if args.radial_tile_edges:
        OPTIONS["radial_tile_edges"] = args.radial_tile_edges
    if args.training_terminal_fusion:
        OPTIONS["training_terminal_fusion"] = True
    if args.training_csr_splits:
        OPTIONS["training_csr_splits"] = args.training_csr_splits
    if args.training_mixed_destination:
        OPTIONS["training_mixed_destination"] = True
    if args.training_destination_first:
        OPTIONS["training_destination_first"] = args.training_destination_first
    if min(args.warmup, args.repeats, args.rounds) < 1:
        raise ValueError("warmup, repeats and rounds must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261017)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model, trainability = load_trainable(args.checkpoint)
    model.skip_input_validation = True
    model.preserve_edge_order = True
    sizes = args.variable_atoms or [args.atoms] * 3
    batches, identities = [], []
    for index, n in enumerate(sizes):
        graph = list(geometry(n, model.max_radius, 20260728 + index, args.neighbors))
        neighbors = neighbor_statistics(graph, args.neighbors)
        graph[1] = graph[1].roll(index)
        order = torch.arange(graph[3].numel(), device="cuda").roll(index * 17)
        for field in (3, 4, 5):
            graph[field] = graph[field].index_select(0, order)
        batches.append(tuple(graph))
        identities.append({"atoms": n, "edges": graph[3].numel(), "neighbors": neighbors,
            "geometry_seed": 20260728 + index, "species_roll": index, "edge_roll": index * 17})
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    qualification = [] if args.measurement_only else qualify(
        model, batches, args.out.with_suffix(".qualification.json"),
        args.reference_tp_chunk_edges, args.reference_save_on_cpu, args.reference_empty_cache,
        args.reference_sc_chunk_nodes, args.reference_sc_chunk_channels)
    measurements = []
    for repeat in range(args.rounds):
        model.load_state_dict(state, strict=True)
        plan = configure_model(model, policy=args.policy, **OPTIONS)
        measured = measure(model, batches, args.warmup, args.repeats,
            None, args.parameter_targets_only)
        measurements.append(measured)
        print(json.dumps({"stage": "measurement", "round": repeat, "tag": args.tag,
                          "median_ms": measured["median_ms"]}), flush=True)
    strict_failures = sum(len(v["strict_elementwise_gradient_failures"]) for v in qualification)
    result = {"status": ("DIAGNOSTIC_NO_LARGE_REFERENCE" if args.measurement_only else
                         "PASS_WITH_ELEMENTWISE_EXCEPTIONS" if strict_failures else "PASS"),
        "performance_qualified": not args.measurement_only,
        "strict_elementwise_gradient_pass": None if args.measurement_only else not strict_failures,
        "acceptance_policy": "User permits small elementwise roundoff; original tensor-norm, output and optimizer criteria remain mandatory",
        "tag": args.tag, "policy": args.policy, "plan": plan,
        "checkpoint": args.checkpoint, "graph_execution_policy": graph_policy,
        "batches": identities, "variable_atom_count": len(set(sizes)) > 1,
        "trainability": trainability, "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "tp_coefficient_trainability": [layer.tp.weight.requires_grad for layer in model.interactions],
        "qualification": qualification, "strict_elementwise_gradient_failures": strict_failures,
        "measurements": measurements,
        "median_ms": statistics.median(v["median_ms"] for v in measurements),
        "timed_work": "energy+conservative force+E/F loss+mixed double backward+AdamW update",
        "loss": "(energy/N)^2+100*mean(force^2); zero synthetic targets; not accuracy training",
        "data_preparation_in_timing": False, "arithmetic": "FP32; AMP/TF32 off",
        "profiling_after_all_timings": True,
        "reference_tp_checkpoint_chunk_edges": args.reference_tp_chunk_edges,
        "reference_sc_checkpoint_chunk_nodes": args.reference_sc_chunk_nodes,
        "reference_sc_checkpoint_chunk_channels": args.reference_sc_chunk_channels,
        "reference_saved_tensors_on_cpu": args.reference_save_on_cpu,
        "reference_empty_cache_before_backward": args.reference_empty_cache,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "source": str(Path(__import__("chorus").__file__).resolve()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    result["allocator_configuration"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    result["backward_targets"] = "all_trainable_parameters" if args.parameter_targets_only else "all_leaves"
    result["gradient_execution"] = "autograd.grad+accumulation" if args.parameter_targets_only else "Tensor.backward"
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "tag": args.tag,
                      "median_ms": result["median_ms"]}), flush=True)
    if args.profile:
        # Profilers run in a separate post-timing phase, never between rounds.
        # Their warmup/timed sample is diagnostic and excluded from the result.
        model.load_state_dict(state, strict=True)
        configure_model(model, policy=args.policy, **OPTIONS)
        measure(model, batches, 1, 1, args.out.with_suffix(".profile.txt"), args.parameter_targets_only)


if __name__ == "__main__":
    with forbid_cuda_graph() as graph_policy:
        main(graph_policy)
