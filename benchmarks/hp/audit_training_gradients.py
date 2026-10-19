"""Remote, strict-gradient-first isolation of native training roundoff.

No optimizer or speed claim is made here. The registered original FP32 formula,
checkpoint, three input batches, seeds and elementwise tolerances stay fixed.
Original self-repeat spread is reported, never used to enlarge tolerances.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import torch

from no_cuda_graph import forbid_cuda_graph
from profile_scaling import geometry, neighbor_statistics
from profile_training import load_trainable
from profile_training_nograph import record
from qualify_and_bench import errors
from chorus.hp import configure_model
from chorus.hp.runtime import _HANDLES


MODES = (
    "reference", "generated", "stable_csr", "original_sc", "original_density",
    "original_phase", "original_channel", "original_activation", "original_species",
    "original_messages", "original_harmonics", "original_maps",
    "only_messages", "only_sc", "only_density", "only_phase", "only_channel",
    "stable_original_sc", "stable_original_phase", "stable_original_maps",
    "stable_original_activation", "original_activation_harmonics",
    "stable_original_activation_harmonics",
    "stable_original_activation_harmonics_sc",
    "stable_original_activation_harmonics_sc_phase",
)


def configure(model, mode):
    if mode == "reference":
        return {"plan": configure_model(model, policy="reference"),
                "remaining_handles": {}, "stable_csr": False}
    stable = mode.startswith("stable_") or mode == "stable_csr"
    mode = mode.removeprefix("stable_")
    options = dict(training=True, policy="generated", checkpoint_contraction=True,
                   topology_backend="device")
    if mode in ("original_sc", "original_activation_harmonics_sc",
                "original_activation_harmonics_sc_phase"):
        options["symmetric_contraction"] = False
    if mode == "original_density":
        options["density_fusion"] = False
    if mode in ("original_phase", "original_maps", "original_activation_harmonics_sc_phase"):
        options["phase_fusion"] = False
    if mode in ("original_species", "original_maps"):
        options["species_fusion"] = False
    plan = configure_model(model, **options)
    remove = set()
    if mode in ("original_channel", "original_maps"):
        remove.add("_hp_channel_linear")
    if mode in ("original_activation", "original_activation_harmonics", "original_maps",
                "original_activation_harmonics_sc", "original_activation_harmonics_sc_phase"):
        remove.add("_hp_activation")
    if mode in ("original_harmonics", "original_activation_harmonics",
                "original_activation_harmonics_sc", "original_activation_harmonics_sc_phase"):
        remove.add("_hp_harmonics")
    if mode == "original_messages":
        remove.add("_hp_message")
    if mode.startswith("only_"):
        keep = {"messages": {"_hp_message"}, "sc": {"_hp_contraction"},
                "density": {"_hp_density"},
                "phase": {"_hp_phase_trunk", "_hp_phase_coefficients"},
                "channel": {"_hp_channel_linear"}}[mode.removeprefix("only_")]
        remove.update(set(_HANDLES) - keep)
    counts = {}
    for module in model.modules():
        for name in remove:
            if name in vars(module):
                object.__delattr__(module, name)
        for name in _HANDLES:
            if name in vars(module):
                counts[name] = counts.get(name, 0) + 1
    return {"plan": plan, "remaining_handles": counts, "stable_csr": stable}


def compare(actual, reference):
    values, grad = actual
    ref_values, ref_grad = reference
    output = {}
    output_failures = []
    for name, got, ref in zip(("loss", "energy", "forces", "atomic_energy"), values, ref_values):
        output[name] = errors(got, ref)
        try:
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)
        except AssertionError as error:
            output_failures.append({"name": name, "details": str(error)})
    failures, dependencies, norms = [], [], {}
    for name, ref in ref_grad.items():
        got = grad[name]
        if (ref is None) != (got is None):
            dependencies.append(name)
            continue
        if ref is None:
            continue
        norms[name] = errors(got, ref)
        delta = (got.double() - ref.double()).abs()
        bound = 2e-4 + 3e-3 * ref.double().abs()
        bad = ((~got.isfinite()) | (~ref.isfinite()) |
               (~torch.isclose(got, ref, atol=2e-4, rtol=3e-3)))
        if bool(bad.any()):
            indices = bad.reshape(-1).nonzero().reshape(-1)
            selected = indices[:16]
            failures.append({"parameter": name, "count": int(indices.numel()),
                "max_bound_ratio": float((delta / bound).max()),
                "indices": selected.tolist(), "got": got.reshape(-1)[selected].tolist(),
                "reference": ref.reshape(-1)[selected].tolist(),
                "allowed": bound.reshape(-1)[selected].tolist(), **norms[name]})
    return {"strict_pass": not (output_failures or failures or dependencies),
        "outputs": output, "output_failures": output_failures,
        "gradient_failures": failures, "gradient_failure_tensors": len(failures),
        "gradient_failure_elements": sum(v["count"] for v in failures),
        "dependency_failures": dependencies, "parameter_gradient_errors": norms}


def reference_intersection(references):
    """Does any value satisfy the unchanged criterion for every original repeat?"""
    impossible = []
    for name, first in references[0][1].items():
        if first is None:
            continue
        values = torch.stack([reference[1][name] for reference in references]).double()
        bound = 2e-4 + 3e-3 * values.abs()
        lower, upper = (values-bound).amax(0), (values+bound).amin(0)
        bad = lower > upper
        if bool(bad.any()):
            indices = bad.reshape(-1).nonzero().reshape(-1)
            selected = indices[:16]
            impossible.append({"parameter": name, "count": int(indices.numel()),
                "indices": selected.tolist(),
                "original_values": values.reshape(len(references), -1)[:, selected].tolist(),
                "lower": lower.reshape(-1)[selected].tolist(),
                "upper": upper.reshape(-1)[selected].tolist()})
    return {"compatible": not impossible,
            "incompatible_elements": sum(row["count"] for row in impossible), "details": impossible}


def main(graph_policy):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--atoms", type=int, required=True)
    p.add_argument("--neighbors", type=int, default=50, choices=(32, 50))
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--reference-repeats", type=int, default=3)
    p.add_argument("--candidate-repeats", type=int, default=2)
    p.add_argument("--deterministic-diagnostic", action="store_true",
                   help="Additional diagnostic only; do not replace the original audit records")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261017)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(args.deterministic_diagnostic)
    model, trainability = load_trainable(args.checkpoint)
    model.skip_input_validation = model.preserve_edge_order = True
    identity = {name: (id(v), v.requires_grad, v._version) for name, v in model.named_parameters()}
    result = {"checkpoint": args.checkpoint, "atoms": args.atoms, "neighbors": args.neighbors,
        "graph_policy": graph_policy, "trainability": trainability,
        "parameters": sum(v.numel() for v in model.parameters()),
        "criterion": {"output_atol": 1e-4, "output_rtol": 1e-3,
                      "parameter_gradient_atol": 2e-4, "parameter_gradient_rtol": 3e-3},
        "gpu": torch.cuda.get_device_name(), "batches": [], "arithmetic": "FP32; AMP/TF32 off",
        "deterministic_diagnostic": args.deterministic_diagnostic,
        "acceptance_baseline_replaced": False}
    for batch in range(3):
        graph = list(geometry(args.atoms, model.max_radius, 20260728 + batch, args.neighbors))
        neighbors = neighbor_statistics(graph, args.neighbors)
        graph[1] = graph[1].roll(batch)
        order = torch.arange(graph[3].numel(), device="cuda").roll(batch * 17)
        for field in (3, 4, 5):
            graph[field] = graph[field].index_select(0, order)
        graph = tuple(graph)
        configure_model(model, policy="reference")
        references = [record(model, graph) for _ in range(args.reference_repeats)]
        row = {"batch": batch, "geometry_seed": 20260728 + batch, "species_roll": batch,
            "edge_roll": batch * 17, "neighbors": neighbors,
            "reference_self_repeat": [compare(v, references[0]) for v in references[1:]],
            "reference_intersection": reference_intersection(references),
            "modes": {}}
        result["batches"].append(row)
        print(json.dumps({"stage": "original_repeat", "batch": batch,
            "failure_elements": [v["gradient_failure_elements"] for v in row["reference_self_repeat"]],
            "incompatible_elements": row["reference_intersection"]["incompatible_elements"]}), flush=True)
        for mode in args.modes:
            try:
                plan = configure(model, mode)
                with ExitStack() as stack:
                    if plan["stable_csr"]:
                        import mace_ictc.models.message_topology as topology
                        stack.enter_context(patch.object(topology, "topology", topology.stable_topology))
                    comparisons = []
                    for repeat in range(args.candidate_repeats):
                        actual = record(model, graph)
                        checked = [compare(actual, ref) for ref in references]
                        comparisons.append({"repeat": repeat, "vs_original_repeats": checked})
                        del actual
                row["modes"][mode] = {"status": "PASS" if all(
                    v["strict_pass"] for r in comparisons for v in r["vs_original_repeats"]) else "FAIL_STRICT",
                    "configuration": plan, "comparisons": comparisons}
            except Exception as error:
                row["modes"][mode] = {"status": "ERROR", "type": type(error).__name__, "details": str(error)}
            after = {name: (id(v), v.requires_grad, v._version) for name, v in model.named_parameters()}
            if after != identity:
                raise AssertionError("Gradient audit changed parameter identity, trainability or values")
            entry = row["modes"][mode]
            print(json.dumps({"stage": "candidate", "batch": batch, "mode": mode,
                "status": entry["status"], "worst_failure_elements": None if "comparisons" not in entry else max(
                    v["gradient_failure_elements"] for r in entry["comparisons"] for v in r["vs_original_repeats"])}), flush=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        del references
        torch.cuda.empty_cache()
    result["strict_passing_modes_all_batches"] = [mode for mode in args.modes if all(
        row["modes"][mode]["status"] == "PASS" for row in result["batches"])]
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"strict_passing_modes_all_batches": result["strict_passing_modes_all_batches"]}), flush=True)


if __name__ == "__main__":
    with forbid_cuda_graph() as graph_policy:
        main(graph_policy)
