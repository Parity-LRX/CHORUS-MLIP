"""Same-checkpoint eager inference qualification; no cast/retraining/model swap."""
import argparse
import json
import statistics
from pathlib import Path

import torch
from chorus.hp import configure_model
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from chorus.synthetic import compute_energy_forces, make_fixed_graph
from qualify_and_bench import errors, measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--elements", default="H,C,N,O")
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symmetric-contraction", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--data-h5", type=Path)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--checkpoint-sc", action="store_true")
    parser.add_argument("--radial-jvp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radial-gate-pipeline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radial-destination", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radial-cache-gates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radial-tile-edges", type=int, default=65536)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    potential = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.checkpoint, element_types=args.elements.split(","), device="cuda",
        reuse_checkpoint_u=args.checkpoint_sc,
        **({"ictd_fix_product_backend": "ictd-bridge-u", "ictd_tp_backend": "pytorch"}
           if args.checkpoint_sc else {}))
    model = potential.wrapper.model
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Checkpoint is not FP32: qualification does not cast the potential")
    model.eval().requires_grad_(False)
    optimize_sc = (args.checkpoint_sc or not getattr(model, "ictd_fix_use_reduced_cg", False)
                   if args.symmetric_contraction == "auto" else args.symmetric_contraction == "on")
    if args.data_h5 is None:
        graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=24, dtype=torch.float32,
                                 device=torch.device("cuda"), seed=20261017)
        graph = (graph[0]*0.4,) + graph[1:]
        inputs = {"kind": "synthetic_stress_graph", "seed": 20261017, "position_scale": 0.4}
    else:
        from chorus.data import H5Dataset, collate_fn_h5
        data = H5Dataset("val", file_path=str(args.data_h5), expected_max_radius=model.max_radius)
        batch = collate_fn_h5([data[i] for i in range(args.frames)])
        graph = tuple(batch[i].to(device="cuda", dtype=(torch.long if i in (1,2,5,6)
                                                       else torch.float32))
                      for i in (0,1,2,5,6,7,8))
        inputs = {"kind": "held_out_h5", "path": str(args.data_h5),
                  "frame_indices": list(range(args.frames))}
    inputs.update(atoms=graph[0].shape[0], edges=graph[3].numel())
    reference = tuple(value.detach() for value in compute_energy_forces(model, graph, create_graph=False))
    if any(not bool(torch.isfinite(value).all()) for value in reference):
        failure = {"status": "REFERENCE_NONFINITE", "checkpoint": args.checkpoint, "inputs": inputs,
                   "nonfinite_counts": [int((~torch.isfinite(value)).sum()) for value in reference]}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(failure, indent=2)+"\n")
        raise RuntimeError("Original reference is nonfinite; no HP speedup is reported for this input")
    info = configure_model(model, policy="generated", symmetric_contraction=optimize_sc,
                           checkpoint_contraction=args.checkpoint_sc, topology_backend="device",
                           radial_jvp=args.radial_jvp, radial_tile_edges=args.radial_tile_edges,
                           radial_gate_pipeline=args.radial_gate_pipeline, radial_destination=args.radial_destination,
                           radial_cache_gates=args.radial_cache_gates)
    actual = tuple(value.detach() for value in compute_energy_forces(model, graph, create_graph=False))
    parity = {name: errors(got, expected) for name, got, expected in
              zip(("energy", "forces", "atomic_energy"), actual, reference)}
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-3)
        if float(expected.double().norm()) > 1e-6 and errors(got, expected)["relative_l2"] > 1e-4:
            raise AssertionError("Whole-checkpoint relative L2 parity exceeded 1e-4")
    samples = {"reference": [], "generated": []}
    for repeat in range(3):
        order = ("reference", "generated") if repeat % 2 == 0 else ("generated", "reference")
        for policy in order:
            configure_model(model, policy=policy, symmetric_contraction=optimize_sc,
                            checkpoint_contraction=args.checkpoint_sc, topology_backend="device",
                            radial_jvp=args.radial_jvp, radial_tile_edges=args.radial_tile_edges,
                            radial_gate_pipeline=args.radial_gate_pipeline, radial_destination=args.radial_destination,
                            radial_cache_gates=args.radial_cache_gates)
            samples[policy].append(measure(model, graph, 3, 10))
    median = {key: statistics.median(value["ms"] for value in values) for key, values in samples.items()}
    result = {"checkpoint": args.checkpoint, "device": torch.cuda.get_device_name(), "inputs": inputs,
              "channels": model.channels, "lmax": model.lmax,
              "scope": model.ictd_fix_phase_scope, "plan": info, "parity": parity,
              "samples": samples, "median_ms": median,
              "speedup": median["reference"]/median["generated"], "status": "PASS"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
