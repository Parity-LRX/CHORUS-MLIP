"""Remote diagnostic of extensive virial cancellation and replay variation."""
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from profile_scaling import geometry, reference_values, neighbor_statistics
from qualify_and_bench import errors
from chorus.hp import configure_model, energy_forces_virial
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def main():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LAMMPS_MLIAP_MFF.from_checkpoint(
        "/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth",
        element_types=["H", "C", "N", "O"], device="cuda", reuse_checkpoint_u=True,
        ictd_fix_product_backend="ictd-bridge-u", ictd_tp_backend="pytorch").wrapper.model
    model.eval().requires_grad_(False)
    results = []
    for n in (4096, 10000):
        configure_model(model, policy="reference")
        graph = geometry(n, model.max_radius, 20260728, 50)
        neighbors = neighbor_statistics(graph, 50)
        def original(function):
            # Verification only: saved tensors may be moved to CPU without
            # changing FP32 operators. This is excluded from all timings.
            with (torch.autograd.graph.save_on_cpu(pin_memory=True) if n >= 10000 else nullcontext()):
                return function()
        reference = original(lambda: reference_values(model, graph, True))
        record = {"atoms": n, "neighbors": neighbors, "reference_virial": reference[3].cpu().tolist(),
                  "verification_saved_tensors_on_cpu": n >= 10000,
                  "repeated_strain_reference": [], "edge": {}}
        for _ in range(3):
            value = original(lambda: reference_values(model, graph, True))
            record["repeated_strain_reference"].append({"virial": value[3].cpu().tolist(),
                "virial_error": errors(value[3], reference[3]),
                "force_error": errors(value[1], reference[1])})
            del value
        for policy in ("reference", "generated"):
            configure_model(model, policy=policy, checkpoint_contraction=True, topology_backend="device")
            record["edge"][policy] = []
            for _ in range(3):
                value = (original(lambda: energy_forces_virial(model, *graph)) if policy == "reference"
                         else energy_forces_virial(model, *graph))
                record["edge"][policy].append({"virial": value["virial"].cpu().tolist(),
                    "virial_error": errors(value["virial"], reference[3]),
                    "virial_max_abs_per_atom": float((value["virial"]-reference[3]).abs().max())/n,
                    "force_error": errors(value["forces"], reference[1])})
                del value
        model.skip_input_validation = True
        model.preserve_edge_order = True
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                energy_forces_virial(model, *graph)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            captured = energy_forces_virial(model, *graph)
        record["captured"] = []
        for _ in range(3):
            cg.replay()
            eager = energy_forces_virial(model, *graph)
            record["captured"].append({"virial": captured["virial"].cpu().tolist(),
                "reference_error": errors(captured["virial"], reference[3]),
                "eager_error": errors(captured["virial"], eager["virial"]),
                "force_eager_error": errors(captured["forces"], eager["forces"])})
            del eager
        del cg, captured, reference, graph
        torch.cuda.empty_cache()
        results.append(record)
        print(json.dumps(record, indent=2), flush=True)
    out = Path("/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/results/large50_v32/virial_replay_diagnostic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
