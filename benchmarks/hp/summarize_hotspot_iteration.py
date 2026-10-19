"""Remote XHPC CPU reduction of the native_v41--48 optimization records."""
import json
from pathlib import Path


ROOT = Path("/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/results")


def read(path):
    return json.loads(path.read_text())


def main():
    large = []
    for n, old_set in ((4096, "large50_v35"), (10000, "large50_v38")):
        base = ROOT/"large50_v48"
        dpa = read(base/f"dpa4_lower_n{n}_k50.json")
        old = read(ROOT/old_set/f"chorus_persistent_fastest_n{n}_k50.json")
        entry = {"atoms": n, "edges": dpa["neighbors_within_cutoff"],
                 "dpa_ms": dpa["median_ms"], "old_persistent_ms": old["median_ms"],
                 "dpa_reserved_gib": dpa["peak_reserved"]/2**30,
                 "twofold_target_ms": dpa["median_ms"]/2,
                 "dpa_compile_slots": dpa["compiled_core_slots"],
                 "dpa_triton_modules": dpa["triton_modules"]}
        for mode in ("fastest", "radial_jvp"):
            value = read(base/f"chorus_persistent_{mode}_n{n}_k50.json")
            reserved = max(value["peak_reserved"], value["steady_memory"]["reserved_including_graph_pool"],
                           value["capture_memory"]["capture_peak_reserved"])
            entry[mode] = {"ms": value["median_ms"], "reserved_gib": reserved/2**30,
                           "dpa_speedup": dpa["median_ms"]/value["median_ms"],
                           "reserved_reduction_vs_dpa_percent": 100*(1-reserved/dpa["peak_reserved"]),
                           "parity": value["parity"], "reference_status": value["reference_status"],
                           "capture_validation": value["capture_validation"]}
        entry["latency_reduction_vs_old_percent"] = 100*(1-entry["fastest"]["ms"]/old["median_ms"])
        entry["remaining_latency_reduction_for_twofold_percent"] = 100*(1-entry["twofold_target_ms"]/entry["fastest"]["ms"])
        large.append(entry)
    candidates = []
    for directory, schedules in (("adjoint_v43", ("segmented", "edge_dx")),
                                  ("phase_context_v44", ("phase_context",)),
                                  ("csr_v45", ("csr1", "csr4", "csr8")),
                                  ("split_v46", ("split",))):
        for schedule in schedules:
            value = read(ROOT/directory/f"{schedule}.json")
            candidates.append({"schedule": schedule, "atoms": value["atoms"],
                               "source_snapshot": value["source_snapshot"],
                               "baseline_ms": [v["median_ms"] for v in value["results"]["joint"]],
                               "candidate_ms": value["results"][schedule][0]["median_ms"],
                               "candidate_reserved_gib": value["results"][schedule][0]["peak_reserved"]/2**30})
    training = []
    for n in (128, 512):
        base = ROOT/"training_paired_v48"
        dpa = read(base/f"dpa4_lower_n{n}_k50.json")
        entry = {"atoms": n, "dpa_ms": dpa["median_ms"], "dpa_reserved_gib": dpa["peak_reserved"]/2**30}
        for mode in ("eager", "capture"):
            value = read(base/f"persistent_{mode}_n{n}_k50.json")
            old = read(ROOT/"training_paired_v38"/f"persistent_{mode}_n{n}_k50.json")
            entry[mode] = {"reference_ms": value["median_ms"]["reference"],
                           "generated_ms": value["median_ms"]["generated"],
                           "old_generated_ms": old["median_ms"]["generated"],
                           "dpa_speedup": dpa["median_ms"]/value["median_ms"]["generated"],
                           "speedup_vs_old_generated": old["median_ms"]["generated"]/value["median_ms"]["generated"],
                           "generated_reserved_gib": max(v["peak_reserved"] for v in value["measurements"]
                                                         if v["policy"] == "generated")/2**30,
                           "trainable_parameters": value["trainable_parameters"],
                           "max_parameter_gradient_relative_l2": max(v["relative_l2"] for v in value["parameter_gradient_parity"].values()),
                           "strict_elementwise_failure_count": len(value["strict_elementwise_gradient_failures"]),
                           "optimizer_update_max_relative_l2": value["optimizer_update_max_relative_l2"]}
        training.append(entry)
    result = {"large50": large, "training50": training, "screening_candidates": candidates}
    path = ROOT/"summary_hotspots_v48.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
