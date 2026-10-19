"""Remote CPU reduction of the native large-graph/training audit records."""
import json
import statistics
from pathlib import Path

ROOT = Path("/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/results")


def read(path):
    return json.loads(path.read_text())


def main():
    large, training = [], []
    for n, directory in ((4096, "large50_v35"), (10000, "large50_v38")):
        base = ROOT/directory
        dpa_path = base/f"dpa4_lower_n{n}_k50.json"
        if not dpa_path.exists():
            continue
        dpa = read(dpa_path)
        record = {"atoms": n, "edges": dpa["neighbors_within_cutoff"],
                  "neighbors": dpa["neighbors"], "dpa_ms": dpa["median_ms"],
                  "dpa_reserved_gib": dpa["peak_reserved"]/2**30,
                  "dpa_compile_slots": dpa["compiled_core_slots"],
                  "dpa_triton_modules": dpa["triton_modules"]}
        for mode in ("fastest", "radial_jvp"):
            path = base/f"chorus_persistent_{mode}_n{n}_k50.json"
            if not path.exists():
                record[mode] = {"status": "MISSING"}
                continue
            value = read(path)
            reserved = max(value["peak_reserved"], value["steady_memory"]["reserved_including_graph_pool"],
                           value["capture_memory"]["capture_peak_reserved"])
            record[mode] = {"ms": value["median_ms"], "atoms_per_second": value["atoms_per_second"],
                "reserved_gib": reserved/2**30, "allocated_gib": value["peak_allocated"]/2**30,
                "dpa_speedup": dpa["median_ms"]/value["median_ms"],
                "reserved_reduction_vs_dpa_percent": 100*(1-reserved/dpa["peak_reserved"]),
                "parity": value["parity"], "reference_status": value["reference_status"],
                "capture_validation": value["capture_validation"], "gpu": value["gpu"]}
        large.append(record)
    for n in (128, 512):
        entry = {"atoms": n, "edges": n*50}
        paired = ROOT/"training_paired_v38"
        dpa_path = paired/f"dpa4_lower_n{n}_k50.json"
        dpa = read(dpa_path) if dpa_path.exists() else None
        if dpa:
            entry.update(dpa_ms=dpa["median_ms"], dpa_reserved_gib=dpa["peak_reserved"]/2**30,
                         dpa_compile_slots=dpa["compiled_core_slots"], dpa_triton_note=dpa["triton_note"])
        for mode in ("eager", "capture"):
            path = paired/f"persistent_{mode}_n{n}_k50.json"
            if not path.exists():
                entry[mode] = {"status": "MISSING"}
                continue
            value = read(path)
            groups = {policy: [v for v in value["measurements"] if v["policy"] == policy]
                      for policy in ("reference", "generated")}
            memory = {policy: statistics.median(v["peak_reserved"] for v in group)
                      for policy, group in groups.items()}
            entry[mode] = {"reference_ms": value["median_ms"]["reference"],
                "generated_ms": value["median_ms"]["generated"], "same_mode_speedup": value["speedup"],
                "generated_reserved_gib": memory["generated"]/2**30,
                "reference_reserved_gib": memory["reference"]/2**30,
                "same_mode_reserved_reduction_percent": 100*(1-memory["generated"]/memory["reference"]),
                "dpa_speedup": dpa["median_ms"]/value["median_ms"]["generated"] if dpa else None,
                "generated_atoms_per_second": n*1000/value["median_ms"]["generated"],
                "tp_coefficient_trainability": value["tp_coefficient_trainability"],
                "trainable_parameters": value["trainable_parameters"],
                "max_parameter_gradient_relative_l2": max(v["relative_l2"] for v in value["parameter_gradient_parity"].values()),
                "optimizer_update_max_abs": value["optimizer_update_max_abs"],
                "optimizer_update_max_relative_l2": value["optimizer_update_max_relative_l2"],
                "strict_elementwise_failure_count": len(value["strict_elementwise_gradient_failures"]),
                "capture_validation": [v["capture_validation"] for v in value["measurements"]]}
        training.append(entry)
    scaling = {}
    if len(large) == 2:
        for mode in ("fastest", "radial_jvp"):
            if "atoms_per_second" in large[0][mode] and "atoms_per_second" in large[1][mode]:
                scaling[mode] = {"atoms_increase_factor": large[1]["atoms"]/large[0]["atoms"],
                    "throughput_increase_percent": 100*(large[1][mode]["atoms_per_second"]/large[0][mode]["atoms_per_second"]-1),
                    "latency_increase_factor": large[1][mode]["ms"]/large[0][mode]["ms"]}
    result = {"large50": large, "large50_scaling": scaling, "training50": training}
    path = ROOT/"summary_large_training.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
