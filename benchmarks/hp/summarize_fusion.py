"""Run this result reduction inside an sbatch CPU allocation, not on the Mac."""
import json
import os
from pathlib import Path


root = Path("/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/results")
result_set = os.environ["RESULT_SET"]
points = root/result_set
rows = []
for n in (128, 1024, 4096):
    dpa = json.loads((points/f"dpa4_lower_n{n}.json").read_text())
    full = json.loads((points/f"dpa4_full_n{n}.json").read_text())
    persistent = json.loads((points/f"chorus_persistent_n{n}.json").read_text())
    final = json.loads((points/f"chorus_final_n{n}.json").read_text())
    assert dpa["DP_COMPILE_INFER"] == "1" and dpa["DP_TRITON_INFER"] == "2"
    assert dpa["should_use_compile"] and dpa["compiled_core_slots"] > 0
    assert all(row["level"] == 2 and row["fused_value_path"] == "_TritonSO2ValuePath"
               for row in dpa["triton_modules"])
    assert dpa["model_fp32_tf32_off"] and dpa["parameters"] == 650428
    assert {dpa["CUDA_VISIBLE_DEVICES"], full["CUDA_VISIBLE_DEVICES"],
            final["CUDA_VISIBLE_DEVICES"], persistent["CUDA_VISIBLE_DEVICES"]} == {dpa["CUDA_VISIBLE_DEVICES"]}
    assert dpa.get("allocator_config") == persistent.get("allocator_config")

    def memory(record):
        return max(record["peak_reserved"],
                   record.get("capture_memory", {}).get("capture_peak_reserved", 0),
                   record.get("steady_memory", {}).get("reserved_including_graph_pool", 0))

    for record, parameters in ((final, 792198), (persistent, 1029967)):
        assert record["parameters"] == parameters and not record["imported_external_tp"]
        assert record["capture"] == "captured" and record["outputs"] == "energy+forces+virial"
        assert record["virial_reduction"] == "FP64"
        assert record["capture_validation"]["changed_positions"]
        assert record["parity"]["forces"]["relative_l2"] < 1e-4
        assert record["parity"]["virial"]["relative_l2"] < 1e-4
    assert persistent["plan"]["counts"]["terminal_projections"] == 1
    old_path = points/f"chorus_previous_persistent_n{n}.json"
    matched_previous = old_path.is_file()
    if not matched_previous:
        old_path = root/"paired_level2_efv_v10"/f"chorus_persistent_n{n}.json"
    old = json.loads(old_path.read_text())
    if matched_previous:
        assert old["CUDA_VISIBLE_DEVICES"] == persistent["CUDA_VISIBLE_DEVICES"]
    rows.append({
        "atoms": n, "gpu": dpa["CUDA_VISIBLE_DEVICES"],
        "dpa_lower_ms": dpa["median_ms"], "dpa_full_ms": full["median_ms"],
        "final_ms": final["median_ms"], "persistent_ms": persistent["median_ms"],
        "persistent_speedup_over_dpa": dpa["median_ms"]/persistent["median_ms"],
        "previous_persistent_ms": old["median_ms"],
        "previous_reserved_gib": memory(old)/2**30,
        "previous_to_current_reserved_reduction_percent": 100*(1-memory(persistent)/memory(old)),
        "previous_same_card_pair": matched_previous,
        "previous_to_current_time_reduction_percent": 100*(1-persistent["median_ms"]/old["median_ms"]),
        "dpa_reserved_gib": memory(dpa)/2**30,
        "persistent_reserved_gib": memory(persistent)/2**30,
        "persistent_reserved_reduction_percent": 100*(1-memory(persistent)/memory(dpa)),
        "dpa_resident_mib": dpa["process_resident_mib"],
        "persistent_resident_mib": persistent["process_resident_mib"],
        "parity": persistent["parity"],
    })
result = {"result_set": result_set, "rows": rows}
target = root/f"summary_{result_set}.json"
target.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
