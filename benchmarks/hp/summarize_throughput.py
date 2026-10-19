"""Run under a remote batch allocation; summarize the strict level-2 pair."""
import json
import os
from pathlib import Path


ROOT = Path("/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/results")
FOLDER = ROOT/os.environ.get("RESULT_SET", "paired_level2_efv_v10")
rows = []
for n in (128, 1024, 4096):
    full = json.loads((FOLDER/f"dpa4_full_n{n}.json").read_text())
    lower = json.loads((FOLDER/f"dpa4_lower_n{n}.json").read_text())
    for value in (full, lower):
        assert value["DP_COMPILE_INFER"] == "1"
        assert value["DP_TRITON_INFER"] == "2"
        assert value["should_use_compile"] and value["compiled_core_slots"] > 0
        assert value["triton_modules"] and all(
            entry["level"] == 2 and entry["fused_value_path"] != "NoneType"
            for entry in value["triton_modules"])
    for scope in ("final", "persistent"):
        value = json.loads((FOLDER/f"chorus_{scope}_n{n}.json").read_text())
        assert value["outputs"] == "energy+forces+virial"
        assert value["virial_reduction"] == "FP64"
        assert value["capture"] == "captured"
        assert not value["imported_external_tp"]
        assert value["capture_validation"]["changed_positions"]
        assert value["parity"]["forces"]["relative_l2"] < 1e-4
        reserved = max(value["peak_reserved"], value["capture_memory"]["capture_peak_reserved"],
                       value["steady_memory"]["reserved_including_graph_pool"])
        rows.append({"atoms": n, "scope": scope, "dpa_full_ms": full["median_ms"],
            "dpa_lower_ms": lower["median_ms"], "chorus_ms": value["median_ms"],
            "speedup_vs_full": full["median_ms"]/value["median_ms"],
            "speedup_vs_lower": lower["median_ms"]/value["median_ms"],
            "dpa_lower_peak_allocated_gib": lower["peak_allocated"]/2**30,
            "dpa_lower_peak_reserved_gib": lower["peak_reserved"]/2**30,
            "chorus_capture_peak_allocated_gib": value["capture_memory"]["capture_peak_allocated"]/2**30,
            "chorus_peak_reserved_including_graph_gib": reserved/2**30,
            "reserved_reduction_fraction": 1-reserved/lower["peak_reserved"],
            "dpa_resident_mib": lower["process_resident_mib"],
            "chorus_resident_mib": value["process_resident_mib"],
            "parity": value["parity"], "CUDA_VISIBLE_DEVICES": value["CUDA_VISIBLE_DEVICES"]})
        assert value["CUDA_VISIBLE_DEVICES"] == lower["CUDA_VISIBLE_DEVICES"] == full["CUDA_VISIBLE_DEVICES"]
report = {"formal_baseline": "DPA4-mini C32, inference compile=1, Triton=2, TF32 off",
          "outputs": "energy, forces, virial", "rows": rows,
          "interpretation": "Measured improvements, not a claim that Persistent far exceeds DPA4-mini"}
(ROOT/"summary_level2_efv_v10.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2), flush=True)
