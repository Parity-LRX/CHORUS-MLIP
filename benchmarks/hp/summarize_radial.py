"""Read-only candidate reduction; execute in an XHPC sbatch CPU allocation."""
import json
from pathlib import Path

root = Path("/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/results")
rows = []


def reserved(record):
    return max(record["peak_reserved"], record.get("capture_memory", {}).get("capture_peak_reserved", 0),
               record.get("steady_memory", {}).get("reserved_including_graph_pool", 0))


for directory in sorted(root.glob("radial_*")):
    if not directory.is_dir():
        continue
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".trace.json"):
            continue
        record = json.loads(path.read_text())
        if "atoms" not in record or "median_ms" not in record:
            continue
        n = record["atoms"]
        control_path = directory/f"persistent_tileoff_n{n}.json"
        control = json.loads(control_path.read_text()) if control_path.exists() else None
        if control is not None:
            assert control["CUDA_VISIBLE_DEVICES"] == record["CUDA_VISIBLE_DEVICES"]
        assert record["outputs"] == "energy+forces+virial"
        assert not record["imported_external_tp"] and record["parameters"] == 1029967
        rows.append({"experiment": directory.name, "file": path.name, "atoms": n,
            "gpu": record["CUDA_VISIBLE_DEVICES"], "median_ms": record["median_ms"],
            "reserved_gib": reserved(record)/2**30,
            "force_relative_l2": record["parity"]["forces"]["relative_l2"],
            "same_card_control_ms": None if control is None else control["median_ms"],
            "time_change_percent": None if control is None else 100*(record["median_ms"]/control["median_ms"]-1),
            "reserved_reduction_percent": None if control is None else 100*(1-reserved(record)/reserved(control)),
            "plan": record["plan"]})
target = root/"summary_radial_candidates.json"
target.write_text(json.dumps({"rows": rows}, indent=2)+"\n")
print(json.dumps({"rows": rows}, indent=2))
