"""Batch-only reduction of preserved gradient audits; never alters criteria."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def compact(check):
    return {
        "strict_pass": check["strict_pass"],
        "failure_elements": check["gradient_failure_elements"],
        "failures": [{key: row[key] for key in (
            "parameter", "count", "max_bound_ratio", "indices", "got", "reference", "allowed",
            "relative_l2", "max_abs") if key in row} for row in check["gradient_failures"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = []
    for path in args.inputs:
        result = json.loads(path.read_text())
        item = {"input": str(path), "criterion": result["criterion"],
                "passing_modes_all_batches": result["strict_passing_modes_all_batches"], "batches": []}
        for batch in result["batches"]:
            modes = {}
            for mode, value in batch["modes"].items():
                if "comparisons" not in value:
                    modes[mode] = value
                    continue
                checks = [check for repeat in value["comparisons"] for check in repeat["vs_original_repeats"]]
                worst = max(checks, key=lambda check: (
                    check["gradient_failure_elements"],
                    max((failure["max_bound_ratio"] for failure in check["gradient_failures"]), default=0.)))
                parameters = defaultdict(lambda: {"comparisons_failed": 0, "maximum_bound_ratio": 0.})
                for check in checks:
                    for failure in check["gradient_failures"]:
                        entry = parameters[failure["parameter"]]
                        entry["comparisons_failed"] += 1
                        entry["maximum_bound_ratio"] = max(entry["maximum_bound_ratio"], failure["max_bound_ratio"])
                modes[mode] = {"status": value["status"], "worst": compact(worst), "parameters": dict(parameters)}
            row = {"batch": batch["batch"],
                   "reference_self_repeat": [compact(check) for check in batch["reference_self_repeat"]],
                   "modes": modes}
            item["batches"].append(row)
            for mode in ("generated", "stable_csr", "original_sc", "stable_original_activation_harmonics"):
                if mode in modes:
                    print(json.dumps({"input": str(path), "batch": batch["batch"],
                                      "mode": mode, **modes[mode]}), flush=True)
        report.append(item)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
