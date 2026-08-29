#!/usr/bin/env python3
"""Audit the collected MS25 benchmark for completeness and metric consistency."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DATASETS = ("hea", "zr-o", "mgo-2x2")
MODELS_PER_DATASET = 11
METRICS = ("energy_mae", "energy_rmse", "force_mae", "force_rmse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = json.loads(args.results.read_text())
    records = payload["records"]
    errors: list[str] = []
    warnings: list[str] = []

    keys = [(row["dataset"], row["model"]) for row in records]
    duplicates = sorted(key for key, count in Counter(keys).items() if count != 1)
    if duplicates:
        errors.append(f"duplicate dataset/model keys: {duplicates}")

    expected_count = len(DATASETS) * MODELS_PER_DATASET
    if len(records) != expected_count:
        errors.append(f"record count {len(records)} != {expected_count}")
    for dataset in DATASETS:
        count = sum(row["dataset"] == dataset for row in records)
        if count != MODELS_PER_DATASET:
            errors.append(f"{dataset}: {count} records != {MODELS_PER_DATASET}")

    transfer_count = 0
    calibration_count = 0
    for row in records:
        label = f"{row['dataset']}/{row['model']}"
        if row.get("selection_rule") != "minimum full-validation Force MAE":
            errors.append(f"{label}: inconsistent selection rule")
        if not 0 < int(row["selected_step"]) <= 45_000:
            errors.append(f"{label}: selected step outside the declared budget")
        if int(row["trainable_parameters"]) <= 0:
            errors.append(f"{label}: non-positive parameter count")
        for block_name in ("test",):
            block = row[block_name]
            for metric in METRICS:
                value = float(block[metric])
                if not math.isfinite(value) or value < 0:
                    errors.append(f"{label}: invalid {block_name}.{metric}={value}")
            if block["energy_rmse"] + 1e-12 < block["energy_mae"]:
                errors.append(f"{label}: Energy RMSE < MAE")
            if block["force_rmse"] + 1e-12 < block["force_mae"]:
                errors.append(f"{label}: Force RMSE < MAE")
        source = Path(row["source"])
        if not source.is_file():
            errors.append(f"{label}: missing source {source}")
        if "nequip_family_invalid" in str(source) or "failed_numpy" in str(source):
            errors.append(f"{label}: invalid historical NequIP run entered the table")
        if row["dataset"] == "zr-o":
            calibration = row.get("energy_calibration")
            if calibration is None or calibration.get("fit_split") != "train only":
                errors.append(f"{label}: missing train-only energy calibration")
            else:
                calibration_count += 1
        if row["dataset"] == "mgo-2x2":
            transfer = row.get("mgo_4x4_size_transfer_test")
            if transfer is None:
                errors.append(f"{label}: missing 2x2-to-4x4 transfer test")
            else:
                transfer_count += 1
                for metric in METRICS:
                    value = float(transfer[metric])
                    if not math.isfinite(value) or value < 0:
                        errors.append(f"{label}: invalid transfer {metric}={value}")

    if payload.get("missing_artifacts"):
        errors.append(f"collector still reports missing artifacts: {payload['missing_artifacts']}")
    if not payload.get("complete"):
        errors.append("collector complete flag is false")

    audit = {
        "benchmark": payload.get("benchmark"),
        "status": "PASS" if not errors else "FAIL",
        "record_grain": "one dataset/model pair selected by full-validation Force MAE",
        "record_count": len(records),
        "unique_dataset_model_pairs": len(set(keys)),
        "records_by_dataset": dict(sorted(Counter(row["dataset"] for row in records).items())),
        "zr_o_train_only_calibrations": calibration_count,
        "mgo_size_transfer_records": transfer_count,
        "checks": {
            "complete_model_matrix": not any("records" in item or "record count" in item for item in errors),
            "unique_grain": not duplicates,
            "uniform_selection_rule": not any("selection rule" in item for item in errors),
            "finite_nonnegative_metrics": not any("invalid" in item for item in errors),
            "rmse_not_below_mae": not any("RMSE < MAE" in item for item in errors),
            "source_files_present": not any("missing source" in item for item in errors),
            "invalid_nequip_history_excluded": not any("historical NequIP" in item for item in errors),
            "zr_o_calibration_train_only": calibration_count == MODELS_PER_DATASET,
            "mgo_transfer_complete": transfer_count == MODELS_PER_DATASET,
        },
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    if errors:
        raise SystemExit("MS25 audit failed; inspect " + str(args.output))


if __name__ == "__main__":
    main()
