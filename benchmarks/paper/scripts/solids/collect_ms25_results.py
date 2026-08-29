#!/usr/bin/env python3
"""Collect the audited MS25 cross-model comparison into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATASETS = ("hea", "zr-o", "mgo-2x2")


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def metric_block(values: dict[str, Any]) -> dict[str, float]:
    aliases = {
        "energy_mae": ("energy_mae_mev_per_atom", "energy_mae_ev_per_atom"),
        "energy_rmse": ("energy_rmse_mev_per_atom", "energy_rmse_ev_per_atom"),
        "force_mae": ("force_mae_mev_per_angstrom", "force_mae_ev_per_angstrom"),
        "force_rmse": ("force_rmse_mev_per_angstrom", "force_rmse_ev_per_angstrom"),
    }
    result = {}
    for output, (mev_key, ev_key) in aliases.items():
        if mev_key in values:
            result[output] = float(values[mev_key])
        elif ev_key in values:
            result[output] = 1000.0 * float(values[ev_key])
        else:
            raise KeyError(f"missing {mev_key}/{ev_key}")
    return result


def energy_metric_block(values: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for output in ("energy_mae", "energy_rmse"):
        stem = output.replace("energy_", "energy_")
        mev_key = f"{stem}_mev_per_atom"
        ev_key = f"{stem}_ev_per_atom"
        if mev_key in values:
            result[output] = float(values[mev_key])
        elif ev_key in values:
            result[output] = 1000.0 * float(values[ev_key])
        else:
            raise KeyError(f"missing {mev_key}/{ev_key}")
    return result


def parameter_count(raw: dict[str, Any], path: Path) -> int:
    """Return the audited parameter count, with a legacy-log fallback."""
    if "trainable_parameters" in raw:
        return int(raw["trainable_parameters"])
    train_log = path.parent.parent / "logs" / "train.log"
    if train_log.is_file():
        matches = re.findall(r"\bparams=(\d+)\b", train_log.read_text(errors="replace"))
        if matches:
            return int(matches[-1])
    raise KeyError(f"missing trainable parameter count in {path} and {train_log}")


def chorus_envelope(raw: dict[str, Any], key: str) -> dict[str, Any]:
    row = raw["independent_validation_envelopes"][key]
    metrics = row["simultaneous_metrics"]
    return {
        "step": int(row["step"]),
        "energy_mae": 1000.0 * float(metrics["val_energy_mae"]),
        "energy_rmse": 1000.0 * float(metrics["val_energy_rmse"]),
        "force_mae": 1000.0 * float(metrics["val_force_mae"]),
        "force_rmse": 1000.0 * float(metrics["val_force_rmse"]),
    }


def dpa_envelope(raw: dict[str, Any], metric: str) -> dict[str, Any]:
    key = "energy_mae_mev_per_atom" if metric == "energy_mae" else "force_mae_mev_per_angstrom"
    row = min(raw["validation_scan"], key=lambda item: (float(item[key]), int(item["step"])))
    return {"step": int(row["step"]), **metric_block(row)}


def native_envelope(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": (int(row["epoch"]) + 1) * 250,
        "energy_mae": 1000.0 * float(row["mae_e_per_atom"]),
        "energy_rmse": 1000.0 * float(row["rmse_e_per_atom"]),
        "force_mae": 1000.0 * float(row["mae_f"]),
        "force_rmse": 1000.0 * float(row["rmse_f"]),
    }


def standard_envelope(raw: dict[str, Any], key: str) -> dict[str, Any]:
    row = raw["independent_validation_envelopes"][key]
    return {"step": int(row["step"]), **metric_block(row)}


def append_record(
    records: list[dict[str, Any]],
    missing: list[str],
    *,
    path: Path,
    dataset: str,
    model: str,
    family: str,
    parser_kind: str,
    transfer_path: Path | None = None,
) -> None:
    raw = load(path)
    if raw is None:
        missing.append(str(path))
        return
    if parser_kind == "chorus":
        test = metric_block(raw["test_mev"])
        force = chorus_envelope(raw, "force_mae")
        energy = chorus_envelope(raw, "energy_mae")
        selected_step = int(raw["selected_step"])
    elif parser_kind == "native":
        test = metric_block(raw["test"])
        force = native_envelope(raw["independent_validation_envelopes"]["force_mae"])
        energy = native_envelope(raw["independent_validation_envelopes"]["energy_mae"])
        selected_step = (int(raw["selected_epoch"]) + 1) * 250
    elif parser_kind == "dpa":
        test = metric_block(raw["test"])
        force = dpa_envelope(raw, "force_mae")
        energy = dpa_envelope(raw, "energy_mae")
        selected_step = int(raw["selected"]["step"])
    elif parser_kind in {"nequip", "tece"}:
        test = metric_block(raw["test"])
        force = standard_envelope(raw, "force_mae")
        energy = standard_envelope(raw, "energy_mae")
        selected_step = int(raw["selected_step"])
    else:
        raise ValueError(parser_kind)

    # A zero-valued result across all four independently computed metrics is
    # not physically plausible for these noisy force datasets.  Reject such
    # artifacts explicitly; this catches silent label/geometry pipeline
    # failures instead of allowing them to masquerade as perfect accuracy.
    if all(abs(float(test[key])) < 1.0e-12 for key in test):
        missing.append(f"INVALID_ZERO_METRICS:{path}")
        return

    record: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "family": family,
        "trainable_parameters": parameter_count(raw, path),
        "selection_rule": "minimum full-validation Force MAE",
        "selected_step": selected_step,
        "validation_envelopes": {"force_mae": force, "energy_mae": energy},
        "test": test,
        "source": str(path),
    }
    if dataset == "zr-o":
        if parser_kind in {"chorus", "dpa"}:
            calibration_path = path.parent.parent / "train_only_energy_calibration" / "metrics.json"
        elif parser_kind in {"nequip", "tece"}:
            calibration_path = path.parent / "train_only_energy_calibration" / "metrics.json"
        else:
            calibration_path = None

        if parser_kind == "native":
            calibration = raw.get("energy_calibration")
            if calibration is None:
                missing.append(str(path.parent / "train_only_energy_calibration" / "metrics.json"))
            else:
                raw_energy = {
                    "energy_mae": 1000.0 * float(raw["test"]["raw_energy_mae_ev_per_atom"]),
                    "energy_rmse": 1000.0 * float(raw["test"]["raw_energy_rmse_ev_per_atom"]),
                }
                record["test_raw"] = {**test, **raw_energy}
                record["energy_calibration"] = {
                    "fit_split": "train only",
                    "details": calibration,
                }
        elif calibration_path is not None:
            calibration = load(calibration_path)
            if calibration is None:
                missing.append(str(calibration_path))
            else:
                if parser_kind == "chorus":
                    calibrated_energy = energy_metric_block(calibration["test_calibrated"])
                else:
                    calibrated_energy = energy_metric_block(
                        calibration["splits"]["test"]["calibrated"]
                    )
                record["test_raw"] = dict(test)
                record["test"] = {**test, **calibrated_energy}
                record["energy_calibration"] = {
                    "fit_split": "train only",
                    "source": str(calibration_path),
                    "details": calibration,
                }
    if transfer_path is not None:
        transfer = load(transfer_path)
        if transfer is None:
            missing.append(str(transfer_path))
        elif parser_kind == "chorus":
            record["mgo_4x4_size_transfer_test"] = metric_block(transfer["test_mev"])
        elif parser_kind == "native":
            record["mgo_4x4_size_transfer_test"] = metric_block(transfer["test"])
    if parser_kind == "dpa" and "extra_tests" in raw:
        record["mgo_4x4_size_transfer_test"] = metric_block(
            raw["extra_tests"]["mgo_4x4_size_transfer"]
        )
    if parser_kind in {"nequip", "tece"} and "mgo_4x4_size_transfer_test" in raw:
        record["mgo_4x4_size_transfer_test"] = metric_block(
            raw["mgo_4x4_size_transfer_test"]
        )
    records.append(record)


def markdown(records: list[dict[str, Any]], missing: list[str]) -> str:
    lines = [
        "# MS25 solid-state external-model benchmark",
        "",
        "All values use the official Set1 split. Checkpoints are selected by full-validation Force MAE; test does not select. Energy is meV/atom and force is meV/Å.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Model | Parameters | E MAE | E RMSE | F MAE | F RMSE | Selected step |",
                "|:--|--:|--:|--:|--:|--:|--:|",
            ]
        )
        for row in sorted((item for item in records if item["dataset"] == dataset), key=lambda item: item["model"]):
            values = row["test"]
            lines.append(
                f"| {row['model']} | {row['trainable_parameters']:,} | "
                f"{values['energy_mae']:.2f} | {values['energy_rmse']:.2f} | "
                f"{values['force_mae']:.2f} | {values['force_rmse']:.2f} | "
                f"{row['selected_step']:,} |"
            )
        lines.append("")
    transfer = [row for row in records if "mgo_4x4_size_transfer_test" in row]
    if transfer:
        lines.extend(
            [
                "## MgO 2×2 → 4×4 size transfer",
                "",
                "| Model | E MAE | E RMSE | F MAE | F RMSE |",
                "|:--|--:|--:|--:|--:|",
            ]
        )
        for row in sorted(transfer, key=lambda item: item["model"]):
            values = row["mgo_4x4_size_transfer_test"]
            lines.append(
                f"| {row['model']} | {values['energy_mae']:.2f} | "
                f"{values['energy_rmse']:.2f} | {values['force_mae']:.2f} | "
                f"{values['force_rmse']:.2f} |"
            )
        lines.append("")
    if missing:
        lines.extend(["## Pending artifacts", ""] + [f"- `{path}`" for path in missing] + [""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    a100 = args.raw_root / "a100"
    gpu = args.raw_root / "gpu4090"
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset in DATASETS:
        for mode, label in (
            ("phaseoff", "MACE-ICTC C128"),
            ("final", "MACE-CHORUS-Final C128 R16"),
            ("persistent", "MACE-CHORUS-Persistent C128 R16"),
        ):
            root = a100 / "mace_ictc_family" / dataset / mode
            append_record(
                records,
                missing,
                path=root / "validation_force_selected_eval" / "metrics.json",
                transfer_path=(root / "mgo_4x4_size_transfer_eval" / "metrics.json") if dataset == "mgo-2x2" else None,
                dataset=dataset,
                model=label,
                family="MACE-ICTC",
                parser_kind="chorus",
            )
        native_root = a100 / "native_mace" / dataset
        append_record(
            records,
            missing,
            path=native_root / "validation_force_selected_eval" / "metrics.json",
            transfer_path=(native_root / "mgo_4x4_size_transfer_eval" / "metrics.json") if dataset == "mgo-2x2" else None,
            dataset=dataset,
            model="MACE-SH C128",
            family="MACE-SH",
            parser_kind="native",
        )
        for scope, label in (
            ("off", "NequIP-SH C84"),
            ("final", "NequIP-CHORUS-Final C84 R16"),
            ("persistent", "NequIP-CHORUS-Persistent C84 R16"),
        ):
            append_record(
                records,
                missing,
                path=a100 / "nequip_family" / dataset / scope / "metrics.json",
                dataset=dataset,
                model=label,
                family="NequIP-SH",
                parser_kind="nequip",
            )
        for channels in (32, 48):
            append_record(
                records,
                missing,
                path=gpu / "dpa4" / dataset / f"c{channels}_mix3_fp32_compiled" / "full_val_force_mae_eval" / "metrics.json",
                dataset=dataset,
                model=f"DPA-4 C{channels}",
                family="DPA-4",
                parser_kind="dpa",
            )
        for channels in (36, 48):
            append_record(
                records,
                missing,
                path=gpu / "tece" / dataset / f"c{channels}" / "metrics.json",
                dataset=dataset,
                model=f"TECE C{channels}",
                family="TECE",
                parser_kind="tece",
            )

    result = {
        "benchmark": "MS25 solid-state external-model comparison",
        "complete": not missing,
        "record_count": len(records),
        "expected_record_count": len(DATASETS) * 11,
        "records": records,
        "missing_artifacts": missing,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(markdown(records, missing))
    if args.require_complete and missing:
        raise SystemExit(f"missing {len(missing)} expected artifacts")


if __name__ == "__main__":
    main()
