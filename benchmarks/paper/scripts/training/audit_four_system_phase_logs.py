#!/usr/bin/env python3
"""Audit and summarize CHORUS four-system training logs.

One row is emitted per physical run.  Checkpoint-aligned metrics are taken from
the epoch with minimum validation loss; force and energy MAE are never selected
independently from different epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


VAL_RE = re.compile(
    r"\[epoch\s+(?P<epoch>\d+).*?val loss=(?P<loss>[0-9.eE+-]+)"
    r"\s+Frmse=(?P<frmse>[0-9.eE+-]+)\s+Ermse=(?P<ermse>[0-9.eE+-]+)"
    r"\s+Fmae=(?P<fmae>[0-9.eE+-]+)\s+Emae=(?P<emae>[0-9.eE+-]+)"
)
PARAM_RE = re.compile(r"\bparams=(\d+)")
SEED_RE = re.compile(r"_seed(\d+)_epochs(\d+)\.log$")


def classify(name: str, campaign: str) -> str:
    if "phase_full_l_softplus" in name:
        return "full_u1"
    if "phase_diagonal_full_l" in name:
        return "diagonal_jk"
    if "phase_cartesian_full_l" in name:
        return "cartesian_two_real"
    if "phase_signed_full_l" in name:
        return "signed_real"
    if "phase_positive_full_l" in name:
        return "positive_gate"
    if "phase_radial_full_l" in name:
        return "radial_only_phase"
    if "attention" in name:
        return "density_attention" if "density_attention_final" in campaign else "legacy_attention"
    if "bridge_u" in name:
        return "baseline"
    return "other"


def system_of(name: str) -> str:
    for key in ("revised_aspirin", "revised_benzene", "revised_ethanol", "cheng_water"):
        if name.startswith(key):
            return key
    return "unknown"


def parse_log(path: Path, raw_root: Path) -> dict[str, object] | None:
    match = SEED_RE.search(path.name)
    if match is None:
        return None
    seed, requested_epochs = map(int, match.groups())
    campaign = path.relative_to(raw_root).parts[0]
    rows: list[dict[str, float | int]] = []
    params = None
    done_marker = False
    for line in path.open(errors="ignore"):
        if "done. best loss" in line:
            done_marker = True
        if params is None and (pm := PARAM_RE.search(line)):
            params = int(pm.group(1))
        # Every validation record is printed twice; retain timestamped logger rows only.
        if not line[:4].isdigit():
            continue
        vm = VAL_RE.search(line)
        if vm:
            rows.append(
                {
                    "epoch": int(vm.group("epoch")),
                    **{key: float(vm.group(key)) for key in ("loss", "frmse", "ermse", "fmae", "emae")},
                }
            )
    if not rows:
        return None
    selected = min(rows, key=lambda row: float(row["loss"]))
    final = rows[-1]
    complete = int(final["epoch"]) == requested_epochs - 1
    return {
        "campaign": campaign,
        "system": system_of(path.name),
        "mode": classify(path.name, campaign),
        "seed": seed,
        "requested_epochs": requested_epochs,
        "completed_epochs": int(final["epoch"]) + 1,
        "complete": complete,
        "done_marker": done_marker,
        "execution": "makefx" if "makefx" in path.name else "eager",
        "parameters": params,
        "selected_epoch": int(selected["epoch"]),
        "selected_loss": selected["loss"],
        "selected_force_rmse": selected["frmse"],
        "selected_energy_rmse": selected["ermse"],
        "selected_force_mae": selected["fmae"],
        "selected_energy_mae": selected["emae"],
        "final_epoch": int(final["epoch"]),
        "final_loss": final["loss"],
        "final_force_mae": final["fmae"],
        "final_energy_mae": final["emae"],
        "log_path": str(path),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not row["complete"]:
            continue
        key = (
            row["campaign"], row["system"], row["mode"],
            row["requested_epochs"], row["execution"],
        )
        groups[key].append(row)
    result = []
    for key, group in sorted(groups.items()):
        campaign, system, mode, epochs, execution = key
        item: dict[str, object] = {
            "campaign": campaign,
            "system": system,
            "mode": mode,
            "epochs": epochs,
            "execution": execution,
            "n_complete": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda x: int(x["seed"]))),
        }
        for metric in ("selected_loss", "selected_force_mae", "selected_energy_mae"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def aggregate_protocol(
    rows: list[dict[str, object]], *, epochs: int, execution: str
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if (
            bool(row["complete"])
            and int(row["requested_epochs"]) == epochs
            and str(row["execution"]) == execution
        ):
            groups[(str(row["system"]), str(row["mode"]))].append(row)
    result = []
    for (system, mode), group in sorted(groups.items()):
        item: dict[str, object] = {
            "system": system,
            "mode": mode,
            "epochs": epochs,
            "execution": execution,
            "n_complete": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda x: int(x["seed"]))),
        }
        for metric in ("selected_loss", "selected_force_mae", "selected_energy_mae"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for path in sorted(args.raw_root.rglob("*.log"))
        if (row := parse_log(path, args.raw_root)) is not None
    ]
    if not rows:
        raise SystemExit("no parseable training logs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", rows)
    aggregates = aggregate(rows)
    write_csv(args.output_dir / "aggregates_by_campaign.csv", aggregates)
    write_csv(
        args.output_dir / "aggregates_300_eager.csv",
        aggregate_protocol(rows, epochs=300, execution="eager"),
    )
    write_csv(
        args.output_dir / "aggregates_500_makefx.csv",
        aggregate_protocol(rows, epochs=500, execution="makefx"),
    )
    quality = {
        "parsed_runs": len(rows),
        "complete_runs": sum(bool(row["complete"]) for row in rows),
        "incomplete_runs": sum(not bool(row["complete"]) for row in rows),
        "systems": sorted({str(row["system"]) for row in rows}),
        "campaigns": sorted({str(row["campaign"]) for row in rows}),
        "selection_rule": "minimum validation loss; all reported MAEs from the same epoch",
    }
    (args.output_dir / "quality_summary.json").write_text(json.dumps(quality, indent=2) + "\n")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
