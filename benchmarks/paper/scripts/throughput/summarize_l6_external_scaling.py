#!/usr/bin/env python3
"""Merge the external L6 scan with the canonical CHORUS-O2U1 L6 artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_external(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        row = dict(payload["rows"][0])
        row.update(
            {
                "model": (
                    "DPA4-C32-L6-m1"
                    if payload["engine"] == "dpa4"
                    else "TECE-C36-L6-m2"
                ),
                "angular_configuration": json.dumps(
                    payload["angular_configuration"], sort_keys=True
                ),
                "backend": payload["backend"],
                "state_source": payload["state_source"],
                "graph_build_in_timing": payload["graph_build_in_timing"],
            }
        )
        rows.append(row)
    return rows


def load_o2u1(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows: list[dict] = []
    for source in payload["rows"]:
        if source["family"] != "o2u1" or source["lmax"] != 6:
            continue
        row = dict(source)
        row.update(
            {
                "model": "CHORUS-O2U1-L6-m2",
                "angular_configuration": json.dumps(
                    {
                        "neutral_lmax": 3,
                        "tail_lmax": 6,
                        "tail_mmax": 2,
                        "tail_message_mmax": 1,
                    },
                    sort_keys=True,
                ),
                "backend": "MakeFX/Inductor",
                "state_source": "random-initialization",
                "graph_build_in_timing": False,
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--o2u1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_external(args.raw_dir) + load_o2u1(args.o2u1)
    rows.sort(key=lambda row: (row["task"], row["natoms"], row["model"]))
    payload = {
        "protocol": {
            "gpu": "NVIDIA A100-SXM4-40GB",
            "dtype": "float32",
            "tf32": False,
            "directed_edges": "32N target",
            "inference": "energy plus conservative forces",
            "train": "complete energy+force AdamW update",
            "steady_state_excludes_compile_and_graph_preparation": True,
            "warning": (
                "The three L6 labels are not identical workloads: O2U1 keeps "
                "an L3 neutral carrier and adds an 8-channel compact L4-L6 tail; "
                "TECE and DPA4 raise their full descriptor to L6. DPA4 also "
                "includes its internal neighbor-list path in timed calls."
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
